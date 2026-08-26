#!/usr/bin/env bash
# Set up anima on an Apple Silicon Mac.
#
#   ./setup-mac.sh              core only -- the piece runs, with fakes
#   ./setup-mac.sh --full       plus local model, speech-to-text and voice
#
# Deliberately incremental. The core install is two pure-Python packages and
# gets you a working loop in seconds; --full pulls hundreds of megabytes of
# machine learning wheels. Verify the loop before you wait on the download.
#
# What this does NOT do is fetch model weights. Those are gigabytes, the good
# ones change every few months, and which to use is a decision rather than a
# default. The script tells you where to put them.

set -euo pipefail
cd "$(dirname "$0")"

FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m  ! %s\033[0m\n' "$*"; }

# --- sanity ------------------------------------------------------------------

if [[ "$(uname -s)" != "Darwin" ]]; then
  warn "This script is for macOS. On Linux install the same packages with pip."
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  warn "Not Apple Silicon. Everything still works, but llama.cpp will run on"
  warn "the CPU without Metal, which is dramatically slower."
fi

if ! command -v python3 >/dev/null; then
  echo "python3 not found. Install it from python.org or: brew install python" >&2
  exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
say "Python $PY_VERSION"

# --- virtual environment -----------------------------------------------------

if [[ ! -d .venv ]]; then
  say "Creating .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip

# --- core --------------------------------------------------------------------

say "Installing core (pyyaml, numpy)"
pip install --quiet pyyaml numpy

say "Checking the loop runs"
python -m anima.main --check

if [[ $FULL -eq 0 ]]; then
  cat <<'NOTE'

Core install done. The piece runs now, with a fake at every stage:

    source .venv/bin/activate
    python -m anima.main --once

When you want the real thing:

    ./setup-mac.sh --full

NOTE
  exit 0
fi

# --- audio -------------------------------------------------------------------

say "Installing audio (sounddevice, soundfile)"
if command -v brew >/dev/null; then
  brew list portaudio >/dev/null 2>&1 || brew install portaudio
  # Kokoro's phonemiser needs espeak-ng present even on macOS.
  brew list espeak-ng >/dev/null 2>&1 || brew install espeak-ng
else
  warn "Homebrew not found. Install it from brew.sh, then re-run --full."
  warn "Without portaudio there is no microphone and no speaker."
fi
pip install --quiet sounddevice soundfile

# --- brain -------------------------------------------------------------------

say "Installing llama-cpp-python (compiles with Metal; this takes a while)"
pip install --quiet llama-cpp-python

# --- ears --------------------------------------------------------------------

say "Installing faster-whisper"
pip install --quiet faster-whisper

# --- voice -------------------------------------------------------------------

say "Installing Kokoro (Apache 2.0, ~82M params, fast on Apple Silicon)"
pip install --quiet kokoro || warn "Kokoro failed; see README for alternatives."

cat <<'NOTE'

  Chatterbox (MIT, clones a voice from a few seconds of audio) is the other
  option worth trying, but it is a large install and only wanted if a specific
  voice matters to the piece:

      pip install chatterbox-tts

NOTE

# --- lexicon -----------------------------------------------------------------

say "Building the Blake lexicon and grammar"
if [[ ! -f corpus/blake-poems.txt ]]; then
  warn "corpus/blake-poems.txt missing -- run ./corpus/fetch.sh first."
else
  python -m anima.lexicon build corpus/blake-poems.txt \
      -o lexicon/blake.json --modern-grammar
  python -m anima.grammar lexicon/blake.json -o lexicon/blake.gbnf
  python -m anima.lexicon check lexicon/blake.json persona/watchman.md
fi

# --- what is left ------------------------------------------------------------

cat <<'NOTE'

Installed. One thing remains, and it is a decision rather than a step:

  1. Get a GGUF model. Somewhere reputable on Hugging Face; a 4-bit quant of
     a mid-size instruct model is the place to start. Put it anywhere and set:

         brain:
           backend: llamacpp
           llamacpp:
             model: /path/to/your-model.gguf

  2. Turn on the real voice:

         voice:
           backend: kokoro

  3. Prove the grammar is actually in force -- this loads the model and
     generates a probe reply, so it fails loudly rather than silently:

         python -m anima.main --check

  4. Then the run that matters. Same questions, grammar off and on:

         python -m anima.probe --label "prompt only" -o runs/off.json
         # set brain.llamacpp.grammar back to lexicon/blake.gbnf
         python -m anima.probe --label "grammar"     -o runs/on.json
         python -m anima.probe --compare runs/off.json runs/on.json

     The gap in lexicon compliance is what the constraint is worth.

NOTE
