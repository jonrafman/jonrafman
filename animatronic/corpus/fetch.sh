#!/usr/bin/env bash
# Download Blake's writings as plain text, for building the closed lexicon.
#
#   cd animatronic && ./corpus/fetch.sh
#   python -m anima.lexicon build corpus/*.txt -o lexicon/blake.json
#   python -m anima.lexicon stats lexicon/blake.json
#
# Project Gutenberg only, because those texts are clean, proofread, plain, and
# unambiguously free to take. They are not the whole oeuvre -- the long
# prophetic books need the manual sources listed at the bottom.
#
# THE ONE RULE: only Blake's own words go in this directory.
#
# Searching for Blake turns up far more books *about* him than *by* him --
# Symons, Langridge, Garnett, and a century of criticism. Dropping one of those
# in here would quietly hand the sculpture a Victorian biographer's vocabulary
# and you would never trace the wrongness back to its source. Prefaces,
# footnotes, and scholarly introductions in editions of Blake are the same
# problem in miniature; strip them where you can.

set -euo pipefail

cd "$(dirname "$0")"

# Gutenberg serves plain text at a predictable path. Format: id|filename
BOOKS=(
  "574|poems-of-william-blake"                 # Songs of Innocence & Experience, Book of Thel
  "1934|songs-of-innocence-and-experience"     # fuller edition of the Songs
  "45315|marriage-of-heaven-and-hell"          # The Marriage of Heaven and Hell
)

echo "Fetching Blake from Project Gutenberg..."

for entry in "${BOOKS[@]}"; do
  id="${entry%%|*}"
  name="${entry##*|}"
  out="${name}.txt"

  if [[ -s "$out" ]]; then
    echo "  have    $out"
    continue
  fi

  echo "  get     $out (gutenberg #$id)"
  if curl -fsSL --retry 3 --retry-delay 2 \
      -A "anima-lexicon/0.1 (personal art project; contact via repo)" \
      "https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt" -o "$out"; then
    # Be a good citizen: Gutenberg throttles rapid automated requests.
    sleep 2
  else
    echo "  FAILED  $out -- fetch it by hand from https://www.gutenberg.org/ebooks/${id}"
    rm -f "$out"
  fi
done

echo
echo "Downloaded:"
ls -1 ./*.txt 2>/dev/null | sed 's/^/  /' || echo "  (nothing)"

cat <<'NOTE'

--------------------------------------------------------------------------
STILL MISSING: the long prophetic books.

Gutenberg does not carry Milton, Jerusalem, or The Four Zoas as plain text,
and those are a large share of Blake's total vocabulary -- the strangest
share, and the reason to do this at all. Add them by hand:

  Wikisource (cleanest; use the "Download as plain text" option per work)
    https://en.wikisource.org/wiki/Author:William_Blake

  Internet Archive -- poetical works incl. the minor prophetic books,
  public domain scan with full text available
    https://archive.org/details/aca5924.0001.001.umich.edu

  Internet Archive -- Keynes, "The complete writings of William Blake with
  variant readings". The most complete text there is. Access may be
  lending-only rather than open download; check before relying on it.
    https://archive.org/details/completewritings0000blak_y4c4

  The William Blake Archive -- scholarly diplomatic transcriptions of all
  nineteen illuminated books. No bulk export, but authoritative.
    https://www.blakearchive.org

Two practical notes:

  Duplicates cost nothing. The lexicon is a set of word forms, so overlapping
  editions only ever improve coverage. Take everything you can get.

  OCR'd scans invent words. A page image read badly yields "tbe", "thc",
  "vvith" -- and every one of those becomes a word the sculpture is permitted
  to say. For any text that came from a scan rather than from proofread
  transcription, build with --min-count 2 to drop forms appearing only once.
  You lose genuine rare words, which in a poet hurts; weigh it per source.
--------------------------------------------------------------------------
NOTE
