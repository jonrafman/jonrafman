# anima

A speaking sculpture. An abstract presence in a room that hears a visitor,
decides what to say, says it aloud, and renders the whole performance as light.
No face, nothing moving. It runs entirely offline.

Its vocabulary is closed. It can only use words William Blake wrote, and it
cannot break that rule — not because it is asked not to, but because at the
moment of choosing each word, the others do not exist.

---

## Quick start

Clone it and it runs. No model, no microphone, no speaker, no light.

```bash
pip install pyyaml numpy
python -m anima.main --check     # verify the setup
python -m anima.main --once      # type at it
```

That is the whole piece working, with a fake at every stage: typed input,
canned replies, near-silent audio, a light drawn as a bar in the terminal.
Each real component switches on one line at a time in `config.yaml`.

---

## How it works

```
dormant ──▶ trigger ──▶ attending ──▶ thinking ──▶ speaking ──▶ attending
   ▲                                                                │
   └──────────────── visitor stops answering ◀──────────────────────┘
```

| Stage | Module | Backends |
|---|---|---|
| Someone is there | `trigger.py` | keyboard, always, button, PIR |
| Hearing | `ears.py` | typed, faster-whisper |
| Deciding | `brain/` | scripted, llama.cpp, ollama, claude |
| Speaking | `voice.py` | silent, espeak, piper, kokoro, chatterbox, elevenlabs |
| Being seen | `light.py` + `presence.py` | console, PWM, addressable |

Every stage sits behind an interface with a working fake, so the loop runs on
a laptop with none of the hardware present.

### Two decisions that shape everything

**Sentences stream.** The brain yields whole sentences as they are generated,
and each is spoken while the rest of the reply is still being written. Waiting
for a complete reply before speaking is what makes a talking object feel dead.

**The light has states.** With no face and no motion, four light states carry
everything a body would: *dormant*, *attending*, *thinking*, *speaking*. The
thinking state is the one that earns its keep — model generation plus speech
synthesis is several seconds of dead air, and giving that time its own visible
behaviour turns it into deliberation. The visitor does not experience latency,
they experience consideration.

---

## The closed vocabulary

This is the part that matters.

Asking a model to restrict its vocabulary does not work. It has no way to
check itself against a word list mid-sentence, and it drifts. But a local model
exposes its sampler, and a grammar compiled from a lexicon masks every token
that could not continue a legal string. A word Blake never wrote has
probability zero — structurally, not statistically.

```bash
./corpus/fetch.sh                                                # get the texts
python -m anima.lexicon build corpus/blake-poems.txt \
    -o lexicon/blake.json --modern-grammar
python -m anima.grammar lexicon/blake.json -o lexicon/blake.gbnf
```

Then point `brain.llamacpp.grammar` at the `.gbnf`. That is the whole
mechanism.

### What the vocabulary does to the character

*Songs of Innocence and of Experience* plus *The Book of Thel* gives **1,519
word forms**. The conversational machinery survives intact — every pronoun,
core verb, question word, negation and connective. What vanishes is the modern
and the mundane, and those absences shape the character more than any authored
decision:

- **There is no "yes."** Blake never uses it. The sculpture can deny but not
  affirm, so agreement has to become something else.
- **There is no "remember"** — but *forget* and *forgot* are both there.
- No *computer*, *machine*, *program*, *okay*, *hello*, *people*, *something*.
  Asked whether it is a computer, it cannot form the word. Incapacity, not
  evasion.
- No *today*, *tomorrow*, *yesterday*, *ago*. It cannot place itself in time
  relative to now.

Write the persona inside the lexicon too, or the model is shown a voice it is
then forbidden from using:

```bash
python -m anima.lexicon check lexicon/blake.json persona/watchman.md
```

Other lexicons are compiled and ready in `lexicon/` — `ecclesiastes` (891
forms), `revelation` (1,290), `kjv` (12,797). Swapping is one config line: the
corpus is data, not code.

### On spelling

Blake's orthography would defeat any speech engine — `thro` read as "throw",
`shewd` as "shooed". `--modernise` fixes the spellings at lexicon-build time,
and `--modern-grammar` also converts *thou/doth/hath* into *you/does/has*.
Both are spelling and grammar changes, not vocabulary ones: the same words,
written so a voice can say them.

---

## Hardware

Compute goes in the plinth or an adjacent room; the sculpture holds only the
light, a speaker, and a microphone. With no servos in the piece, **the loudest
mechanical thing in the room becomes the computer's fan** — which is the
argument for decoupling them.

- **Compute** — anything that runs a local model well. Memory *bandwidth*
  governs speaking speed far more than core count, because generation is
  bandwidth-bound. Apple Silicon with high unified-memory bandwidth suits this
  particularly well. A hard vocabulary constraint means model size matters
  less than usual; the lexicon does much of the aesthetic work.
- **Microphone** — the single biggest determinant of whether this works. A mic
  three feet away in a gallery transcribes the room, not the person. A
  telephone handset solves four problems at once: the mic is at their mouth,
  the speaker at their ear, lifting it is the trigger, and it tells the visitor
  what to do without a sign.
- **Light** — one dimmable channel, an addressable ring behind diffusion, or a
  DMX fixture. Gamma correction is applied throughout; without it a smooth fade
  looks like a stutter.

---

## Commands

```bash
python -m anima.main --check              # verify every stage
python -m anima.main                      # run the piece
python -m anima.main --once               # one conversation
python -m anima.main --say "..."          # bench-test voice and light
python -m anima.main --states             # cycle light states for tuning
python -m anima.devices                   # list audio devices

python -m anima.lexicon build ... -o ...  # compile a vocabulary
python -m anima.lexicon stats <lexicon>   # summarise it
python -m anima.lexicon check <lex> <txt> # what a text uses that it must not
python -m anima.grammar <lexicon> -o ...  # compile the sampling grammar
python -m anima.pronounce <lexicon>       # find spellings a voice will mangle

python -m anima.probe -o runs/a.json      # run the question battery
python -m anima.probe --compare a b       # diff two runs
```

### The run worth doing first

```bash
python -m anima.probe --label "prompt only" -o runs/off.json   # grammar: ""
python -m anima.probe --label "grammar"     -o runs/on.json    # grammar set
python -m anima.probe --compare runs/off.json runs/on.json
```

Same 25 questions, grammar off then on. The difference in lexicon compliance
is exactly what sampler-level constraint is worth over prompting — the number
that justifies running a local model at all.

For calibration: a careful human writing deliberately inside this lexicon
still lands around 87%. Expect prompting alone to do worse.

---

## Layout

```
anima/          the piece
  main.py       the show loop
  brain/        scripted · llamacpp · ollama · claude
  light.py      drivers and the speech envelope
  presence.py   dormant · attending · thinking · speaking
  lexicon.py    compiling a closed vocabulary
  grammar.py    lexicon -> GBNF sampling grammar
  pronounce.py  archaic spelling -> speakable
  probe.py      the question battery
corpus/         source texts
lexicon/        compiled vocabularies and grammars
persona/        who the character is
probes/         the question battery
config.yaml     every dial, commented
```
