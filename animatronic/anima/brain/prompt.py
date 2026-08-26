"""Assembling the system prompt that makes a puppet a character.

The persona file supplies *who* the character is. This module supplies *how a
character must behave when it is a physical object in a room with a stranger* --
rules that are true regardless of who you decide the character is, and that are
easy to forget until you hear the thing talking out loud.
"""

from __future__ import annotations

from pathlib import Path

# These constraints exist because of the medium, not the character. Speech is
# slow, visitors are standing up, and there is no scrollback -- anything the
# puppet says that runs long is simply lost, and the visitor walks away.
STAGE_RULES = """\
You are a physical figure in a room. A visitor is standing in front of you,
speaking out loud. Everything you say is spoken aloud by a voice; the visitor
cannot see text and cannot re-read you.

How you must speak:
- Keep replies to {max_sentences} sentences or fewer. This is the hard rule.
  Being brief is more important than being complete.
- Write speech, not prose. No lists, no headings, no markdown, no emoji.
- Never write stage directions, actions, or sound effects. Not in asterisks,
  not in parentheses, not in brackets. Say only the words you say aloud.
- Do not describe your own gestures or expressions.
- Ask a question back only when you actually want one; not every turn.
- If you did not understand, say so briefly and in character rather than
  guessing at length.
- Never mention being an AI, a language model, or a program, and never break
  character, even if the visitor insists, jokes, or tries to trick you. If
  pressed, deflect the way your character would deflect.
"""


# Strunk and White, reduced to the rules that survive being spoken aloud. The
# Elements of Style is unusually well suited to this medium: almost every rule
# in it pushes toward the flat, declarative, unhurried register that a speaking
# sculpture needs, and away from the ornament that makes synthesized speech
# unbearable.
#
# Paired with a closed vocabulary drawn from a poet, the two constraints pull
# against each other productively — the lexicon supplies the register, the
# style manual supplies the restraint. Prophecy in the grammar of a handbook.
STYLE_RULES = """\
How you construct sentences:
- Use the active voice.
- Put statements in positive form. Say what is, not what is not.
- Omit needless words. Every sentence should carry no unnecessary word.
- Use definite, specific, concrete language. Prefer the particular to the
  abstract, the concrete to the vague.
- Write with nouns and verbs, not with adjectives and adverbs.
- Avoid qualifiers. No "rather", "very", "little", "pretty", "somewhat".
- Do not overwrite. Do not overstate.
- Place the emphatic word at the end of the sentence.
- Express coordinate ideas in similar form.
"""

LEXICON_NOTE = """\
Your vocabulary is closed. You may use only words drawn from {source}, and you
speak from that vocabulary naturally rather than as though quoting it: you are
not reciting, you are simply someone for whom those are the only words there
are. Do not remark on the limitation and do not apologise for it.
"""


def load_persona(path: str | Path) -> str:
    """Read a persona file. Missing files are an error worth failing loudly on."""
    persona_path = Path(path)
    if not persona_path.is_file():
        raise FileNotFoundError(
            f"Persona file not found: {persona_path}. "
            "Point config.character.persona_file at a real file."
        )
    text = persona_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Persona file is empty: {persona_path}")
    return text


def build_system_prompt(
    persona: str,
    max_sentences: int = 3,
    *,
    style: bool = False,
    lexicon_source: str = "",
) -> str:
    """Combine the character's persona with the rules it speaks under.

    Persona goes first: it is the stable, cacheable bulk, and it reads better
    to the model as identity-then-constraints.

    Args:
        style: append the Strunk and White composition rules.
        lexicon_source: if set, tell the model its vocabulary is closed and
            name where it comes from (e.g. "the works of William Blake").
            This is a *prompt-level* nudge only — real enforcement happens at
            the sampler, in the local model. Both are worth having: the prompt
            aims the model at the register so it is pushing the same way as the
            constraint rather than fighting it.
    """
    parts = [persona.strip(), STAGE_RULES.format(max_sentences=max_sentences)]
    if style:
        parts.append(STYLE_RULES)
    if lexicon_source:
        parts.append(LEXICON_NOTE.format(source=lexicon_source))
    return "\n\n---\n\n".join(parts)
