"""anima -- a conversational animatronic figure.

A local voice loop for a physical character: it hears a visitor, decides what to
say, says it, and moves its mouth in time with the words.

Every stage is swappable through config.yaml, and every stage has a fake, so the
whole thing runs on a laptop with no microphone, no speaker, no servo, and no
model. Build the puppet's nervous system first; decide who it is later.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
