"""The three conditioning set-ups compared in the paper (Sect. 3.1):

  chat          dialogue history only
  persona       persona summary only (plus the user's last message to answer)
  persona+chat  persona summary and dialogue history
"""
from __future__ import annotations

from typing import Dict, List

from . import CONDITIONS

SYSTEM = "You are a helpful conversational assistant. Reply with a single, natural next turn."


def persona_block(persona: Dict) -> str:
    return (f"User persona: {persona['summary']}\n"
            f"Typical interests: {', '.join(persona.get('keywords', [])[:6])}")


def history_messages(history: List[str]) -> List[Dict[str, str]]:
    """PersonaChat histories alternate user / assistant, ending with the user's turn."""
    n = len(history)
    return [{"role": "user" if (n - 1 - i) % 2 == 0 else "assistant", "content": h} for i, h in enumerate(history)]


def build_messages(condition: str, history: List[str], persona: Dict | None) -> List[Dict[str, str]]:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if condition != "chat" and persona is None:
        raise ValueError(f"condition {condition!r} needs a persona")
    if condition == "chat":
        return [{"role": "system", "content": SYSTEM}] + history_messages(history)
    system = f"{SYSTEM}\nTailor your reply to this user.\n{persona_block(persona)}"
    if condition == "persona":
        return [{"role": "system", "content": system}, {"role": "user", "content": history[-1]}]
    return [{"role": "system", "content": system}] + history_messages(history)
