"""Strip an email down to what the sender actually wrote.

A reply carries the whole thread below it, plus a signature and often a legal
disclaimer. Those are the longest part of many emails and none of it is the
message. Cutting them in code is free and deterministic; leaving them in
spends the window on quoted text and shifts the decision toward whatever the
thread said before.

    clean(body)                     -> the sender's own text
    as_state(sender, subject, body) -> the state string to pass to decide()
"""

from __future__ import annotations

import re

# "On Tue, 3 Jun 2025 at 14:02, Someone <a@b.com> wrote:" and its variants.
_REPLY_HEADER = re.compile(
    r"^\s*(on\s.{0,120}?\swrote:|-{2,}\s*original message\s*-{2,}|"
    r"-{2,}\s*forwarded message\s*-{2,}|"
    r"from:\s.+?\bsent:\s|_{5,})",
    re.I | re.M | re.S)

# A signature block starts at a line that is exactly "--" or "-- ".
_SIG = re.compile(r"^--\s*$", re.M)

_DISCLAIMER = re.compile(
    r"(this (e-?mail|message) (and any attachments )?is (intended|confidential)"
    r"|confidentiality notice"
    r"|if you (are not|have received) th(is|e intended)"
    r"|please consider the environment before printing"
    r"|unsubscribe\b.{0,80}$)",
    re.I | re.S)

_QUOTED_LINE = re.compile(r"^\s*>", re.M)


def clean(body: str, keep_quoted: bool = False) -> str:
    """The sender's own text, with the thread, signature and disclaimer cut.

    `keep_quoted` leaves quoted lines in place, which matters when the decision
    is about the thread rather than the reply.
    """
    if not body:
        return ""
    text = str(body).replace("\r\n", "\n").replace("\r", "\n")

    cut = len(text)
    for pattern in (_REPLY_HEADER, _SIG, _DISCLAIMER):
        m = pattern.search(text)
        if m and m.start() < cut:
            cut = m.start()
    text = text[:cut]

    if not keep_quoted:
        text = _QUOTED_LINE.sub("", text)

    # Collapse the runs of blank lines the cuts leave behind.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def as_state(sender: str = "", subject: str = "", body: str = "",
             keep_quoted: bool = False) -> str:
    """The state string, one `key: value` line per field.

    The same shape the rest of this package feeds the model, so an email and a
    ticket look alike to it.
    """
    parts = []
    if sender:
        parts.append(f"from: {str(sender).strip()}")
    if subject:
        parts.append(f"subject: {str(subject).strip()}")
    text = clean(body, keep_quoted=keep_quoted)
    if text:
        parts.append(f"body: {text}")
    return "\n".join(parts)
