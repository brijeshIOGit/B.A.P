"""Mask secrets and personal data (R1).

mask() runs at ingress, before logs, before every model call and on ticket notes.
The order matters: keys first, then emails, cards, Aadhaar, phones, so one pattern
does not swallow another.
"""

import re

# Secret keys. 16+ characters after the prefix so words like "sk-learn" survive.
SECRET_RE = re.compile(
    r"\b(?:"
    r"sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{16,}"  # OpenAI, Anthropic
    r"|gsk_[A-Za-z0-9]{16,}"  # Groq
    r"|xai-[A-Za-z0-9]{16,}"  # xAI
    r"|(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}"  # Stripe
    r"|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"  # GitHub
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}"  # AWS access key id
    r"|AIza[0-9A-Za-z_\-]{35}"  # Google API key
    r"|xox[abprs]-[A-Za-z0-9\-]{10,}"  # Slack
    r")")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# 13-19 digits with optional spaces or dashes. Only masked when the Luhn checksum is valid.
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
# 12 digits, first digit 2-9, optional same separator between groups of 4.
AADHAAR_RE = re.compile(r"(?<!\d)[2-9]\d{3}([ -]?)\d{4}\1\d{4}(?!\d)")
# Indian mobile: optional +91 / 0, then 10 digits starting 6-9.
PHONE_RE = re.compile(r"(?<!\d)(?:\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
# Any other international number: a + country code, then 7-14 more digits in groups.
INTL_PHONE_RE = re.compile(r"(?<![\w+])\+\d{1,3}(?:[\s.-]?\(?\d{1,4}\)?){2,5}(?!\d)")
# A bare number right after one of these is a value in code ("return 9876543210", "x == ..."),
# not a phone. A plain "=" is NOT on the list, so "phone = 9876543210" is still masked (fail closed).
CODE_VALUE_BEFORE = re.compile(r"(?:\breturn|==|!=|<=|>=|[<>*/%])\s*$")
PLACEHOLDERS = {"[SECRET]": "secret", "[EMAIL]": "email", "[CARD]": "card", "[AADHAAR]": "aadhaar",
                "[PHONE]": "phone"}


def luhn_ok(digits: str) -> bool:
    """True when the digit string passes the Luhn checksum used by card numbers."""
    total, double = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if double:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        double = not double
    return total % 10 == 0


def _mask_cards(text: str, found: list) -> str:
    """Replace Luhn-valid card numbers with [CARD]."""

    def replace(match):
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            found.append("card")
            return "[CARD]"
        return match.group()

    return CARD_RE.sub(replace, text)


def _mask_phones(text: str, found: list) -> str:
    """Replace phone numbers with [PHONE], except a bare number used as a value in code."""

    def replace(match):
        raw = match.group()
        bare = raw.isdigit()  # no +91, no spaces: could be a constant like 9876543210
        if bare and CODE_VALUE_BEFORE.search(text[max(0, match.start() - 12):match.start()]):
            return raw
        found.append("phone")
        return "[PHONE]"

    text = PHONE_RE.sub(replace, text)
    return _sub(INTL_PHONE_RE, "[PHONE]", "phone", text, found)


def _sub(pattern, token: str, kind: str, text: str, found: list) -> str:
    """Replace every match of pattern with token and note the kind once."""
    if pattern.search(text):
        found.append(kind)
        text = pattern.sub(token, text)
    return text


def mask(text: str) -> tuple[str, list]:
    """Return (masked_text, kinds_found). Safe to call twice: tokens never match again."""
    if not isinstance(text, str) or not text:
        return text, []
    found: list = []
    text = _sub(SECRET_RE, "[SECRET]", "secret", text, found)
    text = _sub(EMAIL_RE, "[EMAIL]", "email", text, found)
    text = _mask_cards(text, found)
    text = _sub(AADHAAR_RE, "[AADHAAR]", "aadhaar", text, found)
    text = _mask_phones(text, found)
    # Text that was masked earlier (e.g. by graph.user_input) still reports what it held.
    found += [kind for token, kind in PLACEHOLDERS.items() if token in text]
    return text, sorted(set(found))


def mask_text(text: str) -> str:
    """mask() without the list of kinds."""
    return mask(text)[0]


def mask_code(code: str) -> str:
    """Mask generated code: secrets, cards and Aadhaar only.

    Made-up emails such as user@example.com stay, because an email validator's
    tests need them and they cannot be user data (the model never saw any).
    """
    if not code:
        return code
    found: list = []
    code = _sub(SECRET_RE, "[SECRET]", "secret", code, found)
    code = _mask_cards(code, found)
    return _sub(AADHAAR_RE, "[AADHAAR]", "aadhaar", code, found)
