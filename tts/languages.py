"""Language identifiers shared by the core, API and agent layers."""

from __future__ import annotations

LANG_TELUGU = "Telugu"
LANG_ENGLISH = "English"
LANG_MIXED = "Telugu + English"
LANG_ROMANIZED = "Romanized Telugu"

LANGUAGES = (LANG_TELUGU, LANG_ENGLISH, LANG_MIXED, LANG_ROMANIZED)

# API-friendly aliases. They are intentionally explicit; an unknown value must
# fail rather than silently falling back to the most permissive language mode.
LANGUAGE_ALIASES = {
    "te": LANG_TELUGU,
    "telugu": LANG_TELUGU,
    "en": LANG_ENGLISH,
    "english": LANG_ENGLISH,
    "mixed": LANG_MIXED,
    "te-en": LANG_MIXED,
    "tenglish": LANG_MIXED,
    "telugu+english": LANG_MIXED,
    "romanized": LANG_ROMANIZED,
    "romanized-telugu": LANG_ROMANIZED,
    "tanglish": LANG_ROMANIZED,
}


def canonical_language(value: str | None, default: str = LANG_TELUGU) -> str:
    if not value:
        return default
    value = value.strip()
    if value in LANGUAGES:
        return value
    alias = LANGUAGE_ALIASES.get(value.lower())
    if alias is None:
        raise ValueError(
            f"Unsupported language {value!r}. Choose one of: {', '.join(LANGUAGES)}."
        )
    return alias
