"""Speech-aware text frontend for Telugu, English and Telugu-English text.

The model is trained on written text, not on markup, URLs, emoji or raw digits.
This module turns those forms into words that a TTS model can pronounce, without
translating or paraphrasing the user's prose.  The transformation is deliberately
conservative and every rewrite is covered by a unit test.

It is *not* a full linguistic normalizer.  Ambiguous abbreviations and
code-mixed number pronunciation remain explicit warnings rather than guesses.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from tts.languages import LANG_ENGLISH, LANG_MIXED, LANG_ROMANIZED, LANG_TELUGU, canonical_language
from tts.text_normalizer import normalize_text


TELUGU_DIGITS = str.maketrans("౦౧౨౩౪౫౬౭౮౯", "0123456789")
_TELUGU_RANGE = re.compile(r"[\u0C00-\u0C7F]")
_LATIN_RANGE = re.compile(r"[A-Za-z]")

_TELUGU_ONES = (
    "సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు",
    "ఎనిమిది", "తొమ్మిది", "పది", "పదకొండు", "పన్నెండు", "పదమూడు",
    "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పంతొమ్మిది",
)
_TELUGU_TENS = {
    20: "ఇరవై", 30: "ముప్పై", 40: "నలభై", 50: "యాభై",
    60: "అరవై", 70: "డెబ్బై", 80: "ఎనభై", 90: "తొంభై",
}
_ENGLISH_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_ENGLISH_TENS = (
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
)
_TELUGU_MONTHS = (
    "జనవరి", "ఫిబ్రవరి", "మార్చి", "ఏప్రిల్", "మే", "జూన్", "జూలై",
    "ఆగస్టు", "సెప్టెంబర్", "అక్టోబర్", "నవంబర్", "డిసెంబర్",
)
_ENGLISH_MONTHS = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)

_UNIT_WORDS = {
    "te": {
        "km": "కిలోమీటర్లు", "kms": "కిలోమీటర్లు", "m": "మీటర్లు",
        "cm": "సెంటీమీటర్లు", "mm": "మిల్లీమీటర్లు", "kg": "కిలోగ్రాములు",
        "kgs": "కిలోగ్రాములు", "g": "గ్రాములు", "mg": "మిల్లీగ్రాములు",
        "l": "లీటర్లు", "lt": "లీటర్లు", "ml": "మిల్లీలీటర్లు",
        "gb": "గిగాబైట్లు", "mb": "మెగాబైట్లు", "tb": "టెరాబైట్లు",
        "kb": "కిలోబైట్లు", "mbps": "మెగాబిట్స్ పర్ సెకను",
        "gbps": "గిగాబిట్స్ పర్ సెకను", "kbps": "కిలోబిట్స్ పర్ సెకను",
        "ms": "మిల్లీసెకన్లు", "s": "సెకన్లు",
        "sec": "సెకన్లు", "secs": "సెకన్లు", "hz": "హెర్ట్జ్",
        "khz": "కిలోహెర్ట్జ్", "mhz": "మెగాహెర్ట్జ్", "ghz": "గిగాహెర్ట్జ్",
        "w": "వాట్లు", "kw": "కిలోవాట్లు", "v": "వోల్ట్లు", "a": "ఆంపియర్లు",
        "mah": "మిల్లీ ఆంపియర్ గంటలు", "°c": "డిగ్రీల సెల్సియస్",
        "°f": "డిగ్రీల ఫారెన్‌హీట్", "%": "శాతం",
    },
    "en": {
        "km": "kilometers", "kms": "kilometers", "m": "meters", "cm": "centimeters",
        "mm": "millimeters", "kg": "kilograms", "kgs": "kilograms", "g": "grams",
        "mg": "milligrams", "l": "liters", "lt": "liters", "ml": "milliliters",
        "gb": "gigabytes", "mb": "megabytes", "tb": "terabytes", "kb": "kilobytes",
        "mbps": "megabits per second", "gbps": "gigabits per second",
        "kbps": "kilobits per second",
        "ms": "milliseconds", "s": "seconds", "sec": "seconds", "secs": "seconds",
        "hz": "hertz", "khz": "kilohertz", "mhz": "megahertz", "ghz": "gigahertz",
        "w": "watts", "kw": "kilowatts", "v": "volts", "a": "amperes",
        "mah": "milliamp hours", "°c": "degrees Celsius", "°f": "degrees Fahrenheit",
        "%": "percent",
    },
}

_UNIT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(\d[\d,]*(?:\.\d+)?)\s*"
    r"(kms|km|kgs|kg|mah|ghz|mhz|khz|hz|gbps|mbps|kbps|gb|mb|tb|kb|ms|secs|sec|ml|lt|cm|mm|mg|kw|"
    r"°c|°f|kg|g|l|m|s|w|v|a|%)(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Keep model names, standards and identifiers intact (COVID-19, GPT-4, 5G, MP3,
# H100).  Without this guard the number/unit normalizer would read "5G" as
# "five grams" or "COVID-19" as "COVID minus nineteen".
_PROTECTED_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])("
    r"[A-Za-z]{2,}-\d+[A-Za-z0-9-]*"
    r"|[A-Za-z]+\d+[A-Za-z0-9]*"
    r"|\d+[GgKkDdPp]\b"
    r")(?![A-Za-z0-9])"
)
_TOKEN_START = "\uE100"
_TOKEN_END = "\uE101"
_TOKEN_MARK_BASE = 0xE300
_TOKEN_MARK_LIMIT = 0xE3FF


def _protect_tokens(text: str) -> tuple[str, list[str]]:
    tokens: list[str] = []

    def replace(match: re.Match[str]) -> str:
        if len(tokens) >= _TOKEN_MARK_LIMIT - _TOKEN_MARK_BASE:
            return match.group(0)
        tokens.append(match.group(0))
        return _TOKEN_START + chr(_TOKEN_MARK_BASE + len(tokens) - 1) + _TOKEN_END

    protected = _PROTECTED_TOKEN.sub(replace, text)
    return protected, tokens


def _restore_tokens(text: str, tokens: list[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        index = ord(match.group(1)) - _TOKEN_MARK_BASE
        if 0 <= index < len(tokens):
            return tokens[index]
        return match.group(0)

    return re.sub(
        re.escape(_TOKEN_START) + r"([\uE300-\uE3FF])" + re.escape(_TOKEN_END),
        replace,
        text,
    )

_DATE_DMY = re.compile(r"(?<!\d)(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})(?!\d)")
_DATE_ISO = re.compile(r"(?<!\d)(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})(?!\d)")
_TIME = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?:\s*([AaPp])\.?\s*[Mm]\.?)?(?!\d)")
_CURRENCY = re.compile(
    r"(?:(₹|రూ\.?|రూపాయలు|Rs\.?|INR)\s*)(\d[\d,]*(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_PERCENT = re.compile(r"(?<![\d.])(\d[\d,]*(?:\.\d+)?)\s*%")
_ORDINAL = re.compile(r"(?<!\d)(\d{1,4})(st|nd|rd|th|వ)(?![A-Za-z])", re.IGNORECASE)
_FRACTION = re.compile(r"(?<![\d/])(\d{1,4})\s*/\s*(\d{1,4})(?![\d/])")
_NUMBER = re.compile(r"(?<![A-Za-z0-9])([+-]?)(\d[\d,]*(?:\.\d+)?)(?![A-Za-z0-9])")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_HTML_TAG = re.compile(r"<[^>]{1,200}>")
_MARKDOWN_EMPHASIS = re.compile(r"(\*\*|__|\*|_|~~|`)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)
_HASHTAG = re.compile(r"(?<!\w)#([\w\u0C00-\u0C7F]+)")

_TELUGU_ORDINALS = {
    1: "మొదటి", 2: "రెండవ", 3: "మూడవ", 4: "నాలుగవ", 5: "ఐదవ",
    6: "ఆరవ", 7: "ఏడవ", 8: "ఎనిమిదవ", 9: "తొమ్మిదవ", 10: "పదవ",
}
_ENGLISH_ORDINALS = {
    0: "zeroth", 1: "first", 2: "second", 3: "third", 4: "fourth",
    5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth",
    10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth",
}
# Endings used when an ordinal suffix applies to the final word of a compound.
_ENGLISH_ORDINAL_ENDINGS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}
_TELUGU_ORDINAL_ENDINGS = {
    "సున్నా": "సున్నావ", "ఒకటి": "ఒకటవ", "రెండు": "రెండవ",
    "మూడు": "మూడవ", "నాలుగు": "నాలుగవ", "ఐదు": "ఐదవ",
    "ఆరు": "ఆరవ", "ఏడు": "ఏడవ", "ఎనిమిది": "ఎనిమిదవ",
    "తొమ్మిది": "తొమ్మిదవ", "పది": "పదవ", "ఇరవై": "ఇరవయ్యవ",
    "ముప్పై": "ముప్పయ్యవ", "నలభై": "నలభయ్యవ", "యాభై": "యాభయ్యవ",
    "అరవై": "అరవయ్యవ", "డెబ్బై": "డెబ్బయ్యవ", "ఎనభై": "ఎనభయ్యవ",
    "తొంభై": "తొంభయ్యవ", "వంద": "వందవ", "వేలు": "వేలవ",
    "వెయ్యి": "వెయ్యవ", "లక్ష": "లక్షవ", "లక్షలు": "లక్షలవ",
    "కోటి": "కోటవ", "కోట్లు": "కోట్లవ",
}


@dataclass(frozen=True)
class FrontendResult:
    text: str
    language: str
    warnings: tuple[str, ...] = ()
    replaced_markup: bool = False


def contains_telugu(text: str) -> bool:
    return bool(_TELUGU_RANGE.search(text or ""))


def _has_latin(text: str) -> bool:
    return bool(_LATIN_RANGE.search(text or ""))


def _number_language(text: str, start: int, end: int, language: str) -> str:
    if language == LANG_TELUGU or language == LANG_ROMANIZED:
        return "te"
    if language == LANG_ENGLISH:
        return "en"
    # Mixed text: use the script of the nearest word. This is a heuristic, not a
    # language detector; it avoids reading every digit in the wrong language.
    before = text[max(0, start - 24):start]
    after = text[end:end + 24]
    before_match = list(re.finditer(r"[A-Za-z\u0C00-\u0C7F]+", before))
    after_match = re.search(r"[A-Za-z\u0C00-\u0C7F]+", after)
    candidates: list[str] = []
    if before_match:
        candidates.append(before_match[-1].group(0))
    if after_match:
        candidates.append(after_match.group(0))
    telugu_votes = sum(1 for candidate in candidates if _TELUGU_RANGE.search(candidate))
    latin_votes = sum(1 for candidate in candidates if _LATIN_RANGE.search(candidate))
    return "te" if telugu_votes >= latin_votes else "en"


def _two_digit_telugu(number: int) -> str:
    if number < 20:
        return _TELUGU_ONES[number]
    tens, ones = divmod(number, 10)
    word = _TELUGU_TENS[tens * 10]
    return f"{word} {_TELUGU_ONES[ones]}" if ones else word


def _below_thousand_telugu(number: int) -> str:
    if number < 100:
        return _two_digit_telugu(number)
    hundreds, remainder = divmod(number, 100)
    if hundreds == 1:
        head = "వంద" if remainder == 0 else "నూట"
    else:
        head = f"{_TELUGU_ONES[hundreds]} వందలు" if remainder == 0 else f"{_TELUGU_ONES[hundreds]} వందల"
    return head if remainder == 0 else f"{head} {_two_digit_telugu(remainder)}"


def number_to_telugu(number: int) -> str:
    """Cardinal Telugu number words for non-negative integers.

    Indian grouping is used (వంద / వేయి / లక్ష / కోటి).  The forms are the common
    spoken forms, not a literary enumeration.
    """
    if number < 0:
        return f"మైనస్ {number_to_telugu(-number)}"
    if number < 1000:
        return _below_thousand_telugu(number)
    if number < 100_000:
        thousands, remainder = divmod(number, 1000)
        if thousands == 1:
            head = "వెయ్యి"
        elif remainder == 0:
            head = f"{_below_thousand_telugu(thousands)} వేలు"
        else:
            head = f"{_below_thousand_telugu(thousands)} వేల"
        return head if remainder == 0 else f"{head} {_below_thousand_telugu(remainder)}"
    if number < 10_000_000:
        lakhs, remainder = divmod(number, 100_000)
        if lakhs == 1:
            head = "లక్ష"
        elif remainder == 0:
            head = f"{_below_thousand_telugu(lakhs)} లక్షలు"
        else:
            head = f"{_below_thousand_telugu(lakhs)} లక్షల"
        return head if remainder == 0 else f"{head} {number_to_telugu(remainder)}"
    crores, remainder = divmod(number, 10_000_000)
    if crores == 1:
        head = "కోటి"
    elif remainder == 0:
        head = f"{number_to_telugu(crores)} కోట్లు"
    else:
        head = f"{number_to_telugu(crores)} కోట్ల"
    return head if remainder == 0 else f"{head} {number_to_telugu(remainder)}"


def number_to_english(number: int) -> str:
    if number < 0:
        return f"minus {number_to_english(-number)}"
    if number < 20:
        return _ENGLISH_ONES[number]
    if number < 100:
        tens, ones = divmod(number, 10)
        return _ENGLISH_TENS[tens] + (f"-{_ENGLISH_ONES[ones]}" if ones else "")
    if number < 1000:
        hundreds, remainder = divmod(number, 100)
        head = f"{_ENGLISH_ONES[hundreds]} hundred"
        return head if remainder == 0 else f"{head} {number_to_english(remainder)}"
    for value, label in ((1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")):
        if number >= value:
            count, remainder = divmod(number, value)
            head = f"{number_to_english(count)} {label}"
            return head if remainder == 0 else f"{head} {number_to_english(remainder)}"
    return str(number)


def _integer_to_words(value: int, language: str) -> str:
    return number_to_telugu(value) if language == "te" else number_to_english(value)


def _digits_to_words(digits: str, language: str) -> str:
    words = []
    for digit in digits:
        index = int(digit)
        words.append(_TELUGU_ONES[index] if language == "te" else _ENGLISH_ONES[index])
    return " ".join(words)


def _decimal_to_words(token: str, language: str) -> str:
    token = token.replace(",", "")
    sign = ""
    if token.startswith(("+", "-")):
        sign, token = token[0], token[1:]
        sign = "మైనస్ " if sign == "-" and language == "te" else ("minus " if sign == "-" else "")
    integer, _, fraction = token.partition(".")
    spoken = _integer_to_words(int(integer or "0"), language)
    if not fraction:
        return f"{sign}{spoken}"
    point = "పాయింట్" if language == "te" else "point"
    return f"{sign}{spoken} {point} {_digits_to_words(fraction, language)}"


def _english_ordinal_ending(word: str, number: int) -> str:
    if number % 100 in {11, 12, 13}:
        return _ENGLISH_ORDINAL_ENDINGS.get(word, f"{word}th")
    if word.endswith("y"):
        return f"{word[:-1]}ieth"
    return _ENGLISH_ORDINAL_ENDINGS.get(
        word,
        {1: "first", 2: "second", 3: "third"}.get(number % 10, f"{word}th"),
    )


def _ordinal_word(number: int, language: str) -> str:
    if language == "te":
        if number in _TELUGU_ORDINALS:
            return _TELUGU_ORDINALS[number]
        cardinal = number_to_telugu(number)
        head, separator, last = cardinal.rpartition(" ")
        ending = _TELUGU_ORDINAL_ENDINGS.get(last)
        if ending is not None:
            return f"{head}{separator}{ending}"
        return f"{cardinal}వ"
    if number in _ENGLISH_ORDINALS:
        return _ENGLISH_ORDINALS[number]
    cardinal = number_to_english(number)
    head, separator, last = cardinal.rpartition(" ")
    if "-" in last:
        prefix, _, final = last.rpartition("-")
        last = f"{prefix}-{_english_ordinal_ending(final, number)}"
    else:
        last = _english_ordinal_ending(last, number)
    return f"{head}{separator}{last}"


def _year_to_words(token: str, language: str) -> str:
    number = int(token)
    if language == "te":
        return number_to_telugu(number)
    if 1100 <= number <= 1999:
        high, low = divmod(number, 100)
        if low == 0:
            return f"{number_to_english(high)} hundred"
        return f"{number_to_english(high)} {number_to_english(low)}"
    return number_to_english(number)


def _date_words(day: int, month: int, year: int, language: str) -> str:
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return ""
    if language == "te":
        return f"{number_to_telugu(day)} {_TELUGU_MONTHS[month - 1]} {_year_to_words(str(year), language)}"
    return f"{_ENGLISH_MONTHS[month - 1]} {_ordinal_word(day, language)} {_year_to_words(str(year), language)}"


def _time_words(hour: int, minute: int, meridiem: str | None, language: str) -> str:
    if hour > 23 or minute > 59:
        return ""
    # Compute the 24-hour value for choosing "morning/evening", then show a
    # 12-hour clock.  A 24-hour input is read with the matching a.m./p.m. period.
    period = meridiem.lower() if meridiem else None
    day_hour = hour
    if period is not None and 0 <= hour <= 12:
        day_hour = (hour % 12) + (12 if period == "p" else 0)
    elif period is None and hour > 12:
        period = "p"
    if hour == 0:
        display_hour = 12
    elif hour > 12:
        display_hour = hour - 12
    else:
        display_hour = hour
    if language == "te":
        if period is None:
            prefix = ""
        elif day_hour < 5:
            prefix = "రాత్రి "
        elif day_hour < 12:
            prefix = "ఉదయం "
        elif day_hour < 16:
            prefix = "మధ్యాహ్నం "
        elif day_hour < 20:
            prefix = "సాయంత్రం "
        else:
            prefix = "రాత్రి "
        if minute == 0:
            phrase = f"{prefix}{number_to_telugu(display_hour)} గంటలు"
        else:
            phrase = (
                f"{prefix}{number_to_telugu(display_hour)} గంటల "
                f"{number_to_telugu(minute)} నిమిషాలు"
            )
        return phrase
    if minute == 0:
        if period:
            return f"{number_to_english(display_hour)} {'a m' if period == 'a' else 'p m'}"
        return f"{number_to_english(display_hour)} o'clock"
    if minute < 10:
        phrase = f"{number_to_english(display_hour)} oh {number_to_english(minute)}"
    else:
        phrase = f"{number_to_english(display_hour)} {number_to_english(minute)}"
    if period:
        phrase += " a m" if period == "a" else " p m"
    return phrase


def _currency_words(amount_text: str, language: str) -> str:
    amount_text = amount_text.replace(",", "")
    try:
        amount = float(amount_text)
    except ValueError:
        return ""
    rupees = int(amount)
    paise = int(round((amount - rupees) * 100))
    if paise == 100:
        rupees += 1
        paise = 0
    if language == "te":
        rupee_word = "రూపాయి" if rupees == 1 else "రూపాయలు"
        parts = [f"{number_to_telugu(rupees)} {rupee_word}"]
        if paise:
            paise_word = "పైసా" if paise == 1 else "పైసలు"
            parts.append(f"{number_to_telugu(paise)} {paise_word}")
        return " ".join(parts)
    rupee_word = "rupee" if rupees == 1 else "rupees"
    parts = [f"{number_to_english(rupees)} {rupee_word}"]
    if paise:
        paise_word = "paisa" if paise == 1 else "paise"
        parts.append(f"{number_to_english(paise)} {paise_word}")
    return " and ".join(parts)


def _unit_words(number_text: str, unit: str, language: str) -> str:
    table = _UNIT_WORDS[language]
    spoken_unit = table.get(unit.lower())
    if spoken_unit is None:
        return ""
    number_text = number_text.replace(",", "")
    if unit.lower() == "%":
        spoken_number = _decimal_to_words(number_text, language)
    else:
        spoken_number = _decimal_to_words(number_text, language)
    return f"{spoken_number} {spoken_unit}"


def _protect_segments(text: str, language: str) -> tuple[str, bool]:
    """Replace markup/URLs/email with speakable placeholders before token work."""
    changed = False

    def link_replacement(match: re.Match[str]) -> str:
        nonlocal changed
        changed = True
        return match.group(1)

    text = _MARKDOWN_LINK.sub(link_replacement, text)
    text = _URL.sub("లింక్" if language != LANG_ENGLISH else "link", text)
    text = _EMAIL.sub("ఇమెయిల్" if language != LANG_ENGLISH else "email", text)
    text = _HTML_TAG.sub(" ", text)
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _MARKDOWN_EMPHASIS.sub("", text)
    text = _HASHTAG.sub(lambda match: match.group(1), text)
    return text, changed


def _normalize_punctuation(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", ", ").replace("\u2014", ", ").replace("\u2212", "-")
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\.{3,}", "…", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.;:!?…])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]])", r"\1", text)
    return text.strip()


def _remove_unsupported_symbols(text: str) -> tuple[str, bool]:
    removed = False
    output: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if category in {"So", "Cs", "Co", "Cn"}:
            removed = True
            output.append(" ")
        else:
            output.append(character)
    return "".join(output), removed


def prepare_speech_text(
    text: str,
    language: str = LANG_TELUGU,
    *,
    expand_numbers: bool = True,
    expand_symbols: bool = True,
    add_terminal_punctuation: bool = True,
) -> FrontendResult:
    """Normalize text for speech while preserving meaning.

    ``normalize_text`` is still available for metadata/transliteration callers
    that must not receive number expansion.
    """
    language = canonical_language(language)
    warnings: list[str] = []
    original = normalize_text(text)
    if not original:
        raise ValueError("Enter text to synthesize.")
    if language == LANG_ROMANIZED:
        warnings.append(
            "Romanized Telugu must be converted to Telugu script before speech normalization."
        )

    working = normalize_text(original.translate(TELUGU_DIGITS))
    working, markup_changed = _protect_segments(working, language)
    if markup_changed:
        warnings.append("Markup, links or email addresses were replaced with spoken placeholders.")

    # Protect model names, standards and identifiers before number/unit expansion
    # so "5G", "MP3", "H100", "GPT-4" and "COVID-19" are not read as quantities.
    working, protected_tokens = _protect_tokens(working)

    if expand_symbols:
        # Order matters: currency and percent first, then units, then bare numbers.
        def currency_sub(match: re.Match[str]) -> str:
            chosen = _number_language(working, match.start(), match.end(), language)
            return _currency_words(match.group(2), chosen)

        def percent_sub(match: re.Match[str]) -> str:
            chosen = _number_language(working, match.start(), match.end(), language)
            return _unit_words(match.group(1), "%", chosen)

        working = _CURRENCY.sub(currency_sub, working)
        working = _PERCENT.sub(percent_sub, working)

        def unit_sub(match: re.Match[str]) -> str:
            chosen = _number_language(working, match.start(), match.end(), language)
            return _unit_words(match.group(1), match.group(2), chosen)

        working = _UNIT_PATTERN.sub(unit_sub, working)

    if expand_numbers:
        def iso_date(match: re.Match[str]) -> str:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            chosen = _number_language(working, match.start(), match.end(), language)
            return _date_words(day, month, year, chosen) or match.group(0)

        def dmy_date(match: re.Match[str]) -> str:
            day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
            chosen = _number_language(working, match.start(), match.end(), language)
            return _date_words(day, month, year, chosen) or match.group(0)

        def time_sub(match: re.Match[str]) -> str:
            chosen = _number_language(working, match.start(), match.end(), language)
            return _time_words(int(match.group(1)), int(match.group(2)), match.group(3), chosen) or match.group(0)

        working = _DATE_ISO.sub(iso_date, working)
        working = _DATE_DMY.sub(dmy_date, working)
        working = _TIME.sub(time_sub, working)

        def ordinal_sub(match: re.Match[str]) -> str:
            chosen = _number_language(working, match.start(), match.end(), language)
            return _ordinal_word(int(match.group(1)), chosen)

        working = _ORDINAL.sub(ordinal_sub, working)

        def fraction_sub(match: re.Match[str]) -> str:
            numerator, denominator = int(match.group(1)), int(match.group(2))
            chosen = _number_language(working, match.start(), match.end(), language)
            if denominator == 0:
                return match.group(0)
            if chosen == "te":
                if numerator == 1 and denominator == 2:
                    return "సగం"
                if numerator == 1 and denominator == 4:
                    return "పావు"
                if numerator == 3 and denominator == 4:
                    return "ముప్పావు"
                return f"{number_to_telugu(numerator)} బై {number_to_telugu(denominator)}"
            if numerator == 1 and denominator == 2:
                return "half"
            if numerator == 1 and denominator == 4:
                return "a quarter"
            if numerator == 3 and denominator == 4:
                return "three quarters"
            names = {2: ("half", "halves"), 3: ("third", "thirds"), 4: ("quarter", "quarters")}
            singular, plural = names.get(
                denominator,
                (_ordinal_word(denominator, "en"), _ordinal_word(denominator, "en") + "s"),
            )
            if numerator == 1:
                return f"one {singular}"
            return f"{number_to_english(numerator)} {plural}"

        working = _FRACTION.sub(fraction_sub, working)

        def number_sub(match: re.Match[str]) -> str:
            sign, token = match.group(1), match.group(2)
            chosen = _number_language(working, match.start(), match.end(), language)
            digits = token.replace(",", "").split(".")[0]
            # Long identifier-like runs (phone numbers, order IDs) are read
            # digit by digit. Years and comma-grouped amounts are not.
            if len(digits) >= 7 and "." not in token and "," not in token:
                spoken = _digits_to_words(digits, chosen)
                if sign == "-":
                    spoken = ("మైనస్ " if chosen == "te" else "minus ") + spoken
                return spoken
            return _decimal_to_words(f"{sign}{token}", chosen)

        working = _NUMBER.sub(number_sub, working)

    # Restore protected identifiers before symbol removal.  The placeholder
    # characters are private-use code points and would otherwise be stripped.
    if protected_tokens:
        working = _restore_tokens(working, protected_tokens)

    if expand_symbols:
        symbol_replacements = {
            "&": (" మరియు ", " and "),
            "+": (" ప్లస్ ", " plus "),
            "=": (" సమానం ", " equals "),
            "×": (" గుణించి ", " times "),
            "÷": (" భాగించి ", " divided by "),
            "@": (" ఎట్ ", " at "),
            "~": (" సుమారు ", " about "),
        }
        for symbol, (telugu_word, english_word) in symbol_replacements.items():
            working = working.replace(symbol, telugu_word if language != LANG_ENGLISH else english_word)
        working = working.replace("#", " హాష్ " if language != LANG_ENGLISH else " hash ")

    working, removed = _remove_unsupported_symbols(working)
    if removed:
        warnings.append("Emoji or unsupported symbols were removed.")
    working = _normalize_punctuation(working)
    if language == LANG_TELUGU and not contains_telugu(working):
        warnings.append("Telugu mode selected, but the text contains no Telugu-script characters.")
    if language == LANG_MIXED:
        warnings.append("Telugu-English pronunciation and speaker similarity are experimental.")
    # Re-normalize paragraphs without destroying sentence punctuation.
    paragraphs = [re.sub(r"[ \t]+", " ", part).strip() for part in working.split("\n")]
    paragraphs = [part for part in paragraphs if part]
    if add_terminal_punctuation:
        paragraphs = [
            part if re.search(r"[.!?…।]$", part) else part + "."
            for part in paragraphs
        ]
    working = "\n\n".join(paragraphs)
    if not working:
        raise ValueError("Nothing speakable remained after text normalization.")
    return FrontendResult(working, language, tuple(warnings), markup_changed)


def split_segments(text: str) -> list[str]:
    """Split normalized text into paragraph/sentence-like units for previews."""
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text or ""):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        pieces.extend(part.strip() for part in re.split(r"(?<=[.!?…।])\s+", paragraph) if part.strip())
    return pieces
