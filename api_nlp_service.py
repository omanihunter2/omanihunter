"""
خدمة FastAPI لمعالجة العربية لمنصة التحليل اللساني.

تشغيل محلي:
  pip install fastapi uvicorn pydantic camel-tools
  uvicorn api_nlp_service:app --host 0.0.0.0 --port 8000

على Render:
  Build Command: pip install -r requirements.txt
  Start Command: uvicorn api_nlp_service:app --host 0.0.0.0 --port $PORT

النهايات:
  GET  /health
  POST /analyze
  POST /classify
  POST /summarize
  POST /suggest_tags
  GET  /docs
  GET  /openapi.json
"""
from __future__ import annotations

import math
import os
import re
import subprocess
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Arabic Linguistic Platform NLP API",
    version="2.5.1",
    description=(
        "Arabic NLP service with CAMeL Tools when available, optional Farasa hook, "
        "improved clitic-aware fallback, classification, summarization, and tag suggestion."
    ),
)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = "ar"
    max_items: int = 20


class AnalyzeRequest(TextRequest):
    engine: str = "auto"  # auto | camel | farasa | fallback
    split_clitics: bool = True
    include_debug: bool = False


DIAC = re.compile(r"[\u064B-\u065F\u0670\u0640\u06D6-\u06ED]")
AR_WORD = re.compile(r"[\u0621-\u063A\u0641-\u064A]+", re.UNICODE)
SENT_SPLIT = re.compile(r"[.!؟?؛،\n]+")

# Normalized sets are used for matching after hamza/alef/ya normalization.
RAW_STOPWORDS = """
من إلى الى عن على علي في ثم أو او أم ام بل لكن لا لم لن قد هل حتى حتي إذا اذا إن ان أن ما كي لعل ليت رب هذا هذه ذلك تلك هو هي هم هن نحن انا أنا كان كانت يكون كل غير بعد قبل بين عند مع كما وقد فقد و ف ب ك ل
""".split()

RAW_PREPOSITIONS = {
    "من", "إلى", "الى", "الي", "عن", "على", "علي", "في", "ب", "ك", "ل", "رب", "حتى", "حتي", "منذ", "خلا", "عدا", "حاشا"
}
RAW_CONJUNCTIONS = {"و", "ف", "ثم", "أو", "او", "أم", "ام", "بل", "لكن"}
RAW_PARTICLES = {
    "أن", "ان", "إن", "لن", "لم", "لا", "ما", "قد", "هل", "ليت", "لعل", "كي", "حتى", "حتي", "إذا", "اذا", "إذ", "اذ"
}
RAW_PRONOUNS = {
    "أنا", "انا", "أنت", "انت", "أنتما", "انتما", "أنتم", "انتم", "أنتن", "انتن", "هو", "هي", "نحن", "هم", "هن", "هما",
    "ت", "نا", "ي", "ك", "ه", "ها", "كم", "كن", "كما", "هما"
}
RAW_DEMONSTRATIVES = {"هذا", "هذه", "ذلك", "تلك", "هؤلاء", "هنا", "هناك", "أولئك", "اولئك"}
RAW_RELATIVES = {"الذي", "التي", "الذين", "اللاتي", "اللائي", "اللواتي", "اللذان", "اللتان", "من", "ما"}
RAW_INTERROGATIVES = {"من", "ما", "ماذا", "متى", "متي", "اين", "أين", "كيف", "كم", "أي", "اي", "لماذا", "أيها", "ايها"}

KNOWN_VERBS = {
    "كان", "كنت", "كانت", "يكون", "سأل", "سألت", "سال", "سالت", "وجد", "وجدت", "أكون", "اكون", "مضى", "مضي",
    "أقلب", "اقلب", "أنتظر", "انتظر", "أسمع", "اسمع", "يأتي", "ياتي", "يأتينا", "ياتينا", "يخبر", "ليخبرنا",
    "خرج", "يهذي", "يجتاحني", "أفعله", "افعله", "حل", "عاش", "بنى", "بناه", "ترك", "تركت", "تركته", "يستمر",
    "أستيقظ", "استيقظ", "أدرك", "ادرك", "استعدوا", "فتحت", "أصبحت", "اصبحت", "أحمل", "احمل", "أضم", "اضم",
    "يهمك", "تحتاج", "نعرض", "يلي", "ساءت", "ليست", "فليست",
    "سيتركان", "يتركان", "يترك", "تهملوا", "تهمل", "سيحقق", "يحقق", "يتكاسل", "يتقدم", "نساعده", "نساعد"
}

VALID_PREFIX_CLITICS = {"و", "ف", "ب", "ك", "ل"}
VALID_SUFFIX_PRONOUNS = ["كما", "هما", "كم", "كن", "نا", "ها", "هم", "هن", "ه", "ك", "ي"]

CATEGORY_KEYWORDS = {
    "شعر": ["قصيدة", "بيت", "قافية", "شاعر", "أبيات", "بحر", "غزل"],
    "خطبة": ["أيها", "الحمد", "السلام", "الناس", "أوصيكم", "أما بعد"],
    "أكاديمي": ["بحث", "دراسة", "منهج", "نتائج", "مراجع", "فرضية", "تحليل"],
    "إعلامي": ["قال", "أعلن", "أخبار", "تقرير", "مصادر", "صحيفة", "وكالة"],
    "تعليمي": ["درس", "شرح", "تمرين", "الطلاب", "تعلم", "المعلم", "الصف"],
    "سردي": ["كان", "كنت", "قال", "ذهب", "عاد", "رأى", "وجد", "فجأة"],
    "تراثي": ["قال", "روي", "حدثنا", "باب", "فصل", "رحمه", "كتاب"],
}


# -----------------------------------------------------------------------------
# Normalization and basic tokenization
# -----------------------------------------------------------------------------

def strip_diacritics(text: str) -> str:
    return DIAC.sub("", text or "")


def normalize(text: str) -> str:
    text = strip_diacritics(text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    return text


def nset(items: set[str] | list[str]) -> set[str]:
    return {normalize(x) for x in items}


# Words whose final/initial letters are lexical, not detachable clitics.
# This list addresses recurrent false splits observed in the paired platform reports.
LEXICAL_NO_SPLIT = nset({
    "الذي", "التي", "الذين", "اللاتي", "اللائي", "اللذان", "اللتان",
    "الماضي", "ماضي", "فجأة", "يهذي", "يأتي", "يلي", "يجتاحني",
    "هذيانا", "جنونا", "حلما", "صفحة", "صفحات", "عملية", "الحياة",
    "حاسوبية", "عربية", "ثرية", "لتوه"
})


STOPWORDS = nset(RAW_STOPWORDS)
PREPOSITIONS = nset(RAW_PREPOSITIONS)
CONJUNCTIONS = nset(RAW_CONJUNCTIONS)
PARTICLES = nset(RAW_PARTICLES)
PRONOUNS = nset(RAW_PRONOUNS)
DEMONSTRATIVES = nset(RAW_DEMONSTRATIVES)
RELATIVES = nset(RAW_RELATIVES)
INTERROGATIVES = nset(RAW_INTERROGATIVES)
KNOWN_VERBS_N = nset(KNOWN_VERBS)


def clean_word(word: str) -> str:
    # Keep Arabic letters only; remove punctuation such as ، ؛ ؟ etc.
    return "".join(AR_WORD.findall(strip_diacritics(word or "")))


def words(text: str) -> List[str]:
    # Remove diacritics before tokenization. Otherwise a final tanween such as
    # نجاحًا is incorrectly tokenized as "نجاح" + "ا".
    plain = strip_diacritics(text or "")
    return [w for w in (clean_word(x) for x in AR_WORD.findall(plain)) if w]


def sentences(text: str) -> List[str]:
    return [s.strip() for s in SENT_SPLIT.split(text or "") if s.strip()]


def is_single_letter_particle(part: str) -> bool:
    return normalize(part) in {"و", "ف", "ب", "ك", "ل"}


# -----------------------------------------------------------------------------
# Clitic splitting
# -----------------------------------------------------------------------------

def _looks_like_real_prefix(surface: str, prefix: str, rest: str) -> bool:
    n = normalize(surface)
    nr = normalize(rest)
    if n in LEXICAL_NO_SPLIT or len(rest) < 2:
        return False
    if prefix in {"و", "ف"}:
        return (nr in PRONOUNS or nr in PARTICLES or nr in DEMONSTRATIVES or
                nr in RELATIVES or nr in INTERROGATIVES or nr.startswith("ال") or
                nr in KNOWN_VERBS_N or len(rest) >= 3)
    if prefix in {"ب", "ك", "ل"}:
        return nr.startswith("ال") or nr in PRONOUNS or len(rest) >= 3
    return False


def split_attached_word(surface: str) -> List[str]:
    """Conservative fallback segmentation used only when CAMeL is unavailable.

    CAMeL-enabled requests are segmented from the selected CAMeL analysis, not
    merely from spelling. This prevents errors such as الذي -> الذ + ي and
    فجأة -> ف + جأة.
    """
    w = clean_word(surface)
    if not w:
        return []
    n = normalize(w)
    if n in PREPOSITIONS or n in CONJUNCTIONS or n in PARTICLES or n in PRONOUNS or n in LEXICAL_NO_SPLIT:
        return [w]

    parts: List[str] = []
    first = w[0]
    rest = w[1:]
    if normalize(first) in VALID_PREFIX_CLITICS and _looks_like_real_prefix(w, normalize(first), rest):
        parts.append(first)
        w = rest
        n = normalize(w)

    # Split suffixes only when the remaining base is plausible. Never strip final
    # yaa from lexical forms merely because it resembles a possessive pronoun.
    suffix = ""
    for s in VALID_SUFFIX_PRONOUNS:
        if not normalize(w).endswith(normalize(s)) or len(w) <= len(s) + 2:
            continue
        base = w[:-len(s)]
        nb = normalize(base)
        if s == "ي" and normalize(surface) not in {"نفسي", "قلبي", "كتابي", "بيتي", "عالمي"}:
            continue
        if nb in LEXICAL_NO_SPLIT or nb.endswith(("ا", "ى")):
            continue
        if nb in KNOWN_VERBS_N or nb.startswith("ال") or len(base) >= 3:
            suffix = w[-len(s):]
            w = base
            break

    if w:
        parts.append(w)
    if suffix:
        parts.append(suffix)
    return parts or [clean_word(surface)]


def expanded_tokens(text: str, split_clitics: bool = True) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for word_index, surface in enumerate(words(text), start=1):
        parts = split_attached_word(surface) if split_clitics else [surface]
        for sub, part in enumerate(parts, start=1):
            part = clean_word(part)
            if part:
                out.append((surface, part, f"{word_index}.{sub}"))
    return out


def _camel_has_clitic(value: Any) -> bool:
    v = str(value or "").strip().lower()
    return v not in {"", "0", "na", "none", "null", "_"}


def _camel_prefix_letters(ana: Dict[str, Any], surface: str) -> List[str]:
    """Return only detachable conjunction/preposition letters.

    The definite article and verbal inflectional prefixes remain attached.
    """
    values = " ".join(str(ana.get(k, "")) for k in ("prc3", "prc2", "prc1", "prc0")).lower()
    result: List[str] = []
    cursor = surface
    mapping = [("wa", "و"), ("fa", "ف"), ("bi", "ب"), ("ka", "ك"), ("li", "ل") ]
    for code, letter in mapping:
        if cursor.startswith(letter) and (code + "_" in values or code in values.split()):
            result.append(letter)
            cursor = cursor[1:]
    return result


def camel_units_for_word(surface: str, ana: Dict[str, Any], word_index: int, split_clitics: bool) -> List[Tuple[str, str, str]]:
    """Segment a surface word only when the selected CAMeL analysis confirms it."""
    clean = clean_word(surface)
    if not clean or not split_clitics or normalize(clean) in LEXICAL_NO_SPLIT:
        return [(clean, clean, f"{word_index}.1")] if clean else []

    prefixes = _camel_prefix_letters(ana, clean)
    remainder = clean[len(prefixes):]
    suffix = ""
    enc0 = ana.get("enc0")
    if _camel_has_clitic(enc0):
        for candidate in VALID_SUFFIX_PRONOUNS:
            if normalize(remainder).endswith(normalize(candidate)) and len(remainder) > len(candidate) + 1:
                # final yaa is detached only when CAMeL explicitly identifies enc0.
                suffix = remainder[-len(candidate):]
                remainder = remainder[:-len(candidate)]
                break

    units: List[Tuple[str, str, str]] = []
    sub = 1
    for prefix in prefixes:
        units.append((clean, prefix, f"{word_index}.{sub}")); sub += 1
    if remainder:
        units.append((clean, remainder, f"{word_index}.{sub}")); sub += 1
    if suffix:
        units.append((clean, suffix, f"{word_index}.{sub}"))
    return units or [(clean, clean, f"{word_index}.1")]


# -----------------------------------------------------------------------------
# Fallback morphology
# -----------------------------------------------------------------------------

def fallback_pos(w: str) -> str:
    n = normalize(w)
    if n in PREPOSITIONS or n in CONJUNCTIONS or n in PARTICLES or is_single_letter_particle(w):
        return "حرف"
    if n in PRONOUNS or n in DEMONSTRATIVES or n in RELATIVES or n in INTERROGATIVES:
        return "اسم"
    if n.startswith("ال") or n.endswith(("ة", "ات", "ون", "ين", "ية")):
        return "اسم"
    if n in KNOWN_VERBS_N:
        return "فعل"
    if re.match(r"^س[اينت].{2,}$", n):
        return "فعل"
    if re.match(r"^[اينت].{2,}(?:وا|ان|ون|ين|ن)$", n) and not n.startswith("ال"):
        return "فعل"
    if n.startswith(("ي", "ت", "ن")) and len(n) >= 4:
        return "فعل"
    if n.startswith("ا") and len(n) >= 4 and n in KNOWN_VERBS_N:
        return "فعل"
    return "اسم"


def fallback_features(w: str, pos: str) -> str:
    n = normalize(w)
    features: List[str] = []
    if pos == "حرف":
        if n in PREPOSITIONS or n in {"ب", "ك", "ل"}:
            features.append("نوع الأداة: حرف جر")
        elif n in CONJUNCTIONS or n in {"و", "ف"}:
            features.append("نوع الحرف: حرف عطف")
        else:
            features.append("نوع الأداة: حرف")
    elif pos == "فعل":
        if n.startswith(("ي", "ت", "ن", "ا")):
            features.append("زمن/تمام الفعل: غير تام / مضارع")
            features.append("صيغة الفعل: مرفوع")
        else:
            features.append("زمن/تمام الفعل: تام / ماض")
        features.append("صيغة الفعل: فعل مصرف")
        features.append("البناء: مبني للمعلوم")
    else:
        if n.startswith("ال"):
            features.append("التعريف: معرفة")
        elif n not in PRONOUNS:
            features.append("التعريف: نكرة")
        if n in PRONOUNS:
            features.append("نوع الضمير/الاسم: ضمير شخصي")
        if n in DEMONSTRATIVES:
            features.append("نوع الضمير/الاسم: اسم إشارة")
        if n in RELATIVES:
            features.append("نوع الضمير/الاسم: اسم موصول")
        if n in INTERROGATIVES:
            features.append("نوع الضمير/الاسم: اسم استفهام")
        if n.endswith(("ة", "ية")):
            features.append("الجنس: مؤنث محتمل")
        if n.endswith(("ات", "ون", "ين")):
            features.append("العدد: جمع")
        elif n not in PRONOUNS:
            features.append("العدد: مفرد")
    return "، ".join(features)


def fallback_stem_prefix_suffix(w: str, pos: str) -> Tuple[str, str, str]:
    n = normalize(w)
    prefix = ""
    suffix = ""
    stem = w

    if pos != "حرف" and n.startswith("ال") and len(w) > 3:
        prefix = "ال"
        stem = w[2:]

    # Do not interpret lexical final letters as pronouns (الذي، الماضي، يهذي...).
    if n not in LEXICAL_NO_SPLIT:
        for s in ["كما", "هما", "كم", "كن", "نا", "ها", "هم", "هن", "ه", "ك", "ي"]:
            if len(stem) > len(s) + 2 and normalize(stem).endswith(normalize(s)):
                suffix = stem[-len(s):]
                stem = stem[:-len(s)]
                break
    return prefix, stem, suffix


def guess_root(stem: str) -> str:
    n = normalize(stem)
    n = re.sub(r"^ال", "", n)
    n = re.sub(r"^(و|ف|ب|ك|ل)", "", n)
    n = re.sub(r"(كما|هما|كم|كن|نا|ها|هم|هن|ه|ك|ي)$", "", n)
    letters = [c for c in n if AR_WORD.fullmatch(c)]
    return "".join(letters[:3]) if letters else n[:3]


def sanitize_root_value(root: Any, word: str = "", lemma: str = "") -> str:
    raw = str(root or "").strip()
    if re.search(r"[A-Za-z]", raw) or raw.upper() in {"NTWS", "NA", "N/A", "NONE", "NULL", "UNK", "UNKNOWN"}:
        raw = ""
    raw = "".join(AR_WORD.findall(strip_diacritics(raw)))
    key = normalize(word or lemma)
    exact = {
        "الوقت": "وقت", "وقت": "وقت",
        "الذي": "", "التي": "", "الذين": "", "اللاتي": "", "اللائي": "",
        "صفحات": "صفح", "صفحة": "صفح",
        "نماذج": "نمذج", "نموذج": "نمذج",
        "نصوص": "نصص", "نص": "نصص",
        "الفنون": "فنن", "فنون": "فنن", "فن": "فنن",
    }
    if key in exact:
        return exact[key]
    return raw or guess_root(lemma or word)


def derive_arabic_pattern(word: str, lemma: str = "", pos: str = "") -> str:
    w = normalize(word)
    if w in PARTICLES or w in PREPOSITIONS or w in CONJUNCTIONS:
        return "أداة"
    if w in RELATIVES:
        return "اسم موصول مبني (لا وزن صرفي)"
    if w in PRONOUNS:
        return "ضمير مبني (لا وزن صرفي)"
    if w in DEMONSTRATIVES:
        return "اسم إشارة مبني (لا وزن صرفي)"
    exact = {
        "الوقت": "فَعْل", "وقت": "فَعْل",
        "صفحات": "فَعَلات (جمع مؤنث سالم)",
        "نماذج": "فَعَالِل (جمع تكسير)",
        "نصوص": "فُعُول (جمع تكسير)",
        "الفنون": "فُعُول (جمع تكسير)", "فنون": "فُعُول (جمع تكسير)",
    }
    if w in exact:
        return exact[w]
    if w.endswith("ات") and len(w) >= 4:
        return "جمع مؤنث سالم"
    if w.endswith(("ون", "ين")) and len(w) >= 4:
        return "جمع مذكر سالم محتمل"
    return "غير محدد"


def sanitize_pattern_value(pattern: Any, word: str = "", lemma: str = "", pos: str = "") -> str:
    raw = str(pattern or "").strip()
    invalid = (not raw or raw == "غير محدد" or bool(re.search(r"[A-Za-z#0-9]", raw))
               or raw.upper() in {"NTWS", "NA", "N/A", "NONE", "NULL", "UNK", "UNKNOWN"})
    return derive_arabic_pattern(word, lemma, pos) if invalid else raw


def fallback_token(surface: str, part: str, position: str) -> Dict[str, Any]:
    clean = clean_word(part)
    pos = fallback_pos(clean)
    fallback_prefix, fallback_stem, fallback_suffix = fallback_stem_prefix_suffix(clean, pos)
    lemma = normalize(fallback_stem) if fallback_stem else normalize(clean)
    prefix, suffix = grammatical_affixes(clean, {}, pos, fallback_prefix, fallback_suffix)
    core_part = lexical_core_part(clean, pos, {}, lemma)
    root = robust_root_value("", clean, lemma, core_part, pos)
    return {
        "position": position,
        "surface": surface,
        "word": core_part,
        "normalized": normalize(core_part),
        "lemma": lemma,
        "root": root,
        "pattern": derive_arabic_pattern(clean, lemma, pos),
        "pos": pos,
        "prefix": prefix,
        "stem": core_part,
        "suffix": suffix,
        "features": fallback_features(core_part, pos),
        "confidence": "fallback-rule-phase20",
    }


# -----------------------------------------------------------------------------
# CAMeL Tools integration
# -----------------------------------------------------------------------------

_camel_mle = None
_camel_load_error: Optional[str] = None


def camel_tools_installed() -> bool:
    try:
        import camel_tools  # noqa: F401
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def camel_normalized_pos_sets() -> Dict[str, set[str]]:
    return {
        "prep": PREPOSITIONS,
        "conj": CONJUNCTIONS,
        "part": PARTICLES,
        "pron": PRONOUNS,
    }


def get_camel_mle():
    global _camel_mle, _camel_load_error
    if _camel_mle is not None:
        return _camel_mle
    try:
        from camel_tools.disambig.mle import MLEDisambiguator
        _camel_mle = MLEDisambiguator.pretrained()
        _camel_load_error = None
        return _camel_mle
    except Exception as exc:  # keep service alive and expose reason in debug
        _camel_load_error = str(exc)
        return None


def map_camel_pos(pos_en: str, part: str) -> str:
    p = (pos_en or "").lower()
    n = normalize(part)
    if n in PREPOSITIONS or n in CONJUNCTIONS or n in PARTICLES or is_single_letter_particle(part):
        return "حرف"
    if p.startswith("verb") or p in {"verb", "iv", "pv", "cv"}:
        return "فعل"
    if p in {"prep", "conj", "part", "punc", "sub_conj", "interrog_part"}:
        return "حرف"
    return "اسم"


def arabic_camel_features(ana: Dict[str, Any], part: str, pos: str) -> str:
    features: List[str] = []
    n = normalize(part)
    if pos == "حرف":
        if n in PREPOSITIONS or n in {"ب", "ك", "ل"}:
            return "نوع الأداة: حرف جر"
        if n in CONJUNCTIONS or n in {"و", "ف"}:
            return "نوع الحرف: حرف عطف"
        return "نوع الأداة: حرف"

    if pos == "فعل":
        asp = ana.get("asp") or ""
        if asp == "p":
            features.append("زمن/تمام الفعل: تام / ماض")
        elif asp == "i":
            features.append("زمن/تمام الفعل: غير تام / مضارع")
        elif asp:
            features.append(f"الزمن/التمام: {asp}")
        vox = ana.get("vox")
        if vox == "a":
            features.append("البناء: مبني للمعلوم")
        elif vox == "p":
            features.append("البناء: مبني للمجهول")
        features.append("صيغة الفعل: فعل مصرف")
    else:
        stt = ana.get("stt") or ""
        cas = ana.get("cas") or ""
        gen = ana.get("gen") or ""
        num = ana.get("num") or ""
        if n.startswith("ال") or stt == "d":
            features.append("التعريف: معرفة")
        elif stt == "i":
            features.append("التعريف: نكرة")
        if cas == "n":
            features.append("الحالة الإعرابية: مرفوع")
        elif cas == "a":
            features.append("الحالة الإعرابية: منصوب")
        elif cas == "g":
            features.append("الحالة الإعرابية: مجرور")
        if gen == "m":
            features.append("الجنس: مذكر")
        elif gen == "f":
            features.append("الجنس: مؤنث")
        if num == "s":
            features.append("العدد: مفرد")
        elif num == "d":
            features.append("العدد: مثنى")
        elif num == "p":
            features.append("العدد: جمع")
        if n in PRONOUNS:
            features.append("نوع الضمير/الاسم: ضمير شخصي")
        if n in DEMONSTRATIVES:
            features.append("نوع الضمير/الاسم: اسم إشارة")
        if n in RELATIVES:
            features.append("نوع الضمير/الاسم: اسم موصول")
        if n in INTERROGATIVES:
            features.append("نوع الضمير/الاسم: اسم استفهام")

    return "، ".join(features)


_BW_FALLBACK = str.maketrans({
    "'": "ء", "|": "آ", ">": "أ", "&": "ؤ", "<": "إ", "}": "ئ",
    "A": "ا", "b": "ب", "p": "ة", "t": "ت", "v": "ث", "j": "ج",
    "H": "ح", "x": "خ", "d": "د", "*": "ذ", "r": "ر", "z": "ز",
    "s": "س", "$": "ش", "S": "ص", "D": "ض", "T": "ط", "Z": "ظ",
    "E": "ع", "g": "غ", "f": "ف", "q": "ق", "k": "ك", "l": "ل",
    "m": "م", "n": "ن", "h": "ه", "w": "و", "Y": "ى", "y": "ي",
    "F": "ً", "N": "ٌ", "K": "ٍ", "a": "َ", "u": "ُ", "i": "ِ",
    "~": "ّ", "o": "ْ", "`": "ٰ"
})


def camel_arabic_value(value: Any) -> str:
    """Convert CAMeL/Buckwalter text to Arabic when necessary."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if re.search(r"[A-Za-z$*><&|{}']", raw):
        try:
            from camel_tools.utils.charmap import CharMapper
            raw = CharMapper.builtin_mapper("bw2ar")(raw)
        except Exception:
            raw = raw.translate(_BW_FALLBACK)
    return raw


def camel_root_value(value: Any, word: str = "", lemma: str = "") -> str:
    raw = camel_arabic_value(value)
    # Roots may be dot-separated (و.ق.ت). Preserve Arabic radicals only.
    raw = "".join(AR_WORD.findall(strip_diacritics(raw)))
    return sanitize_root_value(raw, word, lemma)


def force_pos_from_surface(surface: str, ana: Dict[str, Any], mapped_pos: str) -> str:
    """Resolve high-confidence surface forms that an undiacritized MLE choice may misclassify.

    This does not replace CAMeL's analysis. It only corrects forms carrying explicit
    Arabic verbal morphology, especially future سـ and plural/dual verb endings.
    """
    n = normalize(clean_word(surface))
    p = str(ana.get("pos") or "").lower()
    asp = str(ana.get("asp") or "").lower()
    if p.startswith("verb") or p in {"verb", "iv", "pv", "cv"} or asp in {"i", "p", "c"}:
        return "فعل"
    # Definite nouns must not be mistaken for first-person verbs merely because
    # normalized ال begins with ا.
    if n.startswith("ال"):
        return mapped_pos
    # Future marker سـ followed by an imperfect person marker.
    if re.match(r"^س[اينت].{2,}$", n):
        return "فعل"
    # Imperfect/imperative forms with unmistakable verbal endings.
    if re.match(r"^[اينت].{2,}(?:وا|ان|ون|ين|ن)$", n):
        return "فعل"
    if n in KNOWN_VERBS_N:
        return "فعل"
    return mapped_pos


def _strip_confirmed_pronoun(surface: str, ana: Dict[str, Any]) -> Tuple[str, str]:
    """Remove only a CAMeL-confirmed object/possessive enclitic."""
    w = clean_word(surface)
    if not _camel_has_clitic(ana.get("enc0")):
        return w, ""
    for ending in VALID_SUFFIX_PRONOUNS:
        if normalize(w).endswith(normalize(ending)) and len(w) > len(ending) + 1:
            return w[:-len(ending)], w[-len(ending):]
    return w, ""


def lexical_core_part(surface: str, pos: str, ana: Dict[str, Any], lemma: str = "") -> str:
    """Return the linguistic part displayed by the platform.

    Policy:
      * detachable و/ف/ب/ك/ل are already separate rows;
      * future سـ is removed, while the imperfect marker ي/ت/أ/ن remains;
      * conjugational endings and attached pronouns are removed;
      * for nouns, number endings are removed while the definite article remains;
      * broken plurals use the CAMeL lemma, retaining ال when present.
    """
    w, attached_pronoun = _strip_confirmed_pronoun(surface, ana)
    n = normalize(w)
    lemma_clean = clean_word(strip_diacritics(lemma))
    lemma_n = normalize(lemma_clean)

    # When a possessive pronoun is attached to a noun ending in taa marbuta,
    # Arabic orthography changes ة to ت: مدرسة + نا -> مدرستنا.
    # Restore the lexical form after removing the confirmed pronoun.
    if (pos == "اسم" and attached_pronoun and w.endswith("ت")
            and lemma_clean.endswith("ة")):
        w = w[:-1] + "ة"
        n = normalize(w)

    if pos == "فعل":
        # سـ is a future particle, but the imperfect person marker is part of the
        # requested displayed core: سيحقق -> يحقق; سيتركان -> يترك.
        if re.match(r"^س[اينت]", n) and len(w) > 3:
            w = w[1:]
            n = normalize(w)
        # Longest endings first. Keep the imperfect prefix itself.
        for ending in ("تما", "تن", "تم", "وا", "ان", "ون", "ين", "نا", "ن"):
            if n.endswith(ending) and len(w) > len(ending) + 2:
                w = w[:-len(ending)]
                n = normalize(w)
                break
        # Past person endings are removed only when CAMeL marks perfect aspect.
        if str(ana.get("asp") or "") == "p":
            for ending in ("ت", "نا"):
                if n.endswith(ending) and len(w) > len(ending) + 2:
                    w = w[:-len(ending)]
                    break
        return w or clean_word(surface)

    if pos == "اسم":
        has_article = n.startswith("ال")
        # Prefer the lexeme for broken plurals and number-inflected nouns.
        if (lemma_clean and lemma_n != n and not re.search(r"(?:ات|ان|ون|ين)$", lemma_n)
                and lemma_n not in PRONOUNS and lemma_n not in RELATIVES
                and lemma_n not in DEMONSTRATIVES):
            base = lemma_clean
            if normalize(base).startswith("ال"):
                base = base[2:]
            if has_article:
                base = "ال" + base
            # Use the lemma when CAMeL explicitly reports dual/plural or the
            # surface has a regular number ending.
            if str(ana.get("num") or "") in {"d", "p"} or re.search(r"(?:ات|ان|ون|ين)$", n):
                return base
        for ending in ("ات", "ان", "ون", "ين"):
            if n.endswith(ending) and len(w) > len(ending) + 2:
                return w[:-len(ending)]
        return w

    return w


def robust_root_value(root: Any, surface: str, lemma: str, core: str, pos: str) -> str:
    """Prefer CAMeL's radical root, then use guarded lexeme-based recovery."""
    key = normalize(surface)
    exact = {
        "الطالبان": "طلب", "طالبان": "طلب", "الطالب": "طلب", "طالب": "طلب",
        "سيتركان": "ترك", "يتركان": "ترك", "يترك": "ترك",
        "تهملوا": "همل", "تهمل": "همل", "اهمل": "همل",
        "سيحقق": "حقق", "يحقق": "حقق", "حقق": "حقق",
        "يتكاسل": "كسل", "يتقدم": "قدم", "نساعده": "سعد", "نساعد": "سعد",
    }
    if key in exact:
        return exact[key]
    candidate = camel_root_value(root, surface, lemma)
    if candidate and 3 <= len(candidate) <= 4:
        return candidate
    # The lemma is safer than the inflected surface. This remains a fallback;
    # weak/hamzated/doubled roots require dictionary data or a correction row.
    return sanitize_root_value("", core or surface, lemma)


def grammatical_affixes(surface: str, ana: Dict[str, Any], pos: str, fallback_prefix: str = "", fallback_suffix: str = "") -> Tuple[str, str]:
    """Return inflectional prefixes/suffixes without creating standalone rows.

    Detachable conjunctions/prepositions are represented as rows. Here we keep
    verbal markers (س، أ/ن/ي/ت) and endings (وا، ان...) in prefix/suffix fields.
    """
    w = clean_word(surface)
    n = normalize(w)
    prefixes: List[str] = []
    suffixes: List[str] = []

    # Definite article remains attached to nouns.
    if pos == "اسم" and n.startswith("ال") and len(n) > 3:
        prefixes.append("ال")

    if pos == "فعل":
        cursor = n
        if cursor.startswith("س") and len(cursor) >= 4 and cursor[1:2] in {"ا", "ن", "ي", "ت"}:
            prefixes.append("س")
            cursor = cursor[1:]
        asp = str(ana.get("asp") or "")
        if asp == "i" and cursor[:1] in {"ا", "ن", "ي", "ت"}:
            prefixes.append(cursor[:1])

        # Longest first; these are inflectional endings, not independent tokens.
        for ending in ("تما", "تن", "تم", "وا", "ان", "ون", "ين", "نا", "ن", "ت"):
            if n.endswith(ending) and len(n) > len(ending) + 2:
                suffixes.append(ending)
                break
    elif pos == "اسم":
        for ending in ("ات", "ان", "ون", "ين"):
            if n.endswith(ending) and len(n) > len(ending) + 2:
                suffixes.append(ending)
                break

    # Object/possessive pronouns confirmed by CAMeL.
    if _camel_has_clitic(ana.get("enc0")):
        for ending in VALID_SUFFIX_PRONOUNS:
            if n.endswith(normalize(ending)) and len(n) > len(ending) + 1:
                if ending not in suffixes:
                    suffixes.append(ending)
                break

    if fallback_prefix and fallback_prefix not in prefixes:
        prefixes.insert(0, fallback_prefix)
    if fallback_suffix and fallback_suffix not in suffixes:
        suffixes.append(fallback_suffix)
    return "+".join(prefixes), "+".join(suffixes)


def infer_stem_from_surface(surface: str, prefix: str, suffix: str, lemma: str) -> str:
    w = clean_word(surface)
    for p in [x for x in prefix.split("+") if x]:
        if w.startswith(p) and len(w) > len(p):
            w = w[len(p):]
    for s in reversed([x for x in suffix.split("+") if x]):
        if w.endswith(s) and len(w) > len(s):
            w = w[:-len(s)]
    return w or clean_word(strip_diacritics(lemma)) or clean_word(surface)


def camel_token(surface: str, part: str, position: str, ana: Dict[str, Any]) -> Dict[str, Any]:
    clean = clean_word(part)
    n = normalize(clean)
    pos = force_pos_from_surface(surface, ana, map_camel_pos(str(ana.get("pos", "")), clean))

    # For one-letter clitics and known particles, avoid misleading CAMeL stems/prefixes.
    if pos == "حرف":
        return {
            "position": position,
            "surface": surface,
            "word": clean,
            "normalized": n,
            "lemma": n,
            "root": n,
            "pattern": "حرف",
            "pos": "حرف",
            "prefix": "",
            "stem": clean,
            "suffix": "",
            "features": arabic_camel_features(ana, clean, "حرف"),
            "confidence": "camel-tools+rules",
        }

    fallback_prefix, fallback_stem, fallback_suffix = fallback_stem_prefix_suffix(clean, pos)
    lemma = camel_arabic_value(ana.get("lex") or ana.get("lemma")) or normalize(fallback_stem or clean)
    lemma = strip_diacritics(lemma).replace("_", "")
    prefix, suffix = grammatical_affixes(clean, ana, pos, fallback_prefix, fallback_suffix)
    core_part = lexical_core_part(clean, pos, ana, str(lemma))
    root = robust_root_value(ana.get("root"), clean, str(lemma), core_part, pos)
    stem_raw = camel_arabic_value(ana.get("stem"))
    analyzed_stem = clean_word(strip_diacritics(stem_raw))
    stem = core_part or analyzed_stem or infer_stem_from_surface(clean, prefix, suffix, str(lemma))
    # `bw` is a Buckwalter analysis string, not a morphological pattern.
    pattern = sanitize_pattern_value(camel_arabic_value(ana.get("pattern")), clean, str(lemma), pos)

    return {
        "position": position,
        "surface": surface,
        "word": core_part,
        "normalized": normalize(core_part),
        "lemma": lemma,
        "root": root,
        "pattern": pattern,
        "pos": pos,
        "prefix": prefix,
        "stem": stem,
        "suffix": suffix,
        "features": arabic_camel_features(ana, clean, pos),
        "confidence": "camel-tools",
    }


def camel_analyze_text(text: str, split_clitics: bool = True) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], int]:
    mle = get_camel_mle()
    if mle is None:
        return None, _camel_load_error, 0

    surface_words = words(text)
    try:
        disambig = mle.disambiguate(surface_words)
    except Exception as exc:
        return None, str(exc), 0

    out: List[Dict[str, Any]] = []
    repaired = 0
    for word_index, (surface, item) in enumerate(zip(surface_words, disambig), start=1):
        if not getattr(item, "analyses", None):
            units = expanded_tokens(surface, split_clitics=split_clitics)
            for _, part, position in units:
                out.append(fallback_token(surface, part, position))
            continue

        ana = item.analyses[0].analysis
        units = camel_units_for_word(surface, ana, word_index, split_clitics)
        if len(units) > 1:
            repaired += 1

        # Analyze detachable units contextually only where needed. The lexical core
        # retains the selected whole-word analysis; clitic rows use safe rule labels.
        lexical_indexes = [i for i, (_, part, _) in enumerate(units) if normalize(part) not in VALID_PREFIX_CLITICS and normalize(part) not in PRONOUNS]
        lexical_index = lexical_indexes[0] if lexical_indexes else 0
        for unit_index, (unit_surface, part, position) in enumerate(units):
            npart = normalize(part)
            if npart in VALID_PREFIX_CLITICS or (npart in PRONOUNS and unit_index != lexical_index):
                out.append(fallback_token(unit_surface, part, position))
            else:
                out.append(camel_token(unit_surface, part, position, ana))
    return out, None, repaired


# -----------------------------------------------------------------------------
# Optional Farasa hook
# -----------------------------------------------------------------------------

def farasa_segment(text: str) -> Optional[List[str]]:
    jar = os.environ.get("FARASA_SEGMENTER_JAR")
    if not jar or not os.path.exists(jar):
        return None
    try:
        p = subprocess.run(
            ["java", "-jar", jar],
            input=text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.split()
    except Exception:
        return None
    return None


# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    mle = get_camel_mle() if camel_tools_installed() else None
    return {
        "ok": True,
        "service": "Arabic NLP Service",
        "version": "2.5.1",
        "camel_tools_available": camel_tools_installed(),
        "camel_model_loaded": mle is not None,
        "camel_model_error": None if mle is not None else _camel_load_error,
        "farasa_jar_configured": bool(os.environ.get("FARASA_SEGMENTER_JAR")),
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    units = expanded_tokens(req.text, split_clitics=req.split_clitics)
    engine_used = "fallback-rule-improved"
    tokens: Optional[List[Dict[str, Any]]] = None
    debug: Dict[str, Any] = {}

    if req.engine in {"auto", "camel"}:
        tokens, err, repaired = camel_analyze_text(req.text, split_clitics=req.split_clitics)
        if tokens is not None:
            engine_used = "camel-tools+validated-affixes-roots"
            debug["validated_segmented_words"] = repaired
        elif err:
            debug["camel_error"] = err

    if tokens is None and req.engine in {"auto", "farasa"}:
        seg = farasa_segment(req.text)
        if seg:
            farasa_units = [(clean_word(w), clean_word(w), str(i)) for i, w in enumerate(seg, start=1) if clean_word(w)]
            tokens = [fallback_token(surface, part, position) for surface, part, position in farasa_units]
            engine_used = "farasa-segmenter+fallback-rules"

    if tokens is None:
        tokens = [fallback_token(surface, part, position) for surface, part, position in units]

    pos_counts = Counter(t.get("pos", "غير محدد") for t in tokens)
    result: Dict[str, Any] = {
        "ok": True,
        "engine": engine_used,
        "meta": {
            "token_count": len(tokens),
            "pos_counts": dict(pos_counts),
            "split_clitics": req.split_clitics,
            "segmentation_mode": "camel-confirmed" if engine_used.startswith("camel-tools") else "conservative-rules",
        },
        "tokens": tokens,
    }
    if req.include_debug:
        result["debug"] = debug
    return result


@app.post("/classify")
def classify(req: TextRequest) -> Dict[str, Any]:
    txt = normalize(req.text)
    scores: Dict[str, int] = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(txt.count(normalize(k)) for k in kws)
    if not any(scores.values()):
        scores["عام"] = 1
    total = sum(scores.values()) or 1
    ranked = sorted(
        [{"label": k, "score": v, "confidence": round(v / total, 4)} for k, v in scores.items()],
        key=lambda x: x["score"],
        reverse=True,
    )
    return {"ok": True, "engine": "keyword-classifier", "classification": ranked[: req.max_items]}


@app.post("/summarize")
def summarize(req: TextRequest) -> Dict[str, Any]:
    sents = sentences(req.text)
    ws = [normalize(w) for w in words(req.text)]
    freq = Counter(w for w in ws if w not in STOPWORDS and len(w) > 2)
    scored = []
    for idx, sent in enumerate(sents):
        toks = [normalize(w) for w in words(sent)]
        score = sum(freq.get(t, 0) for t in toks) / max(1, len(toks))
        scored.append((score, idx, sent))
    top_n = max(1, min(req.max_items or 3, 8, math.ceil(len(sents) * 0.25) if sents else 1))
    selected = sorted(sorted(scored, reverse=True)[:top_n], key=lambda x: x[1])
    summary = " ".join(sent for _, _, sent in selected)
    return {"ok": True, "engine": "extractive-frequency", "summary": summary, "sentences": [s for _, _, s in selected]}


@app.post("/suggest_tags")
def suggest_tags(req: TextRequest) -> Dict[str, Any]:
    units = expanded_tokens(req.text, split_clitics=True)
    terms = [normalize(part) for _, part, _ in units]
    freq = Counter(w for w in terms if len(w) > 2 and w not in STOPWORDS and w not in PREPOSITIONS and w not in PARTICLES)
    tags = [{"tag": w, "score": c} for w, c in freq.most_common(req.max_items or 20)]
    return {"ok": True, "engine": "tf-frequency-tags", "tags": tags}
