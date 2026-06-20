import re
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Arabic NLP Service")

class TextRequest(BaseModel):
    text: str
    text_type: str | None = "عام"

ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
PUNCT = re.compile(r"[^\u0621-\u063A\u0641-\u064A\s]")

PREPOSITIONS = {
    "من", "الى", "إلى", "الي", "عن", "على", "علي", "في", "ب", "ك", "ل",
    "حتى", "حتي", "رب", "منذ", "خلا", "عدا", "حاشا"
}

CONJUNCTIONS = {
    "و", "ف", "ثم", "او", "أو", "ام", "أم", "بل", "لكن"
}

PARTICLES = {
    "ان", "أن", "إن", "لن", "لم", "لا", "ما", "قد", "هل", "ليت", "لعل",
    "كي", "حتى", "حتي", "اذا", "إذا", "اذ", "إذ"
}

PRONOUNS = {
    "انا", "أنا", "نحن", "انت", "أنت", "انتم", "أنتم", "هو", "هي", "هم", "هن",
    "كما", "كم", "كن", "نا", "ي", "ك", "ه", "ها", "هما", "هم", "هن"
}

DEMONSTRATIVES = {
    "هذا", "هذه", "ذلك", "تلك", "هؤلاء", "اولئك", "أولئك", "هنا", "هناك"
}

RELATIVES = {
    "الذي", "التي", "اللذان", "اللتان", "الذين", "اللاتي", "اللواتي", "من", "ما"
}

QUESTION_NAMES = {
    "من", "ما", "ماذا", "متى", "متي", "اين", "أين", "كيف", "كم", "اي", "أي", "ايها", "أيها"
}

COMMON_VERB_PREFIXES = ("س", "ي", "ت", "ن", "أ", "ا")
ATTACHED_PREFIXES = ("وال", "فال", "بال", "كال", "لل", "و", "ف", "ب", "ك", "ل")
ATTACHED_SUFFIXES = ("كما", "هما", "كم", "كن", "نا", "ها", "هم", "هن", "ه", "ك", "ي", "ا")

KNOWN_VERBS = {
    "كان", "كنت", "وجد", "وجدت", "سأل", "سألت", "مضى", "خرج", "عاش", "حل",
    "بنى", "بناه", "ترك", "تركت", "أصبح", "أصبحت", "أحمل", "أضم", "أدرك",
    "استعدوا", "يستمر", "أستيقظ", "يهمك", "تحتاج", "يقلب", "أقلب", "انتظر",
    "أنتظر", "أسمع", "يأتي", "يأتينا", "يخبر", "ليخبرنا", "يهذي", "يجتاحني"
}

def strip_diacritics(text: str) -> str:
    return ARABIC_DIACRITICS.sub("", text)

def normalize_arabic(text: str) -> str:
    text = strip_diacritics(text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    text = text.replace("ة", "ة")
    return text

def clean_text(text: str) -> str:
    text = strip_diacritics(text)
    text = PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_particle(word: str) -> bool:
    n = normalize_arabic(word)
    return word in PREPOSITIONS or n in PREPOSITIONS or word in CONJUNCTIONS or n in CONJUNCTIONS or word in PARTICLES or n in PARTICLES

def particle_type(word: str) -> str:
    n = normalize_arabic(word)
    if word in PREPOSITIONS or n in PREPOSITIONS:
        return "نوع الأداة: حرف جر"
    if word in CONJUNCTIONS or n in CONJUNCTIONS:
        return "نوع الحرف: حرف عطف"
    return "نوع الأداة: حرف"

def is_special_noun(word: str) -> str | None:
    n = normalize_arabic(word)
    if word in PRONOUNS or n in PRONOUNS:
        return "نوع الضمير/الاسم: ضمير شخصي"
    if word in DEMONSTRATIVES or n in DEMONSTRATIVES:
        return "نوع الضمير/الاسم: اسم إشارة"
    if word in RELATIVES or n in RELATIVES:
        return "نوع الضمير/الاسم: اسم موصول"
    if word in QUESTION_NAMES or n in QUESTION_NAMES:
        return "نوع الضمير/الاسم: اسم استفهام"
    return None

def split_attached(word: str) -> List[str]:
    original = word
    n = normalize_arabic(word)

    if is_particle(word):
        return [word]

    parts = []

    # handle و / ف before a real word
    if len(word) > 2 and word[0] in ("و", "ف"):
        rest = word[1:]
        if rest and not is_particle(word):
            parts.append(word[0])
            word = rest
            n = normalize_arabic(word)

    # handle single-letter prepositions
    if len(word) > 3 and word[0] in ("ب", "ك", "ل"):
        rest = word[1:]
        parts.append(word[0])
        word = rest
        n = normalize_arabic(word)

    # suffix pronouns
    suffix = ""
    for s in ATTACHED_SUFFIXES:
        if len(word) > len(s) + 2 and word.endswith(s):
            suffix = s
            word = word[:-len(s)]
            break

    if word:
        parts.append(word)
    if suffix:
        parts.append(suffix)

    return parts if parts else [original]

def guess_pos(part: str) -> tuple[str, str]:
    n = normalize_arabic(part)

    if is_particle(part):
        return "حرف", particle_type(part)

    special = is_special_noun(part)
    if special:
        return "اسم", special

    if n.startswith("ال") and len(n) > 3:
        return "اسم", "التعريف: معرفة"

    if part in ATTACHED_SUFFIXES or n in PRONOUNS:
        return "اسم", "نوع الضمير/الاسم: ضمير شخصي"

    if part in KNOWN_VERBS or n in {normalize_arabic(v) for v in KNOWN_VERBS}:
        return "فعل", "صيغة الفعل: فعل مصرف"

    if n.startswith(("ي", "ت", "ن", "ا")) and len(n) >= 4:
        # avoid classifying common nouns as verbs too aggressively
        if not n.startswith("ال"):
            return "فعل", "زمن/تمام الفعل: غير تام / مضارع، صيغة الفعل: فعل مصرف"

    if n.endswith(("ت")) and len(n) >= 3:
        return "فعل", "زمن/تمام الفعل: تام / ماض، صيغة الفعل: فعل مصرف"

    return "اسم", "التعريف: نكرة"

def guess_root(stem: str) -> str:
    n = normalize_arabic(stem)
    n = re.sub(r"^ال", "", n)
    n = re.sub(r"^(و|ف|ب|ك|ل)", "", n)
    n = re.sub(r"(كما|هما|كم|كن|نا|ها|هم|هن|ه|ك|ي)$", "", n)
    letters = [c for c in n if re.match(r"[\u0621-\u063A\u0641-\u064A]", c)]
    return "".join(letters[:3]) if letters else n

def analyze_fallback(text: str) -> Dict[str, Any]:
    text = clean_text(text)
    words = text.split()
    tokens = []
    pos_counts = {}
    pos = 1

    for surface in words:
        parts = split_attached(surface)
        sub = 1

        for part in parts:
            clean_part = clean_text(part)
            if not clean_part:
                continue

            part_pos, features = guess_pos(clean_part)
            pos_counts[part_pos] = pos_counts.get(part_pos, 0) + 1

            n = normalize_arabic(clean_part)
            prefix = ""
            stem = clean_part
            suffix = ""

            if clean_part in ("و", "ف", "ب", "ك", "ل"):
                prefix = ""
                stem = clean_part
            elif part_pos == "اسم" and n.startswith("ال") and len(n) > 3:
                prefix = "ال"
                stem = clean_part[2:]

            for s in ATTACHED_SUFFIXES:
                if len(stem) > len(s) + 2 and stem.endswith(s):
                    suffix = s
                    stem = stem[:-len(s)]
                    break

            tokens.append({
                "position": f"{pos}.{sub}",
                "surface": surface,
                "word": clean_part,
                "normalized": n,
                "lemma": stem if stem else clean_part,
                "root": guess_root(stem if stem else clean_part),
                "pattern": "غير محدد",
                "pos": part_pos,
                "prefix": prefix,
                "stem": stem,
                "suffix": suffix,
                "features": features,
                "confidence": "fallback-rule-improved"
            })

            sub += 1

        pos += 1

    return {
        "ok": True,
        "engine": "fallback-rule-improved",
        "meta": {
            "token_count": len(tokens),
            "pos_counts": pos_counts
        },
        "tokens": tokens
    }

@app.get("/health")
def health():
    return {
        "ok": True,
        "engine": "fallback-rule-improved",
        "camel_available": False
    }

@app.post("/analyze")
def analyze(req: TextRequest):
    return analyze_fallback(req.text)

@app.post("/classify")
def classify(req: TextRequest):
    text = req.text
    if any(w in text for w in ["قال", "رأى", "كان", "ذهب", "عاد"]):
        label = "سرد / قصة"
    elif any(w in text for w in ["بحث", "دراسة", "منهج", "نتائج"]):
        label = "نص أكاديمي"
    elif any(w in text for w in ["أيها", "الحمد", "السلام عليكم"]):
        label = "خطبة / خطاب"
    else:
        label = "عام"
    return {"ok": True, "label": label, "confidence": "rule-based"}

@app.post("/summarize")
def summarize(req: TextRequest):
    clean = clean_text(req.text)
    sentences = re.split(r"[.!؟?؛،]+|\n+", req.text)
    sentences = [s.strip() for s in sentences if s.strip()]
    summary = " ".join(sentences[:3]) if sentences else clean[:300]
    return {"ok": True, "summary": summary}

@app.post("/suggest_tags")
def suggest_tags(req: TextRequest):
    clean = clean_text(req.text)
    words = [normalize_arabic(w) for w in clean.split() if len(w) > 3]
    freq = {}
    for w in words:
        if w in PREPOSITIONS or w in PARTICLES or w in CONJUNCTIONS:
            continue
        freq[w] = freq.get(w, 0) + 1
    tags = sorted(freq, key=freq.get, reverse=True)[:10]
    return {"ok": True, "tags": tags}
