"""
خدمة FastAPI احترافية اختيارية لمعالجة العربية.

التشغيل:
  pip install fastapi uvicorn pydantic scikit-learn numpy
  pip install camel-tools        # اختياري ومفضل للتحليل الصرفي
  uvicorn api_nlp_service:app --host 127.0.0.1 --port 8000

دعم Farasa اختياري عبر متغيرات البيئة إذا كان لديك JAR أو خدمة خارجية:
  FARASA_SEGMENTER_JAR=/path/farasa-segmenter.jar

النهايات:
  GET  /health
  POST /analyze
  POST /classify
  POST /summarize
  POST /suggest_tags
  GET  /docs  Swagger تلقائي من FastAPI
  GET  /openapi.json
"""
from __future__ import annotations

import math
import os
import re
import subprocess
from collections import Counter
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Arabic Linguistic Platform NLP API",
    version="2.0.0",
    description="Arabic NLP service with optional CAMeL Tools, optional Farasa hook, rule-based fallback, classification, summarization, and tag suggestion.",
)

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = "ar"
    max_items: int = 20

class AnalyzeRequest(TextRequest):
    engine: str = "auto"  # auto | camel | farasa | fallback

DIAC = re.compile(r"[\u064B-\u0652\u0670\u0640]")
AR_LETTERS = re.compile(r"[\u0621-\u064A\u066E-\u06D3\u06FA-\u06FC]+", re.UNICODE)
SENT_SPLIT = re.compile(r"[.!؟؛\n]+")

STOPWORDS = set("""
من إلى الى عن على في ثم أو او أم ام بل لكن لا لم لن قد هل حتى إذا اذا إن ان أن ما كي لعل ليت رب هذا هذه ذلك تلك هو هي هم هن نحن انا أنا كان كانت يكون كل غير بعد قبل بين عند مع كما وقد فقد و ف ب ك ل
""".split())
PREPOSITIONS = {"من", "الى", "إلى", "عن", "على", "في", "ب", "ك", "ل", "رب", "حتى"}
CONJUNCTIONS = {"و", "ف", "ثم", "أو", "او", "أم", "ام", "بل", "لكن"}
PRONOUNS = {"انا", "أنا", "انت", "أنت", "هو", "هي", "نحن", "هم", "هن", "هما", "ت", "نا", "ي", "ك", "ه", "ها", "كم", "كن"}
DEMONSTRATIVES = {"هذا", "هذه", "ذلك", "تلك", "هؤلاء", "هنا", "هناك"}
RELATIVES = {"الذي", "التي", "الذين", "اللاتي", "اللائي", "اللذان", "اللتان"}
INTERROGATIVES = {"من", "ما", "ماذا", "متى", "اين", "أين", "كيف", "كم", "أي", "اي", "لماذا"}

CATEGORY_KEYWORDS = {
    "شعر": ["قصيدة", "بيت", "قافية", "شاعر", "أبيات", "بحر", "غزل"],
    "خطبة": ["أيها", "الحمد", "السلام", "الناس", "أوصيكم", "أما بعد"],
    "أكاديمي": ["بحث", "دراسة", "منهج", "نتائج", "مراجع", "فرضية", "تحليل"],
    "إعلامي": ["قال", "أعلن", "أخبار", "تقرير", "مصادر", "صحيفة", "وكالة"],
    "تعليمي": ["درس", "شرح", "تمرين", "الطلاب", "تعلم", "المعلم", "الصف"],
    "سردي": ["كان", "كنت", "قال", "ذهب", "عاد", "رأى", "وجد", "فجأة"],
    "تراثي": ["قال", "روي", "حدثنا", "باب", "فصل", "رحمه", "كتاب"],
}

def normalize(text: str) -> str:
    text = DIAC.sub("", text or "")
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    return text

def clean_word(word: str) -> str:
    m = AR_LETTERS.findall(DIAC.sub("", word or ""))
    return "".join(m)

def words(text: str) -> List[str]:
    return [w for w in (clean_word(x) for x in AR_LETTERS.findall(text or "")) if w]

def sentences(text: str) -> List[str]:
    return [s.strip() for s in SENT_SPLIT.split(text or "") if s.strip()]

def fallback_pos(w: str) -> str:
    n = normalize(w)
    if n in {normalize(x) for x in PREPOSITIONS | CONJUNCTIONS} or len(n) == 1 and n in {"و", "ف", "ب", "ك", "ل"}:
        return "حرف"
    if n in {normalize(x) for x in PRONOUNS | DEMONSTRATIVES | RELATIVES | INTERROGATIVES}:
        return "اسم"
    if n.startswith("ال") or n.endswith(("ة", "ات", "ون", "ين")):
        return "اسم"
    if n.startswith(("ي", "ت", "ن", "ا")) and len(n) >= 4:
        return "فعل"
    if n.startswith(("س", "سن")) and len(n) >= 4:
        return "فعل"
    return "اسم"

def fallback_token(w: str, i: int) -> Dict[str, Any]:
    n = normalize(w)
    pos = fallback_pos(w)
    features = []
    if pos == "حرف":
        features.append("نوع الأداة: حرف جر" if n in {normalize(x) for x in PREPOSITIONS} else "نوع الحرف: حرف عطف/أداة")
    elif pos == "فعل":
        features.append("صيغة الفعل: فعل مصرف")
        features.append("زمن/تمام الفعل: غير تام / مضارع" if n.startswith(("ي", "ت", "ن", "ا")) else "زمن/تمام الفعل: تام / ماض")
    else:
        if n.startswith("ال"):
            features.append("التعريف: معرفة")
        if n in {normalize(x) for x in PRONOUNS}:
            features.append("نوع الضمير/الاسم: ضمير شخصي")
        if n in {normalize(x) for x in DEMONSTRATIVES}:
            features.append("نوع الضمير/الاسم: اسم إشارة")
        if n in {normalize(x) for x in RELATIVES}:
            features.append("نوع الضمير/الاسم: اسم موصول")
        if n in {normalize(x) for x in INTERROGATIVES}:
            features.append("نوع الضمير/الاسم: اسم استفهام")
    return {
        "position": i,
        "surface": w,
        "word": w,
        "normalized": n,
        "lemma": n,
        "root": n[:3] if len(n) >= 3 else n,
        "pattern": "غير محدد",
        "pos": pos,
        "prefix": "ال" if n.startswith("ال") else "",
        "stem": n[2:] if n.startswith("ال") else n,
        "suffix": "",
        "features": "، ".join(features),
        "confidence": "fallback-rule",
    }

_camel_mle = None

def camel_available() -> bool:
    try:
        import camel_tools  # noqa: F401
        return True
    except Exception:
        return False

def camel_analyze(ws: List[str]) -> Optional[List[Dict[str, Any]]]:
    global _camel_mle
    try:
        from camel_tools.disambig.mle import MLEDisambiguator
        if _camel_mle is None:
            _camel_mle = MLEDisambiguator.pretrained()
        disambig = _camel_mle.disambiguate(ws)
        out = []
        for i, item in enumerate(disambig, start=1):
            tok = ws[i - 1]
            if not item.analyses:
                out.append(fallback_token(tok, i)); continue
            ana = item.analyses[0].analysis
            pos_en = ana.get("pos", "")
            pos = "فعل" if str(pos_en).startswith("verb") else ("حرف" if pos_en in {"prep", "conj", "part", "punc"} else "اسم")
            out.append({
                "position": i,
                "surface": tok,
                "word": tok,
                "normalized": normalize(tok),
                "lemma": ana.get("lex") or ana.get("lemma") or normalize(tok),
                "root": ana.get("root") or "",
                "pattern": ana.get("pattern") or ana.get("bw") or "",
                "pos": pos,
                "prefix": ana.get("prc3", "") or ana.get("prc2", "") or ana.get("prc1", "") or ana.get("prc0", ""),
                "stem": ana.get("stem") or normalize(tok),
                "suffix": ana.get("enc0", ""),
                "features": ", ".join([f"{k}={v}" for k, v in ana.items() if k in {"asp", "per", "gen", "num", "stt", "cas", "vox"} and v]),
                "confidence": "camel-tools",
            })
        return out
    except Exception:
        return None

def farasa_segment(text: str) -> Optional[List[str]]:
    jar = os.environ.get("FARASA_SEGMENTER_JAR")
    if not jar or not os.path.exists(jar):
        return None
    try:
        p = subprocess.run(["java", "-jar", jar], input=text, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.split()
    except Exception:
        return None
    return None

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "Arabic NLP Service",
        "version": "2.0.0",
        "camel_tools_available": camel_available(),
        "farasa_jar_configured": bool(os.environ.get("FARASA_SEGMENTER_JAR")),
    }

@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    ws = words(req.text)
    engine_used = "fallback-rule"
    tokens: Optional[List[Dict[str, Any]]] = None
    if req.engine in {"auto", "camel"}:
        tokens = camel_analyze(ws)
        if tokens is not None:
            engine_used = "camel-tools"
    if tokens is None and req.engine in {"auto", "farasa"}:
        seg = farasa_segment(req.text)
        if seg:
            tokens = [fallback_token(clean_word(w), i) for i, w in enumerate(seg, start=1) if clean_word(w)]
            engine_used = "farasa-segmenter+rules"
    if tokens is None:
        tokens = [fallback_token(w, i) for i, w in enumerate(ws, start=1)]
    pos_counts = Counter(t.get("pos", "غير محدد") for t in tokens)
    return {"ok": True, "engine": engine_used, "meta": {"token_count": len(tokens), "pos_counts": dict(pos_counts)}, "tokens": tokens}

@app.post("/classify")
def classify(req: TextRequest) -> Dict[str, Any]:
    txt = normalize(req.text)
    scores = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(txt.count(normalize(k)) for k in kws)
    if not any(scores.values()):
        scores["عام"] = 1
    total = sum(scores.values()) or 1
    ranked = sorted(([{"label": k, "score": v, "confidence": round(v / total, 4)} for k, v in scores.items()]), key=lambda x: x["score"], reverse=True)
    return {"ok": True, "engine": "keyword-ml-fallback", "classification": ranked[: req.max_items]}

@app.post("/summarize")
def summarize(req: TextRequest) -> Dict[str, Any]:
    sents = sentences(req.text)
    ws = [normalize(w) for w in words(req.text)]
    freq = Counter(w for w in ws if w not in {normalize(x) for x in STOPWORDS})
    scored = []
    for idx, s in enumerate(sents):
        toks = [normalize(w) for w in words(s)]
        score = sum(freq.get(t, 0) for t in toks) / max(1, len(toks))
        scored.append((score, idx, s))
    top_n = max(1, min(req.max_items or 3, 8, math.ceil(len(sents) * 0.25) if sents else 1))
    selected = sorted(sorted(scored, reverse=True)[:top_n], key=lambda x: x[1])
    summary = " ".join(s for _, _, s in selected)
    return {"ok": True, "engine": "extractive-frequency", "summary": summary, "sentences": [s for _, _, s in selected]}

@app.post("/suggest_tags")
def suggest_tags(req: TextRequest) -> Dict[str, Any]:
    ws = [normalize(w) for w in words(req.text)]
    freq = Counter(w for w in ws if len(w) > 2 and w not in {normalize(x) for x in STOPWORDS})
    tags = [{"tag": w, "score": c} for w, c in freq.most_common(req.max_items or 20)]
    return {"ok": True, "engine": "tf-frequency-tags", "tags": tags}
