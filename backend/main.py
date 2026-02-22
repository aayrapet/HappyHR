from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func
from pydantic import BaseModel
from typing import Optional
import secrets
import os
import json
import asyncio
import struct
import base64
import re
import unicodedata
import fitz  # PyMuPDF
from dotenv import load_dotenv
import websockets
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

load_dotenv()

from database import init_db, get_db
from models import JobConfig, Candidate, InterviewResult
from email_service import send_interview_invite, send_rejection_email, send_decision_email

app = FastAPI(title="HappyHR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await init_db()


# ── Helpers ──────────────────────────────────────────────
_UNICODE_DOT_MAP = str.maketrans({"·": ".", "∙": ".", "•": ".", "․": "."})
_NON_WORD_RE = re.compile(r"\W")
_TOKEN_RE = re.compile(r"[a-z0-9.+#/:-]+")
_TECH_TOKEN_RE = re.compile(r"^[a-z0-9.][a-z0-9.+#/:-]*[a-z0-9+#]$")
_STOP = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "with", "as", "at", "by",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those",
    "experience", "years",
}
_NLP = None
_EMBEDDER = None
_KW_EMB_CACHE: dict[tuple[str, ...], np.ndarray] = {}


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    return _NLP


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def normalize(s: str) -> str:
    s = s.lower()
    s = s.translate(_UNICODE_DOT_MAP)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def prepare_keywords(keywords: list[str]) -> list[tuple[str, bool, Optional[str], Optional[tuple[str, ...]], re.Pattern]]:
    nlp = _get_nlp()
    out: list[tuple[str, bool, Optional[str], Optional[tuple[str, ...]], re.Pattern]] = []

    for kw in keywords:
        kw_n = normalize(kw).strip()
        if not kw_n:
            continue
        is_phrase = " " in kw_n
        surface_pat = re.compile(rf"(?<!\w){re.escape(kw_n)}(?!\w)")

        if is_phrase:
            lemmas: list[str] = []
            for t in nlp(kw_n):
                if t.is_space or t.is_punct:
                    continue
                lem = normalize(t.lemma_)
                if not lem or lem == "-pron-":
                    continue
                lemmas.append(lem)
            out.append((kw_n, True, None, tuple(lemmas), surface_pat))
            continue

        if _NON_WORD_RE.search(kw_n):
            out.append((kw_n, False, None, None, surface_pat))
            continue

        kw_doc = nlp(kw_n)
        kw_lemma = normalize(kw_doc[0].lemma_) if len(kw_doc) else kw_n
        out.append((kw_n, False, kw_lemma, None, surface_pat))

    return out


def keyword_match_lemmatized(cv_text: str, keywords: list[str]) -> float:
    prepared = prepare_keywords(keywords)
    if not prepared:
        return 0.0

    nlp = _get_nlp()
    surface = normalize(cv_text)
    lemmas_seq: list[str] = []
    for t in nlp(surface):
        if t.is_space or t.is_punct:
            continue
        lem = normalize(t.lemma_)
        if not lem or lem == "-pron-":
            continue
        lemmas_seq.append(lem)

    lemmas_set = set(lemmas_seq)
    lemmas_joined = " " + " ".join(lemmas_seq) + " "

    matched = 0
    for _, is_phrase, single_lemma, phrase_lemmas, surface_pattern in prepared:
        if is_phrase:
            phrase = " " + " ".join(phrase_lemmas or ()) + " "
            if phrase.strip() and phrase in lemmas_joined:
                matched += 1
                continue
            if surface_pattern.search(surface):
                matched += 1
            continue

        if single_lemma and single_lemma in lemmas_set:
            matched += 1
            continue
        if surface_pattern.search(surface):
            matched += 1

    return matched / len(prepared)


async def build_screening_feedback(cv_text: str, job: JobConfig) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    scoring_model = os.getenv("SCORING_MODEL", "gemini-2.5-flash-lite")
    fallback = (
        "Thank you for your application. Your profile shows relevant strengths, "
        "and we encourage you to apply again as your experience continues to grow."
    )
    if not api_key:
        return fallback

    prompt = f"""You are an HR assistant.
Write a short and polite rejection feedback for CV screening in exactly 2 sentences.
Tone: professional and encouraging. Do not mention internal hiring policy.

Job title: {job.title}
Required skills: {json.dumps(job.keywords[:12])}
Candidate CV excerpt:
{cv_text[:1800]}

Return only plain text."""
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=scoring_model, contents=prompt)
        text = (response.text or "").strip()
        return text or fallback
    except Exception:
        return fallback


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def extract_candidates(cv_text: str) -> list[str]:
    raw = normalize(cv_text).replace("\n", " ")
    tokens = _TOKEN_RE.findall(raw)

    token_seq: list[str] = []
    for tok in tokens:
        if len(tok) < 2:
            continue
        if tok.isdigit():
            continue
        if _TECH_TOKEN_RE.match(tok) or tok.isalpha():
            token_seq.append(tok)

    filtered_seq = [tok for tok in token_seq if tok not in _STOP]
    ngrams: list[str] = []
    for n in (2, 3):
        for i in range(len(token_seq) - n + 1):
            window = token_seq[i : i + n]
            if all(tok in _STOP for tok in window):
                continue
            ngrams.append(" ".join(window))

    seen: set[str] = set()
    out: list[str] = []
    for cand in filtered_seq + ngrams:
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def _keyword_embeddings(kws: list[str]) -> np.ndarray:
    key = tuple(kws)
    cached = _KW_EMB_CACHE.get(key)
    if cached is not None:
        return cached
    emb = _get_embedder().encode(kws, convert_to_numpy=True, normalize_embeddings=False)
    _KW_EMB_CACHE[key] = emb
    return emb


def embedding_screen(
    cv_text: str,
    keywords: list[str],
    threshold_single: float = 0.75,
    threshold_phrase: float = 0.75,
) -> float:
    clean_keywords = [normalize(kw).strip() for kw in keywords if kw and kw.strip()]
    if not clean_keywords:
        return 0.0

    surface = normalize(cv_text)
    candidates = extract_candidates(cv_text)
    if not candidates:
        return 0.0

    candidate_set = set(candidates)
    kw_emb = _keyword_embeddings(clean_keywords)
    cand_emb = _get_embedder().encode(candidates, convert_to_numpy=True, normalize_embeddings=False)
    sims = cosine_sim_matrix(kw_emb, cand_emb)

    hit = 0
    for i, kw in enumerate(clean_keywords):
        if " " in kw:
            if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", surface):
                hit += 1
                continue
        elif kw in candidate_set:
            hit += 1
            continue

        j = int(np.argmax(sims[i]))
        best_sim = float(sims[i, j])
        thr = threshold_phrase if " " in kw else threshold_single
        if best_sim >= thr:
            hit += 1

    return hit / len(clean_keywords)


def extract_pdf_text(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def keyword_match(cv_text: str, keywords: list[str]) -> float:
    clean_keywords = [kw for kw in keywords if kw and kw.strip()]
    if not clean_keywords:
        return 0.0

    lexical_weight = float(os.getenv("MATCH_WEIGHT_LEXICAL", "0.6"))
    embedding_weight = float(os.getenv("MATCH_WEIGHT_EMBEDDING", "0.4"))
    emb_threshold_single = float(os.getenv("MATCH_EMBED_THRESHOLD_SINGLE", "0.75"))
    emb_threshold_phrase = float(os.getenv("MATCH_EMBED_THRESHOLD_PHRASE", "0.75"))

    total_weight = lexical_weight + embedding_weight
    if total_weight <= 0:
        lexical_weight, embedding_weight, total_weight = 0.6, 0.4, 1.0

    lexical_score = keyword_match_lemmatized(cv_text, clean_keywords)
    try:
        embedding_score = embedding_screen(
            cv_text=cv_text,
            keywords=clean_keywords,
            threshold_single=emb_threshold_single,
            threshold_phrase=emb_threshold_phrase,
        )
    except Exception as e:
        print(f"Embedding screen fallback to lexical score: {e}")
        embedding_score = lexical_score

    return ((lexical_weight * lexical_score) + (embedding_weight * embedding_score)) / total_weight


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _clamp_float(value, minimum: float, maximum: float, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(minimum, min(maximum, numeric))


def _normalize_string_list(value, max_items: int = 12, max_len: int = 160) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(text[:max_len])
        if len(out) >= max_items:
            break
    return out


def _normalize_keyword_hits(value) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    for item in value:
        keyword = ""
        evidence = ""
        if isinstance(item, dict):
            keyword = str(item.get("keyword", "")).strip()
            evidence = str(item.get("evidence", "")).strip()
        elif isinstance(item, str):
            keyword = item.strip()
        if not keyword:
            continue
        out.append({"keyword": keyword[:120], "evidence": evidence[:400]})
    return out


def _normalize_live_memory_event(raw_args, tool_call_id: str, event_index: int | None = None) -> dict:
    args = raw_args
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    if not isinstance(args, dict):
        args = {}

    question_id_raw = str(args.get("question_id", "")).strip().lower()
    if not question_id_raw:
        question_id_raw = "unknown"
    question_id = re.sub(r"[^a-z0-9_]+", "_", question_id_raw).strip("_") or "unknown"
    question_type = str(args.get("question_type", "")).strip().lower()
    if question_type not in {"mandatory", "cv", "followup", "candidate_question", "other"}:
        if question_id == "cv_q":
            question_type = "cv"
        elif re.match(r"^q\d+$", question_id):
            question_type = "mandatory"
        elif question_id.startswith("fup_"):
            question_type = "followup"
        else:
            question_type = "other"

    question_text = str(args.get("question_text", "")).strip()[:280]
    answer_summary = str(args.get("answer_summary", "")).strip()[:1200]

    event = {
        "tool_call_id": tool_call_id,
        "event_index": int(event_index) if event_index is not None else None,
        "question_id": question_id,
        "question_type": question_type,
        "question_text": question_text,
        "answer_summary": answer_summary,
        "question_score": _clamp_float(args.get("question_score"), 0.0, 5.0, 2.5),
        "keyword_hits": _normalize_keyword_hits(args.get("keyword_hits")),
        "expected_themes_covered": _normalize_string_list(args.get("expected_themes_covered")),
        "expected_themes_missing": _normalize_string_list(args.get("expected_themes_missing")),
        "experience_signal": _clamp_float(args.get("experience_signal"), 0.0, 10.0, 5.0),
        "technical_signal": _clamp_float(args.get("technical_signal"), 0.0, 10.0, 5.0),
        "communication_signal": _clamp_float(args.get("communication_signal"), 0.0, 10.0, 5.0),
        "strengths": _normalize_string_list(args.get("strengths")),
        "weaknesses": _normalize_string_list(args.get("weaknesses")),
        "red_flags": _normalize_string_list(args.get("red_flags")),
        "confidence": _clamp_float(args.get("confidence"), 0.0, 1.0, 0.6),
    }
    return event


def _question_sort_key(question_id: str):
    if question_id == "cv_q":
        return (0, 0)
    match = re.match(r"^q(\d+)$", question_id)
    if match:
        return (1, int(match.group(1)))
    return (2, question_id)


def _collapse_live_memory_events(events: list[dict]) -> list[dict]:
    latest_by_question: dict[str, dict] = {}
    for event in events:
        qid = str(event.get("question_id", "")).strip()
        if not qid:
            continue
        latest_by_question[qid] = event
    return sorted(latest_by_question.values(), key=lambda e: _question_sort_key(str(e.get("question_id", ""))))


def _sort_live_memory_events(events: list[dict]) -> list[dict]:
    def sort_key(event: dict):
        raw_idx = event.get("event_index")
        if isinstance(raw_idx, int):
            idx = raw_idx
        else:
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                idx = 10**9
        return (idx, _question_sort_key(str(event.get("question_id", ""))))

    return sorted(events, key=sort_key)


def _weighted_average(values: list[tuple[float, float]], default: float) -> float:
    if not values:
        return default
    weighted_sum = 0.0
    total_weight = 0.0
    for value, weight in values:
        w = max(0.0, float(weight))
        weighted_sum += float(value) * w
        total_weight += w
    if total_weight <= 0:
        return sum(float(v) for v, _ in values) / len(values)
    return weighted_sum / total_weight


def _recommendation_from_global(global_score: float) -> str:
    if global_score < 55:
        return "no"
    if global_score < 65:
        return "maybe"
    if global_score < 80:
        return "yes"
    return "strong_yes"


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _build_live_memory_tools() -> list[dict]:
    """Return tool definitions in OpenAI Realtime API format."""
    return [
        {
            "type": "function",
            "name": "record_question_assessment",
            "description": (
                "Record a structured, non-verbatim assessment right after a candidate "
                "answer is completed for one interview question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "string"},
                    "question_type": {"type": "string"},
                    "question_text": {"type": "string"},
                    "answer_summary": {"type": "string"},
                    "question_score": {"type": "number"},
                    "keyword_hits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "keyword": {"type": "string"},
                                "evidence": {"type": "string"},
                            },
                            "required": ["keyword", "evidence"],
                        },
                    },
                    "expected_themes_covered": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "expected_themes_missing": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "experience_signal": {"type": "number"},
                    "technical_signal": {"type": "number"},
                    "communication_signal": {"type": "number"},
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "weaknesses": {"type": "array", "items": {"type": "string"}},
                    "red_flags": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "question_id",
                    "question_type",
                    "question_text",
                    "answer_summary",
                    "question_score",
                    "keyword_hits",
                    "expected_themes_covered",
                    "expected_themes_missing",
                    "experience_signal",
                    "technical_signal",
                    "communication_signal",
                    "strengths",
                    "weaknesses",
                    "red_flags",
                    "confidence",
                ],
            },
        },
        {
            "type": "function",
            "name": "end_interview",
            "description": (
                "End the interview session. Call this ONCE after you have said your "
                "final goodbye to the candidate, or if the candidate explicitly "
                "requests to stop, or if the candidate is not authorized to work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the interview is ending: 'completed', 'candidate_requested', or 'not_authorized'",
                    }
                },
                "required": ["reason"],
            },
        },
    ]


def validate_live_memory_events(
    events: list[dict], mandatory_questions: list, min_events: int
) -> tuple[bool, str]:
    if len(events) < min_events:
        return False, f"insufficient_live_memory_events:{len(events)}<{min_events}"

    mandatory_ids = {f"q{i + 1}" for i, _ in enumerate(mandatory_questions or [])}
    if mandatory_ids and not any(e.get("question_id") in mandatory_ids for e in events):
        return False, "no_mandatory_question_assessment"

    collapsed = _collapse_live_memory_events(events)
    if not collapsed:
        return False, "no_valid_live_memory_events"
    return True, "ok"


def build_structured_transcript_from_live_memory(
    transcript_parts: list[dict], events: list[dict]
) -> str:
    lines = ["[Structured non-verbatim interview notes]", ""]

    ai_lines = [
        p.get("text", "").strip()
        for p in sorted(transcript_parts, key=lambda p: p.get("index", 0))
        if p.get("role") == "ai" and p.get("text")
    ]
    if ai_lines:
        lines.append("Interviewer prompts:")
        for text in ai_lines:
            lines.append(f"- {text}")
        lines.append("")

    ordered_events = _sort_live_memory_events(events)
    if ordered_events:
        lines.append("Candidate response summaries:")
        for event in ordered_events:
            question_id = event.get("question_id", "unknown")
            question_type = event.get("question_type", "other")
            question_text = event.get("question_text") or "[Question unavailable]"
            answer_summary = event.get("answer_summary") or "[No summary captured]"
            sequence = event.get("event_index")
            seq_prefix = f"#{sequence} " if sequence is not None else ""
            lines.append(f"- {seq_prefix}{question_id} ({question_type}): {question_text}")
            lines.append(f"  Candidate summary: {answer_summary}")
    else:
        lines.append("Candidate response summaries:")
        lines.append("- No structured memory events were captured.")

    return "\n".join(lines).strip()


def score_interview_from_live_memory(events: list[dict], job: JobConfig, candidate: Candidate) -> dict:
    weights = job.scoring_weights or {"experience": 0.4, "technical": 0.4, "communication": 0.2}
    question_text_by_id = {
        f"q{i + 1}": _extract_question_text(q) for i, q in enumerate(job.mandatory_questions or [])
    }
    ordered_events = _sort_live_memory_events(events)

    question_assessments: list[dict] = []
    strengths: list[str] = []
    weaknesses: list[str] = []
    red_flags: list[str] = []
    keyword_evidence_map: dict[str, str] = {}

    for event in ordered_events:
        question_id = event.get("question_id", "unknown")
        question_type = event.get("question_type", "other")
        question_text = event.get("question_text") or question_text_by_id.get(question_id, "")
        question_score = round(_clamp_float(event.get("question_score"), 0.0, 5.0, 2.5), 1)
        experience_signal = round(_clamp_float(event.get("experience_signal"), 0.0, 10.0, 5.0), 1)
        technical_signal = round(_clamp_float(event.get("technical_signal"), 0.0, 10.0, 5.0), 1)
        communication_signal = round(_clamp_float(event.get("communication_signal"), 0.0, 10.0, 5.0), 1)
        confidence = round(_clamp_float(event.get("confidence"), 0.0, 1.0, 0.6), 2)
        expected_themes_covered = _normalize_string_list(event.get("expected_themes_covered"), max_items=20)
        expected_themes_missing = _normalize_string_list(event.get("expected_themes_missing"), max_items=20)

        evidence: list[str] = []
        for hit in event.get("keyword_hits", []):
            keyword = str(hit.get("keyword", "")).strip()
            hit_evidence = str(hit.get("evidence", "")).strip()
            if keyword:
                lowered = keyword.lower()
                if hit_evidence and lowered not in keyword_evidence_map:
                    keyword_evidence_map[lowered] = hit_evidence
                evidence.append(f"{keyword}: {hit_evidence}" if hit_evidence else keyword)

        if expected_themes_covered:
            evidence.append("Covered themes: " + ", ".join(expected_themes_covered))
        if expected_themes_missing:
            evidence.append("Missing themes: " + ", ".join(expected_themes_missing))
        if not evidence:
            evidence = ["No explicit evidence captured in structured memory."]

        question_assessments.append(
            {
                "sequence": len(question_assessments) + 1,
                "question_id": question_id,
                "question_type": question_type,
                "question_text": question_text,
                "answer_summary": event.get("answer_summary", ""),
                "score": question_score,
                "experience_signal": experience_signal,
                "technical_signal": technical_signal,
                "communication_signal": communication_signal,
                "confidence": confidence,
                "expected_themes_covered": expected_themes_covered,
                "expected_themes_missing": expected_themes_missing,
                "evidence": evidence,
            }
        )

        strengths.extend(event.get("strengths", []))
        weaknesses.extend(event.get("weaknesses", []))
        red_flags.extend(event.get("red_flags", []))

    keyword_coverage = []
    for keyword in job.keywords or []:
        lowered = keyword.lower()
        evidence = keyword_evidence_map.get(lowered, "")
        keyword_coverage.append(
            {
                "keyword": keyword,
                "evidence": evidence,
                "score": 1 if evidence else 0,
            }
        )

    experience_score = round(
        _weighted_average(
            [(float(e.get("experience_signal", 5.0)), float(e.get("confidence", 0.6))) for e in ordered_events],
            5.0,
        ),
        1,
    )
    technical_score = round(
        _weighted_average(
            [(float(e.get("technical_signal", 5.0)), float(e.get("confidence", 0.6))) for e in ordered_events],
            5.0,
        ),
        1,
    )
    communication_score = round(
        _weighted_average(
            [(float(e.get("communication_signal", 5.0)), float(e.get("confidence", 0.6))) for e in ordered_events],
            5.0,
        ),
        1,
    )

    theme_scores = {}
    for theme in job.evaluation_themes or []:
        theme_lower = theme.lower()
        covered_weight = 0.0
        missing_weight = 0.0
        for event in ordered_events:
            weight = float(event.get("confidence", 0.6))
            covered_list = [t.lower() for t in event.get("expected_themes_covered", [])]
            missing_list = [t.lower() for t in event.get("expected_themes_missing", [])]
            if theme_lower in covered_list:
                covered_weight += weight
            if theme_lower in missing_list:
                missing_weight += weight

        if covered_weight == 0 and missing_weight == 0:
            theme_score = 5.0
        else:
            theme_score = _clamp_float(5.0 + covered_weight * 1.8 - missing_weight * 1.5, 0.0, 10.0, 5.0)
        theme_scores[theme] = round(theme_score, 1)

    tech_stack_present = 0
    tech_stack_total = len(job.tech_stack or [])
    tech_stack_notes = ""
    tech_stack_match = None
    if tech_stack_total > 0:
        all_hits = set()
        all_summaries = " ".join(str(e.get("answer_summary", "")).lower() for e in ordered_events)
        for event in ordered_events:
            for hit in event.get("keyword_hits", []):
                all_hits.add(str(hit.get("keyword", "")).lower())
        for tech in job.tech_stack:
            tech_lower = str(tech).lower()
            if tech_lower in all_hits or tech_lower in all_summaries:
                tech_stack_present += 1

        ratio = tech_stack_present / tech_stack_total
        if ratio > 0.75:
            tech_stack_match = "strong"
        elif ratio >= 0.5:
            tech_stack_match = "good"
        elif ratio >= 0.25:
            tech_stack_match = "partial"
        else:
            tech_stack_match = "weak"
        tech_stack_notes = (
            f"Candidate showed evidence for {tech_stack_present}/{tech_stack_total} required technologies."
        )

    w_exp = float(weights.get("experience", 0.4))
    w_tech = float(weights.get("technical", 0.4))
    w_comm = float(weights.get("communication", 0.2))
    global_score = round(
        _clamp_float(
            (experience_score * w_exp + technical_score * w_tech + communication_score * w_comm) * 10.0,
            0.0,
            100.0,
            50.0,
        ),
        1,
    )
    recommendation = _recommendation_from_global(global_score)

    strengths = _dedupe_preserve_order(strengths)[:8]
    weaknesses = _dedupe_preserve_order(weaknesses)[:8]
    red_flags = _dedupe_preserve_order(red_flags)[:8]

    if question_assessments:
        by_type: dict[str, int] = {}
        for qa in question_assessments:
            qtype = str(qa.get("question_type") or "other")
            by_type[qtype] = by_type.get(qtype, 0) + 1
        type_summary = ", ".join(f"{k}: {v}" for k, v in sorted(by_type.items()))
        summary_parts = [
            f"Structured live-memory evaluation for {candidate.first_name} {candidate.last_name} across {len(question_assessments)} assessed exchanges."
        ]
        if type_summary:
            summary_parts.append(f"Assessment breakdown by question type ({type_summary}).")
        if strengths:
            summary_parts.append("Strengths observed: " + "; ".join(strengths[:3]) + ".")
        if weaknesses:
            summary_parts.append("Areas to improve: " + "; ".join(weaknesses[:3]) + ".")
        summary = " ".join(summary_parts)
    else:
        summary = "Structured live-memory evaluation was attempted but contained no assessable question events."

    return {
        "summary": summary,
        "questions": question_assessments,
        "question_assessments": question_assessments,
        "keyword_coverage": keyword_coverage,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "experience_score": experience_score,
        "technical_score": technical_score,
        "communication_score": communication_score,
        "global_score": global_score,
        "recommendation": recommendation,
        "red_flags": red_flags,
        "theme_scores": theme_scores,
        "tech_stack_match": tech_stack_match,
        "tech_stack_present": tech_stack_present if tech_stack_total else None,
        "tech_stack_total": tech_stack_total if tech_stack_total else None,
        "tech_stack_notes": tech_stack_notes if tech_stack_total else "",
        "live_memory_event_count": len(ordered_events),
    }


# ── Apply ────────────────────────────────────────────────

@app.post("/api/apply")
async def apply(
    first_name: str = Form(...),
    last_name: str = Form(...),
    email: str = Form(...),
    job_id: int = Form(...),
    cv: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    file_bytes = await cv.read()
    cv_text = extract_pdf_text(file_bytes)

    if not cv_text.strip():
        raise HTTPException(400, "Could not extract text from PDF")

    # Get specific job config
    result = await db.execute(select(JobConfig).where(JobConfig.id == job_id, JobConfig.is_active == 1))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "Job configuration not found or inactive")

    match_pct = keyword_match(cv_text, job.keywords)
    passed = match_pct >= job.match_threshold

    token = secrets.token_urlsafe(32) if passed else None

    candidate = Candidate(
        first_name=first_name,
        last_name=last_name,
        email=email,
        cv_text=cv_text,
        cv_filename=cv.filename or "resume.pdf",
        match_percent=round(match_pct * 100, 1),
        status="invited" if passed else "rejected",
        interview_token=token,
        job_config_id=job.id,
    )
    db.add(candidate)
    await db.commit()

    if passed:
        send_interview_invite(email, first_name, token, job.title)
    else:
        screening_feedback = await build_screening_feedback(cv_text, job)
        send_rejection_email(email, first_name, job.title, screening_feedback)

    return {
        "status": "invited" if passed else "rejected",
        "match_percent": round(match_pct * 100, 1),
    }


# ── Interview Context ───────────────────────────────────

@app.get("/api/interview-context/{token}")
async def interview_context(token: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Candidate).where(Candidate.interview_token == token))
    candidate = result.scalar_one_or_none()
    if not candidate:
        raise HTTPException(404, "Invalid interview token")
    if candidate.status == "interview_completed":
        raise HTTPException(400, "Interview already completed")

    job_result = await db.execute(select(JobConfig).where(JobConfig.id == candidate.job_config_id))
    job = job_result.scalar_one_or_none()

    mq_for_client = [_extract_question_text(q) for q in job.mandatory_questions]
    return {
        "candidate_name": f"{candidate.first_name} {candidate.last_name}",
        "job_title": job.title,
        "job_description": job.description,
        "keywords": job.keywords,
        "mandatory_questions": mq_for_client,
        "cv_text": candidate.cv_text[:3000],
        "max_interview_minutes": job.max_interview_minutes,
    }


# ── Ephemeral Token ─────────────────────────────────────

@app.post("/api/live-token")
async def live_token():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(500, "GEMINI_API_KEY not configured")

    try:
        from google import genai

        client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
        live_model = os.getenv("LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")

        config = {
            "model": f"models/{live_model}",
            "config": {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": "Aoede"
                        }
                    }
                },
            },
        }

        token = client.auth_tokens.create(
            config={"live_connect_constraints": config}
        )
        return {"token": token.name}
    except Exception as e:
        print(f"Ephemeral token error: {e}")
        # Fallback: return the raw API key (not ideal for prod, fine for hackathon)
        return {"token": os.getenv("GEMINI_API_KEY"), "fallback": True}


# ── Interview Complete + Scoring ─────────────────────────

class InterviewCompleteRequest(BaseModel):
    transcript: str


async def get_interview_result_for_candidate(
    db: AsyncSession, candidate_id: int
) -> Optional[InterviewResult]:
    result = await db.execute(
        select(InterviewResult)
        .where(InterviewResult.candidate_id == candidate_id)
        .order_by(InterviewResult.created_at.desc(), InterviewResult.id.desc())
        .limit(1)
    )
    return result.scalars().first()


def apply_scoring_to_candidate(candidate: Candidate, scoring: dict) -> None:
    candidate.status = "interview_completed"
    candidate.global_score = scoring.get("global_score")
    candidate.recommendation = scoring.get("recommendation")


def apply_scoring_to_interview(
    interview_result: InterviewResult, transcript: str, scoring: dict
) -> None:
    interview_result.transcript = transcript
    interview_result.summary = scoring.get("summary")
    interview_result.summary_candidate = scoring.get("summary_candidate")
    interview_result.questions = scoring.get("questions")
    interview_result.keyword_coverage = scoring.get("keyword_coverage")
    interview_result.global_score = scoring.get("global_score")
    interview_result.recommendation = scoring.get("recommendation")
    interview_result.red_flags = scoring.get("red_flags")
    interview_result.strengths = scoring.get("strengths")
    interview_result.weaknesses = scoring.get("weaknesses")
    interview_result.experience_score = scoring.get("experience_score")
    interview_result.technical_score = scoring.get("technical_score")
    interview_result.communication_score = scoring.get("communication_score")
    interview_result.theme_scores = scoring.get("theme_scores")
    interview_result.tech_stack_match = scoring.get("tech_stack_match")
    interview_result.raw_scoring_json = scoring


async def upsert_interview_result(
    db: AsyncSession, candidate_id: int, transcript: str, scoring: dict
) -> InterviewResult:
    interview_result = await get_interview_result_for_candidate(db, candidate_id)
    if not interview_result:
        interview_result = InterviewResult(candidate_id=candidate_id, transcript=transcript)
        db.add(interview_result)
    apply_scoring_to_interview(interview_result, transcript, scoring)
    return interview_result


@app.post("/api/interview-complete/{token}")
async def interview_complete(
    token: str,
    body: InterviewCompleteRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Candidate).where(Candidate.interview_token == token))
    candidate = result.scalar_one_or_none()
    if not candidate:
        raise HTTPException(404, "Invalid interview token")

    existing_interview = await get_interview_result_for_candidate(db, candidate.id)
    if candidate.status == "interview_completed" and existing_interview:
        return {
            "status": "already_scored",
            "global_score": candidate.global_score
            if candidate.global_score is not None
            else existing_interview.global_score,
        }

    job_result = await db.execute(select(JobConfig).where(JobConfig.id == candidate.job_config_id))
    job = job_result.scalar_one_or_none()
    if not job:
        raise HTTPException(500, "No job configuration found")

    # Score with Gemini text model
    scoring = await score_interview(body.transcript, job, candidate)
    scoring["scoring_source"] = scoring.get("scoring_source") or "manual_transcript"
    await upsert_interview_result(db, candidate.id, body.transcript, scoring)
    apply_scoring_to_candidate(candidate, scoring)
    await db.commit()

    return {"status": "scored", "global_score": scoring.get("global_score")}


async def score_interview(transcript: str, job: JobConfig, candidate: Candidate) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    scoring_model = os.getenv("SCORING_MODEL", "gemini-2.5-flash-lite")

    mq_texts = [_extract_question_text(q) for q in job.mandatory_questions]
    weights = job.scoring_weights or {"experience": 0.4, "technical": 0.4, "communication": 0.2}
    themes = job.evaluation_themes or []
    tech_stack = job.tech_stack or []

    themes_instruction = ""
    if themes:
        themes_instruction = f"""
Also score the candidate on each of these themes (0-10 each):
Themes: {json.dumps(themes)}
Add a "theme_scores" field: {{"theme_name": score, ...}}
"""

    tech_stack_instruction = ""
    if tech_stack:
        tech_stack_instruction = f"""
Tech stack comparison -the job requires: {json.dumps(tech_stack)}.
Count how many the candidate demonstrated knowledge of. Add these fields:
- "tech_stack_present": <count of demonstrated technologies>,
- "tech_stack_total": {len(tech_stack)},
- "tech_stack_match": "strong" (>75%) | "good" (50-75%) | "partial" (25-50%) | "weak" (<25%),
- "tech_stack_notes": "<one sentence summary>"
"""

    prompt = f"""You are an expert HR interview evaluator. Analyze this interview transcript and produce a structured evaluation.

Job Title: {job.title}
Job Description: {job.description}
Required Keywords: {json.dumps(job.keywords)}
Mandatory Questions: {json.dumps(mq_texts)}
Scoring Weights: {json.dumps(weights)}

Candidate: {candidate.first_name} {candidate.last_name}

TRANSCRIPT:
{transcript}

Produce a JSON response with EXACTLY this structure:
{{
  "summary": "2-3 sentence overall summary of the candidate's performance",
  "summary_candidate": "2-3 polite and concise sentences addressed to the candidate, with constructive feedback and no internal hiring rationale",
  "questions": [
    {{
      "question_id": "q1",
      "question_text": "the question that was asked",
      "answer_summary": "brief summary of the candidate's answer",
      "score": 0-5,
      "evidence": ["direct quotes or paraphrases supporting the score"]
    }}
  ],
  "keyword_coverage": [
    {{
      "keyword": "the keyword",
      "evidence": "how the candidate demonstrated this skill or empty string",
      "score": 0 or 1
    }}
  ],
  "strengths": ["bullet point strength", "..."],
  "weaknesses": ["bullet point weakness", "..."],
  "experience_score": 0-10,
  "technical_score": 0-10,
  "communication_score": 0-10,
  "global_score": 0-100,
  "recommendation": "strong_yes|yes|maybe|no",
  "red_flags": ["any concerns noted during the interview"]
}}
{themes_instruction}{tech_stack_instruction}
Compute global_score as a weighted average: experience_score * {weights.get("experience", 0.4)} + technical_score * {weights.get("technical", 0.4)} + communication_score * {weights.get("communication", 0.2)}, scaled to 0-100.

Score strictly. Use 5-6 for average performance, 7+ only with strong evidence.

IMPORTANT: Return ONLY valid JSON, no markdown, no extra text."""

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=scoring_model,
            contents=prompt,
        )
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            elif "```" in text:
                text = text[:text.rfind("```")]
        text = text.strip()
        # Try to extract the first complete JSON object if extra text surrounds it
        try:
            scoring = json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                scoring = json.loads(match.group())
            else:
                raise

        # Override recommendation based on global_score thresholds
        gs = scoring.get("global_score", 50)
        if gs < 55:
            scoring["recommendation"] = "no"
        elif gs < 65:
            scoring["recommendation"] = "maybe"
        elif gs < 80:
            scoring["recommendation"] = "yes"
        else:
            scoring["recommendation"] = "strong_yes"

        return scoring
    except Exception as e:
        print(f"Scoring error: {e}")
        return {
            "summary": "Scoring failed due to an error.",
            "summary_candidate": "Thank you for your interview. We appreciate your time and will get back to you soon.",
            "questions": [],
            "keyword_coverage": [],
            "strengths": [],
            "weaknesses": [],
            "experience_score": 5,
            "technical_score": 5,
            "communication_score": 5,
            "global_score": 50,
            "recommendation": "maybe",
            "red_flags": [f"Automated scoring failed: {str(e)}"],
        }


# ── Dashboard Endpoints ─────────────────────────────────

@app.get("/api/candidates")
async def list_candidates(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Candidate, JobConfig.title)
        .outerjoin(JobConfig, Candidate.job_config_id == JobConfig.id)
        .where(Candidate.status.in_(["interview_completed", "accepted", "rejected_after_interview"]))
        .order_by(Candidate.created_at.desc())
    )
    candidates_with_jobs = result.all()
    return [
        {
            "id": c.Candidate.id,
            "first_name": c.Candidate.first_name,
            "last_name": c.Candidate.last_name,
            "email": c.Candidate.email,
            "status": c.Candidate.status,
            "global_score": c.Candidate.global_score,
            "recommendation": c.Candidate.recommendation,
            "match_percent": c.Candidate.match_percent,
            "job_title": c.title,
            "created_at": str(c.Candidate.created_at) if c.Candidate.created_at else None,
        }
        for c in candidates_with_jobs
    ]


@app.get("/api/candidate/{candidate_id}")
async def get_candidate(candidate_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
    c = result.scalar_one_or_none()
    if not c:
        raise HTTPException(404, "Candidate not found")

    interview = await get_interview_result_for_candidate(db, candidate_id)

    return {
        "id": c.id,
        "first_name": c.first_name,
        "last_name": c.last_name,
        "email": c.email,
        "status": c.status,
        "global_score": c.global_score,
        "recommendation": c.recommendation,
        "match_percent": c.match_percent,
        "cv_text": c.cv_text[:2000],
        "created_at": str(c.created_at) if c.created_at else None,
        "interview": {
            "transcript": interview.transcript,
            "summary": interview.summary,
            "summary_candidate": interview.summary_candidate,
            "questions": interview.questions,
            "question_assessments": (
                interview.raw_scoring_json.get("question_assessments") or interview.questions
                if isinstance(interview.raw_scoring_json, dict)
                else interview.questions
            ),
            "keyword_coverage": interview.keyword_coverage,
            "global_score": interview.global_score,
            "recommendation": interview.recommendation,
            "red_flags": interview.red_flags,
            "strengths": interview.strengths,
            "weaknesses": interview.weaknesses,
            "experience_score": interview.experience_score,
            "technical_score": interview.technical_score,
            "communication_score": interview.communication_score,
            "theme_scores": interview.theme_scores,
            "tech_stack_match": interview.tech_stack_match,
            "scoring_source": (
                interview.raw_scoring_json.get("scoring_source")
                if isinstance(interview.raw_scoring_json, dict)
                else None
            ),
            "fallback_reason": (
                interview.raw_scoring_json.get("fallback_reason")
                if isinstance(interview.raw_scoring_json, dict)
                else None
            ),
            "live_memory_event_count": (
                len(interview.raw_scoring_json.get("live_memory_events", []))
                if isinstance(interview.raw_scoring_json, dict)
                and isinstance(interview.raw_scoring_json.get("live_memory_events"), list)
                else None
            ),
        } if interview else None,
    }


class DecisionRequest(BaseModel):
    decision: str  # "accept" or "reject"
    summary_candidate: Optional[str] = None


@app.post("/api/candidate/{candidate_id}/decision")
async def candidate_decision(
    candidate_id: int,
    body: DecisionRequest,
    db: AsyncSession = Depends(get_db),
):
    if body.decision not in ("accept", "reject"):
        raise HTTPException(400, "Decision must be 'accept' or 'reject'")

    result = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
    c = result.scalar_one_or_none()
    if not c:
        raise HTTPException(404, "Candidate not found")

    job_result = await db.execute(select(JobConfig).where(JobConfig.id == c.job_config_id))
    job = job_result.scalar_one_or_none()
    ir_result = await db.execute(
        select(InterviewResult).where(InterviewResult.candidate_id == candidate_id)
    )
    interview = ir_result.scalar_one_or_none()

    new_status = "accepted" if body.decision == "accept" else "rejected_after_interview"
    c.status = new_status
    await db.commit()

    summary_for_email = (
        body.summary_candidate
        or (interview.summary_candidate if interview else None)
        or (interview.summary if interview else None)
    )
    send_decision_email(
        c.email,
        c.first_name,
        job.title if job else "the position",
        body.decision,
        summary_for_email,
    )

    return {"status": new_status}


# ── Job Config (for dashboard) ───────────────────────────

@app.get("/api/job-config")
async def get_job_config(db: AsyncSession = Depends(get_db)):
    # Keep this for backward compatibility or simple use-cases, but return all active
    result = await db.execute(select(JobConfig).where(JobConfig.is_active == 1))
    jobs = result.scalars().all()
    return jobs

@app.get("/api/job-configs/{config_id}")
async def get_single_job_config(config_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(JobConfig).where(JobConfig.id == config_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "Config not found")
    return {
        "id": job.id,
        "title": job.title,
        "description": job.description,
        "keywords": job.keywords,
        "mandatory_questions": job.mandatory_questions,
        "scoring_weights": job.scoring_weights,
        "evaluation_themes": job.evaluation_themes,
        "tech_stack": job.tech_stack,
        "is_active": job.is_active,
    }


# ── Job Config CRUD ──────────────────────────────────────

class QuestionItem(BaseModel):
    question: str
    expected_themes: list[str] = []


class JobConfigCreate(BaseModel):
    title: str
    description: str
    keywords: list[str]
    mandatory_questions: list[QuestionItem]
    match_threshold: float = 0.3
    max_interview_minutes: int = 8
    scoring_weights: dict[str, float] = {"experience": 0.4, "technical": 0.4, "communication": 0.2}
    evaluation_themes: list[str] = []
    tech_stack: list[str] = []


class JobConfigUpdate(JobConfigCreate):
    pass


@app.get("/api/job-configs")
async def list_job_configs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(JobConfig).order_by(JobConfig.created_at.desc()))
    configs = result.scalars().all()
    out = []
    for j in configs:
        count_result = await db.execute(
            select(func.count()).select_from(Candidate).where(Candidate.job_config_id == j.id)
        )
        count = count_result.scalar()
        out.append({
            "id": j.id, "title": j.title, "description": j.description,
            "keywords": j.keywords, "mandatory_questions": j.mandatory_questions,
            "match_threshold": j.match_threshold,
            "max_interview_minutes": j.max_interview_minutes,
            "scoring_weights": j.scoring_weights,
            "evaluation_themes": j.evaluation_themes,
            "tech_stack": j.tech_stack,
            "is_active": bool(j.is_active), "candidate_count": count,
            "created_at": str(j.created_at) if j.created_at else None,
        })
    return out


@app.post("/api/job-configs")
async def create_job_config(body: JobConfigCreate, db: AsyncSession = Depends(get_db)):
    job = JobConfig(
        title=body.title,
        description=body.description,
        keywords=body.keywords,
        mandatory_questions=[q.model_dump() for q in body.mandatory_questions],
        match_threshold=body.match_threshold,
        max_interview_minutes=body.max_interview_minutes,
        scoring_weights=body.scoring_weights,
        evaluation_themes=body.evaluation_themes,
        tech_stack=body.tech_stack,
        is_active=0,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return {"id": job.id, "title": job.title}


@app.put("/api/job-configs/{config_id}")
async def update_job_config(config_id: int, body: JobConfigUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(JobConfig).where(JobConfig.id == config_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "Config not found")
    job.title = body.title
    job.description = body.description
    job.keywords = body.keywords
    job.mandatory_questions = [q.model_dump() for q in body.mandatory_questions]
    job.match_threshold = body.match_threshold
    job.max_interview_minutes = body.max_interview_minutes
    job.scoring_weights = body.scoring_weights
    job.evaluation_themes = body.evaluation_themes
    job.tech_stack = body.tech_stack
    await db.commit()
    return {"ok": True}


@app.delete("/api/job-configs/{config_id}")
async def delete_job_config(config_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(JobConfig).where(JobConfig.id == config_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "Config not found")
    if job.is_active:
        raise HTTPException(400, "Cannot delete the active config")
    count_result = await db.execute(
        select(func.count()).select_from(Candidate).where(Candidate.job_config_id == config_id)
    )
    if count_result.scalar() > 0:
        raise HTTPException(400, "Cannot delete config with linked candidates")
    await db.delete(job)
    await db.commit()
    return {"ok": True}


@app.post("/api/job-configs/{config_id}/activate")
async def activate_job_config(config_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(JobConfig).where(JobConfig.id == config_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "Config not found")
    
    # Toggle active state instead of deactivating others
    job.is_active = 1 if job.is_active == 0 else 0
    await db.commit()
    return {"ok": True, "is_active": job.is_active}


# ── CV Summary for Interview ──────────────────────────────

async def summarize_cv_for_interview(cv_text: str, job_title: str) -> tuple[str, str]:
    """Summarize a candidate's CV and generate one interview question from it.
    Returns (cv_summary, cv_question).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    scoring_model = os.getenv("SCORING_MODEL", "gemini-2.5-flash-lite")

    prompt = f"""You are an HR assistant preparing for an interview for the position of {job_title}.

Given this candidate's CV text, produce a JSON response with:
1. "cv_summary": A concise structured summary (max 200 words) covering: key experience, education, notable projects/skills.
2. "cv_question": One specific, open-ended interview question based on something interesting or relevant in the CV. The question should probe deeper into a real experience or project mentioned.

CV TEXT:
{cv_text[:3000]}

Return ONLY valid JSON, no markdown."""

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=scoring_model,
            contents=prompt,
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            elif "```" in text:
                text = text[:text.rfind("```")]
        data = json.loads(text.strip())
        return data.get("cv_summary", ""), data.get("cv_question", "")
    except Exception as e:
        print(f"[CV] Summary generation error: {e}")
        return cv_text[:500], ""


# ── WebSocket Interview Relay ──────────────────────────────

def _extract_question_text(q) -> str:
    """Extract question text from either string or dict format."""
    return q["question"] if isinstance(q, dict) else q


def build_system_instruction(ctx: dict) -> str:
    # Build the questions block with expected themes inline
    questions_block = ""
    for i, q in enumerate(ctx["mandatory_questions"]):
        q_text = _extract_question_text(q)
        expected = ""
        if isinstance(q, dict) and q.get("expected_themes"):
            themes = q["expected_themes"]
            expected = f"\n   Expected themes to probe for: {', '.join(themes)}"
        questions_block += f"{i+1}. {q_text}{expected}\n"

    keywords = ", ".join(ctx["keywords"])

    cv_summary = ctx.get("cv_summary") or ctx.get("cv_text", "")

    cv_question_block = ""
    if ctx.get("cv_question"):
        cv_question_block = f"""PHASE 2 -CV QUESTION:
- Ask this question based on the candidate's CV: "{ctx["cv_question"]}"
- No follow-up needed on this question. After the candidate answers, give a brief comment and move to Phase 3."""
    else:
        cv_question_block = "PHASE 2 -CV QUESTION:\n- Skip this phase (no CV question). Go directly to Phase 3."

    return f"""You are an experienced HR recruiter named Sarah conducting a structured phone screen for the position of {ctx["job_title"]}.

JOB DESCRIPTION:
{ctx["job_description"]}

CANDIDATE CONTEXT:
Name: {ctx["candidate_name"]}
CV Summary: {cv_summary}

============================================
INTERVIEW FLOW -FOLLOW THESE PHASES IN EXACT ORDER
============================================

PHASE 1 -ADMIN INTRO:
- Greet the candidate warmly by name. Introduce yourself as Sarah from HappyHR.
- Then ask: "Before we begin, a quick administrative check -are you legally authorized to work in France?"
- If the candidate says NO (any form: "no", "not yet", "no visa", "not authorized"):
  Say: "Unfortunately this position requires work authorization in France, so we won't be able to move forward at this time. Thank you for your interest, and we'll keep your profile on file." Then end the conversation.
- If the candidate says YES: proceed to Phase 2.

{cv_question_block}

PHASE 3 -PREDEFINED QUESTIONS:
Ask these questions ONE AT A TIME, in order:
{questions_block}
FOR EACH PREDEFINED QUESTION, follow this sub-protocol:
a) Ask the question clearly.
b) After the candidate answers, internally assess: did the answer cover the "expected themes" listed for that question?
c) FIRST FOLLOW-UP (always ask one): Give a brief human comment on their answer ("Interesting.", "That's helpful.", "Good to know."), then ask about ONE specific expected theme they did NOT cover. Pick the most important uncovered theme.
d) After the first follow-up answer, reassess: are the expected themes now adequately covered?
   - If YES (or no expected themes listed): acknowledge briefly and move to the next question.
   - If NO: ask a SECOND follow-up about ONE remaining uncovered theme. This is the maximum -after the second follow-up, move on regardless.
e) NEVER ask more than 2 follow-ups per question. After at most 2 follow-ups, move to the next question.
f) SKIP a question entirely if the candidate already covered it thoroughly during the CV question or a previous answer. Simply move to the next one.

CLARIFICATION HANDLING:
- If at any point the candidate says "I didn't understand", "what do you mean?", "could you rephrase?", "can you repeat that?", or similar:
  Do NOT move to the next question. Instead, rephrase the same question in simpler, more concrete terms. Then wait for their answer.

PHASE 4 -CANDIDATE QUESTIONS:
- After all predefined questions are done, say: "That covers my questions. Do you have any questions about the role or the company?"
- If the candidate asks a question: give a brief, professional answer. Then ask "Do you have any other questions?"
- Continue until the candidate says they have no more questions ("no", "that's all", "I'm good", etc.).

PHASE 5 -FAREWELL:
- Thank the candidate warmly for their time.
- Let them know they will hear back soon.
- End the conversation gracefully.
- After your farewell message, you MUST call the `end_interview` tool with reason="completed".

============================================
INTERVIEW TERMINATION (MANDATORY)
============================================
- After you finish your farewell in Phase 5, call `end_interview(reason="completed")`. This ends the session.
- If the candidate says they want to stop at ANY point (e.g. "I'm done", "I want to stop", "end the interview", "I'd like to leave", "that's enough"), politely acknowledge, thank them for their time, then call `end_interview(reason="candidate_requested")`.
- If the candidate is not authorized to work (Phase 1 rejection), after delivering the rejection message, call `end_interview(reason="not_authorized")`.
- NEVER continue the interview after calling `end_interview`.

============================================
LIVE MEMORY TOOLING (MANDATORY)
============================================
- You MUST call the function `record_question_assessment` exactly once after each finalized substantive candidate answer across the interview phases.
- Keep canonical IDs for primary questions:
  - CV question: `question_id` = "cv_q", `question_type` = "cv".
  - Mandatory questions: `question_id` = "q1", "q2", etc., `question_type` = "mandatory".
- For additional assessed exchanges (follow-ups, candidate questions, spontaneous technical probes), still call the function with unique IDs such as:
  - follow-up: `question_id` = "fup_<n>", `question_type` = "followup"
  - candidate question: `question_id` = "candq_<n>", `question_type` = "candidate_question"
  - other: `question_id` = "extra_<n>", `question_type` = "other"
- Call the function only once after each finalized answer segment.
- Keep `answer_summary` non-verbatim and factual. Never invent details.
- Fill all fields with strict ranges:
  - `question_score`: 0-5
  - `experience_signal`, `technical_signal`, `communication_signal`: 0-10
  - `confidence`: 0-1
- `keyword_hits` must only include job-relevant evidence observed during the interview.
- `expected_themes_covered` and `expected_themes_missing` must reflect the question's expected themes.
- If evidence is weak, lower scores and include concerns in `weaknesses` or `red_flags`.

============================================
CONVERSATION STYLE RULES
============================================
- Ask ONE question at a time. Never combine multiple questions.
- After asking a question, you MUST stop speaking and wait for the candidate's answer. Do NOT ask a second question immediately.
- Keep responses concise and natural -this is a phone conversation, not an essay.
- When transitioning between questions, always give a brief human-like comment on their previous answer FIRST ("Interesting.", "That's helpful.", "Good to hear.", "We value that kind of experience.") before asking the next question.
- Probe for evidence of these keywords through natural conversation: {keywords}
- Target total interview duration: approximately {ctx["max_interview_minutes"]} minutes. If running long, reduce follow-ups but always complete all predefined questions.

============================================
SAFETY / FAIRNESS RULES
============================================
- Do NOT infer or ask about protected attributes (race, religion, health, disability, age, gender, sexual orientation, marital status, family, pregnancy, etc.).
- Evaluate ONLY job-relevant skills and experience.
- Be equally warm and professional with all candidates.
- Do NOT mention AI, scoring, algorithms, or that you are not human.
- Do NOT invent or assume details from the candidate's CV that are not explicitly stated."""


def build_openai_ws_url() -> str:
    model = os.getenv("LIVE_MODEL", "gpt-realtime")
    return f"wss://api.openai.com/v1/realtime?model={model}"


def pcm16_mono_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw PCM16 mono bytes into a WAV container."""
    channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    data_size = len(pcm_bytes)

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1, channels,
        sample_rate, byte_rate, block_align, bits_per_sample,
        b'data', data_size
    )
    return header + pcm_bytes


async def transcribe_audio_with_stt_v2(audio_chunks: list[str]) -> str:
    """Transcribe candidate audio chunks using Google Cloud Speech-to-Text V2."""
    stt_model = os.getenv("STT_MODEL", "latest_long")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

    if not project_id:
        print("[STT] ERROR: GOOGLE_CLOUD_PROJECT is not set.")
        return ""

    if not audio_chunks:
        return ""

    # Concatenate base64 chunks into raw PCM buffer
    pcm_parts = [base64.b64decode(chunk) for chunk in audio_chunks]
    pcm_buffer = b"".join(pcm_parts)
    
    duration_ms = len(pcm_buffer) // (16000 * 2 // 1000)
    print(f"[STT] PCM buffer: {len(pcm_buffer)} bytes (~{duration_ms}ms of audio)")

    if len(pcm_buffer) < 6000:
        print(f"[STT] Buffer too short ({len(pcm_buffer)} bytes) -skipping")
        return ""

    try:
        from google.cloud.speech_v2 import SpeechClient
        from google.cloud.speech_v2.types import cloud_speech

        def do_transcribe() -> str:
            client = SpeechClient()
            request = cloud_speech.RecognizeRequest(
                recognizer=f"projects/{project_id}/locations/global/recognizers/_",
                config=cloud_speech.RecognitionConfig(
                    explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                        encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                        sample_rate_hertz=24000,
                        audio_channel_count=1,
                    ),
                    language_codes=["en-US"],
                    model=stt_model,
                ),
                content=pcm_buffer,
            )
            response = client.recognize(request=request)
            
            # Combine all results
            results = []
            for result in response.results:
                if result.alternatives:
                    results.append(result.alternatives[0].transcript)
                    
            text = " ".join(results).strip()
            if text:
                print(f"[STT] Speech-to-Text result: '{text}'")
            return text

        return await asyncio.to_thread(do_transcribe)

    except Exception as e:
        import traceback
        print(f"[STT] ERROR during Google Cloud STT transcription: {e}")
        traceback.print_exc()
        return ""


async def build_transcript_from_stt_segments(
    all_candidate_audio: list[dict], transcript_parts: list[dict]
) -> str:
    parts = [dict(p) for p in transcript_parts]
    print(
        f"[POST] Audio segments to transcribe: {len(all_candidate_audio)}, "
        f"AI transcript parts: {len([p for p in parts if p['role'] == 'ai'])}"
    )

    async def _transcribe_one(i: int, audio_entry: dict) -> tuple[int, int, str]:
        try:
            print(
                f"[POST] Transcribing audio segment {i + 1}/{len(all_candidate_audio)} "
                f"(index={audio_entry['index']}, chunks={len(audio_entry['chunks'])})"
            )
            stt_text = await transcribe_audio_with_stt_v2(audio_entry["chunks"])
            print(
                f"[POST] STT result for segment {i + 1}: '{stt_text[:80]}...'"
                if len(stt_text) > 80
                else f"[POST] STT result for segment {i + 1}: '{stt_text}'"
            )
            return (i, audio_entry["index"], stt_text)
        except Exception as e:
            import traceback
            print(f"[POST] STT error for segment {i + 1}: {e}")
            traceback.print_exc()
            return (i, audio_entry["index"], "")

    results = await asyncio.gather(
        *[_transcribe_one(i, entry) for i, entry in enumerate(all_candidate_audio)]
    )

    for _, idx, stt_text in results:
        if stt_text:
            for part in parts:
                if part.get("index") == idx and part["role"] == "user":
                    part["text"] = stt_text
                    break

    parts.sort(key=lambda p: p.get("index", 0))
    filled_parts = [p for p in parts if p.get("text")]
    print(
        f"[POST] Transcript parts after STT: {len(filled_parts)} non-empty "
        f"(of {len(parts)} total)"
    )

    transcript_text = "\n\n".join(
        f"{'Interviewer' if p['role'] == 'ai' else 'Candidate'}: {p['text']}"
        for p in filled_parts
    )
    print(f"[POST] Final transcript length: {len(transcript_text)} chars")
    return transcript_text


@app.websocket("/ws/{token}")
async def websocket_interview(ws: WebSocket, token: str):
    """WebSocket relay: Browser ↔ Backend ↔ Gemini Live API."""
    # Get interview context from DB
    async for db in get_db():
        result = await db.execute(select(Candidate).where(Candidate.interview_token == token))
        candidate = result.scalar_one_or_none()
        if not candidate or candidate.status == "interview_completed":
            await ws.close(code=4004, reason="Invalid or expired token")
            return

        job_result = await db.execute(select(JobConfig).where(JobConfig.id == candidate.job_config_id))
        job = job_result.scalar_one_or_none()
        break

    candidate_id = candidate.id

    # Generate CV summary and CV-based question before building the system prompt
    cv_summary, cv_question = await summarize_cv_for_interview(
        candidate.cv_text, job.title
    )
    print(f"[WS] CV summary generated ({len(cv_summary)} chars), CV question: {'yes' if cv_question else 'no'}")

    ctx = {
        "candidate_name": f"{candidate.first_name} {candidate.last_name}",
        "job_title": job.title,
        "job_description": job.description,
        "keywords": job.keywords,
        "mandatory_questions": job.mandatory_questions,
        "cv_text": candidate.cv_text[:3000],
        "cv_summary": cv_summary,
        "cv_question": cv_question,
        "max_interview_minutes": job.max_interview_minutes,
        "scoring_weights": job.scoring_weights or {"experience": 0.4, "technical": 0.4, "communication": 0.2},
        "evaluation_themes": job.evaluation_themes or [],
        "tech_stack": job.tech_stack or [],
    }

    await ws.accept()

    openai_url = build_openai_ws_url()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        await ws.send_json({"type": "error", "error": "OPENAI_API_KEY not configured"})
        await ws.close()
        return

    # Track transcript for scoring
    model_text_buffer = ""
    transcript_parts = []  # list of {"role": "ai"|"user", "text": str, "index": int}
    candidate_audio_chunks = []  # current speech segment
    all_candidate_audio = []  # list of {"index": int, "chunks": list[str]}
    live_memory_events = []  # list of normalized record_question_assessment payloads
    turn_index = 0
    # Accumulate streamed function-call arguments per call_id
    pending_tool_calls = {}  # {call_id: {"name": str, "arguments": str}}

    try:
        async with websockets.connect(
            openai_url,
            additional_headers={
                "Authorization": f"Bearer {openai_api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
        ) as oai_ws:
            # Wait for session.created
            while True:
                init_raw = await oai_ws.recv()
                init_msg = json.loads(init_raw)
                print(f"[WS] OpenAI init: type={init_msg.get('type')}")
                if init_msg.get("type") == "session.created":
                    break

            # Send session.update with instructions, tools, VAD, voice
            await oai_ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "instructions": build_system_instruction(ctx),
                    "modalities": ["text", "audio"],
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "tools": _build_live_memory_tools(),
                    "tool_choice": "auto",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 1200,
                    },
                },
            }))

            # Wait for session.updated
            while True:
                upd_raw = await oai_ws.recv()
                upd_msg = json.loads(upd_raw)
                print(f"[WS] OpenAI update: type={upd_msg.get('type')}")
                if upd_msg.get("type") == "session.updated":
                    break

            await ws.send_json({"type": "status", "message": "Connected to OpenAI Realtime API"})
            print("[WS] Session configured, sending initial trigger")

            # Send initial user message to start the interview
            await oai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello, I'm ready to start the interview."}],
                },
            }))
            await oai_ws.send(json.dumps({"type": "response.create"}))

            async def openai_to_client():
                """Forward OpenAI Realtime events to the browser client."""
                nonlocal model_text_buffer, turn_index, live_memory_events, pending_tool_calls
                try:
                    async for raw in oai_ws:
                        msg = json.loads(raw)
                        evt = msg.get("type", "")

                        # ── Audio delta ──
                        if evt == "response.audio.delta":
                            delta = msg.get("delta", "")
                            if delta:
                                await ws.send_json({
                                    "type": "audio",
                                    "mimeType": "audio/pcm;rate=24000",
                                    "data": delta,
                                })

                        # ── Text transcript delta ──
                        elif evt == "response.audio_transcript.delta":
                            delta = msg.get("delta", "")
                            if delta:
                                model_text_buffer += delta
                                await ws.send_json({"type": "text", "text": delta})

                        # ── Response done (turn complete) ──
                        elif evt == "response.done":
                            if model_text_buffer.strip():
                                transcript_parts.append({"role": "ai", "text": model_text_buffer.strip(), "index": turn_index})
                                turn_index += 1
                                print(f"[AI]: {model_text_buffer.strip()[:120]}...")
                                model_text_buffer = ""
                            await ws.send_json({"type": "turnComplete"})

                        # ── Function call arguments streaming ──
                        elif evt == "response.function_call_arguments.delta":
                            call_id = msg.get("call_id", "")
                            if call_id not in pending_tool_calls:
                                pending_tool_calls[call_id] = {"name": msg.get("name", ""), "arguments": ""}
                            pending_tool_calls[call_id]["arguments"] += msg.get("delta", "")

                        # ── Function call complete ──
                        elif evt == "response.function_call_arguments.done":
                            call_id = msg.get("call_id", "")
                            call_name = msg.get("name", "") or pending_tool_calls.get(call_id, {}).get("name", "")
                            raw_args = msg.get("arguments", "") or pending_tool_calls.get(call_id, {}).get("arguments", "")
                            pending_tool_calls.pop(call_id, None)

                            try:
                                args = json.loads(raw_args) if raw_args else {}
                            except json.JSONDecodeError:
                                args = {}

                            output = '{"status": "ignored"}'

                            if call_name == "record_question_assessment":
                                try:
                                    normalized_event = _normalize_live_memory_event(
                                        args, call_id, len(live_memory_events) + 1
                                    )
                                    live_memory_events.append(normalized_event)
                                    output = json.dumps({"status": "ok", "recorded": True, "question_id": normalized_event.get("question_id")})
                                    print(
                                        f"[WS] Recorded live memory event for question_id="
                                        f"{normalized_event.get('question_id')} (total={len(live_memory_events)})"
                                    )
                                except Exception as e:
                                    output = json.dumps({"status": "error", "error": str(e)})

                            elif call_name == "end_interview":
                                reason = args.get("reason", "completed") if isinstance(args, dict) else "completed"
                                print(f"[WS] end_interview tool called, reason={reason}")
                                output = json.dumps({"status": "ok", "reason": reason})
                                try:
                                    await ws.send_json({"type": "interviewComplete", "reason": reason})
                                except Exception:
                                    pass

                            # Send function output back + trigger continuation
                            await oai_ws.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": output,
                                },
                            }))
                            await oai_ws.send(json.dumps({"type": "response.create"}))

                        # ── Server VAD speech events (for STT chunking) ──
                        elif evt == "input_audio_buffer.speech_started":
                            print("[WS] Server VAD: user speech started")
                            candidate_audio_chunks = []

                        elif evt == "input_audio_buffer.speech_stopped":
                            print("[WS] Server VAD: user speech stopped")
                            if candidate_audio_chunks:
                                idx = turn_index
                                turn_index += 1
                                all_candidate_audio.append({"index": idx, "chunks": list(candidate_audio_chunks)})
                                transcript_parts.append({"role": "user", "text": "", "index": idx})
                            candidate_audio_chunks = []

                        # ── User transcript from server ──
                        elif evt == "conversation.item.input_audio_transcription.completed":
                            user_text = msg.get("transcript", "").strip()
                            if user_text:
                                print(f"[User]: {user_text[:120]}")
                                # Update the last user placeholder with actual text
                                for i in range(len(transcript_parts) - 1, -1, -1):
                                    if transcript_parts[i]["role"] == "user" and not transcript_parts[i]["text"]:
                                        transcript_parts[i]["text"] = user_text
                                        break

                        # ── Errors ──
                        elif evt == "error":
                            err_detail = msg.get("error", {})
                            err_msg = err_detail.get("message", str(err_detail)) if isinstance(err_detail, dict) else str(err_detail)
                            print(f"[WS] OpenAI error: {err_msg}")
                            await ws.send_json({"type": "error", "error": err_msg})

                except websockets.ConnectionClosed as e:
                    print(f"[WS] OpenAI connection closed: {e}")
                    try:
                        await ws.send_json({"type": "sessionDropped", "reason": str(e)})
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[WS] openai_to_client error: {e}")

            async def client_to_openai():
                """Forward browser client audio to OpenAI Realtime API."""
                nonlocal candidate_audio_chunks, turn_index
                try:
                    while True:
                        data = await ws.receive_text()
                        msg = json.loads(data)

                        if msg.get("type") == "audio" and msg.get("data"):
                            # Store audio for STT fallback
                            candidate_audio_chunks.append(msg["data"])
                            # Forward to OpenAI
                            await oai_ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": msg["data"],
                            }))

                        elif msg.get("type") == "text" and msg.get("text"):
                            transcript_parts.append({"role": "user", "text": msg["text"], "index": turn_index})
                            print(f"[User text]: {msg['text']}")
                            turn_index += 1
                            await oai_ws.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [{"type": "input_text", "text": msg["text"]}],
                                },
                            }))
                            await oai_ws.send(json.dumps({"type": "response.create"}))

                except WebSocketDisconnect:
                    print("[WS] Client disconnected")
                except Exception as e:
                    print(f"[WS] client_to_openai error: {e}")
                finally:
                    print("[WS] Closing OpenAI connection to unblock relay task")
                    try:
                        await oai_ws.close()
                    except Exception:
                        pass

            # Run both relay tasks concurrently
            await asyncio.gather(
                openai_to_client(),
                client_to_openai(),
                return_exceptions=True,
            )

    except Exception as e:
        import traceback
        print(f"[WS] WebSocket interview error: {e}")
        traceback.print_exc()
        try:
            await ws.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass

    # Flush any AI text that was mid-turn when the client disconnected
    if model_text_buffer.strip():
        transcript_parts.append({"role": "ai", "text": model_text_buffer.strip(), "index": turn_index})
        model_text_buffer = ""

    # Post-interview scoring: memory-first, STT fallback
    print(f"[POST] Starting post-interview processing for candidate_id={candidate_id}")
    print(
        f"[POST] Live memory events captured: {len(live_memory_events)}, "
        f"audio segments captured: {len(all_candidate_audio)}"
    )

    enable_live_memory_scoring = _env_flag("ENABLE_LIVE_MEMORY_SCORING", True)
    enable_stt_fallback = True  # Always attempt STT fallback for scoring
    try:
        min_live_memory_events = int(os.getenv("LIVE_MEMORY_MIN_EVENTS", "3"))
    except ValueError:
        min_live_memory_events = 3

    try:
        async for db in get_db():
            candidate_result = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
            candidate_db = candidate_result.scalar_one_or_none()
            if not candidate_db:
                print(f"[POST] ERROR: candidate_id={candidate_id} not found in DB")
                break

            existing_interview = await get_interview_result_for_candidate(db, candidate_id)
            if candidate_db.status == "interview_completed" and existing_interview:
                print("[POST] Already scored -skipping")
                break

            job_result = await db.execute(select(JobConfig).where(JobConfig.id == candidate_db.job_config_id))
            job_db = job_result.scalar_one_or_none()
            if not job_db:
                print(f"[POST] ERROR: job_config not found for candidate_id={candidate_id}")
                break

            scoring = None
            transcript_text = ""
            scoring_source = ""
            fallback_reason = None

            if enable_live_memory_scoring:
                is_valid, live_reason = validate_live_memory_events(
                    live_memory_events, job_db.mandatory_questions or [], min_live_memory_events
                )
                if is_valid:
                    print("[POST] Using live memory scoring as primary path")
                    scoring = score_interview_from_live_memory(live_memory_events, job_db, candidate_db)
                    transcript_text = build_structured_transcript_from_live_memory(
                        transcript_parts, live_memory_events
                    )
                    scoring_source = "live_memory"
                else:
                    fallback_reason = live_reason
                    print(f"[POST] Live memory scoring unavailable: {live_reason}")
            else:
                fallback_reason = "live_memory_scoring_disabled"
                print("[POST] Live memory scoring disabled by config")

            if scoring is None and enable_stt_fallback:
                print("[POST] Running STT fallback pipeline")
                transcript_text = await build_transcript_from_stt_segments(
                    all_candidate_audio, transcript_parts
                )
                if not transcript_text.strip():
                    if all_candidate_audio or any(p["role"] == "ai" for p in transcript_parts):
                        transcript_text = "Interviewer: [Interview audio captured but transcription unavailable]"
                    fallback_reason = fallback_reason or "stt_transcript_unavailable"

                if transcript_text.strip():
                    scoring = await score_interview(transcript_text, job_db, candidate_db)
                    scoring_source = "stt_fallback"
                else:
                    fallback_reason = fallback_reason or "no_interview_data"

            if scoring is None:
                print("[POST] Falling back to neutral score due to missing data")
                scoring = {
                    "summary": "Scoring failed due to missing structured memory and transcript data.",
                    "questions": [],
                    "keyword_coverage": [],
                    "strengths": [],
                    "weaknesses": [],
                    "experience_score": 5,
                    "technical_score": 5,
                    "communication_score": 5,
                    "global_score": 50,
                    "recommendation": "maybe",
                    "red_flags": [
                        f"Automated scoring fallback: {fallback_reason or 'unknown_reason'}"
                    ],
                    "theme_scores": {},
                    "tech_stack_match": None,
                }
                scoring_source = "fallback_default"
                transcript_text = (
                    transcript_text.strip()
                    or "Interviewer: [Interview data captured but automated scoring unavailable]"
                )

            scoring["scoring_source"] = scoring_source
            scoring["live_memory_events"] = live_memory_events
            if fallback_reason and scoring_source != "live_memory":
                scoring["fallback_reason"] = fallback_reason

            print(
                f"[POST] Scoring done via {scoring_source}: "
                f"global_score={scoring.get('global_score')}, recommendation={scoring.get('recommendation')}"
            )

            await upsert_interview_result(db, candidate_db.id, transcript_text, scoring)
            apply_scoring_to_candidate(candidate_db, scoring)
            await db.commit()
            print(f"[POST] DB commit done -candidate_id={candidate_id} marked as interview_completed")
            break

    except Exception as e:
        import traceback
        print(f"[POST] ERROR in post-interview processing: {e}")
        traceback.print_exc()
