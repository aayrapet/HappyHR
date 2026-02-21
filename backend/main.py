from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
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


# ── Apply ────────────────────────────────────────────────

@app.post("/api/apply")
async def apply(
    first_name: str = Form(...),
    last_name: str = Form(...),
    email: str = Form(...),
    cv: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    file_bytes = await cv.read()
    cv_text = extract_pdf_text(file_bytes)

    if not cv_text.strip():
        raise HTTPException(400, "Could not extract text from PDF")

    # Get job config
    result = await db.execute(select(JobConfig).limit(1))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(500, "No job configuration found")

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
        send_rejection_email(email, first_name, job.title)

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

    return {
        "candidate_name": f"{candidate.first_name} {candidate.last_name}",
        "job_title": job.title,
        "job_description": job.description,
        "keywords": job.keywords,
        "mandatory_questions": job.mandatory_questions,
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

    job_result = await db.execute(select(JobConfig).where(JobConfig.id == candidate.job_config_id))
    job = job_result.scalar_one_or_none()

    # Score with Gemini text model
    scoring = await score_interview(body.transcript, job, candidate)

    interview_result = InterviewResult(
        candidate_id=candidate.id,
        transcript=body.transcript,
        summary=scoring.get("summary"),
        questions=scoring.get("questions"),
        keyword_coverage=scoring.get("keyword_coverage"),
        global_score=scoring.get("global_score"),
        recommendation=scoring.get("recommendation"),
        red_flags=scoring.get("red_flags"),
        raw_scoring_json=scoring,
    )
    db.add(interview_result)

    candidate.status = "interview_completed"
    candidate.global_score = scoring.get("global_score")
    candidate.recommendation = scoring.get("recommendation")
    await db.commit()

    return {"status": "scored", "global_score": scoring.get("global_score")}


async def score_interview(transcript: str, job: JobConfig, candidate: Candidate) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    scoring_model = os.getenv("SCORING_MODEL", "gemini-2.5-flash-lite")

    prompt = f"""You are an expert HR interview evaluator. Analyze this interview transcript and produce a structured evaluation.

Job Title: {job.title}
Job Description: {job.description}
Required Keywords: {json.dumps(job.keywords)}
Mandatory Questions: {json.dumps(job.mandatory_questions)}

Candidate: {candidate.first_name} {candidate.last_name}

TRANSCRIPT:
{transcript}

Produce a JSON response with EXACTLY this structure:
{{
  "summary": "2-3 sentence overall summary of the candidate's performance",
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
  "global_score": 0-100,
  "recommendation": "strong_yes|yes|maybe|no",
  "red_flags": ["any concerns noted during the interview"]
}}

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
        return json.loads(text.strip())
    except Exception as e:
        print(f"Scoring error: {e}")
        return {
            "summary": "Scoring failed due to an error.",
            "questions": [],
            "keyword_coverage": [],
            "global_score": 50,
            "recommendation": "maybe",
            "red_flags": [f"Automated scoring failed: {str(e)}"],
        }


# ── Dashboard Endpoints ─────────────────────────────────

@app.get("/api/candidates")
async def list_candidates(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Candidate)
        .where(Candidate.status.in_(["interview_completed", "accepted", "rejected_after_interview"]))
        .order_by(Candidate.global_score.desc().nullslast())
    )
    candidates = result.scalars().all()
    return [
        {
            "id": c.id,
            "first_name": c.first_name,
            "last_name": c.last_name,
            "email": c.email,
            "status": c.status,
            "global_score": c.global_score,
            "recommendation": c.recommendation,
            "match_percent": c.match_percent,
            "created_at": str(c.created_at) if c.created_at else None,
        }
        for c in candidates
    ]


@app.get("/api/candidate/{candidate_id}")
async def get_candidate(candidate_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
    c = result.scalar_one_or_none()
    if not c:
        raise HTTPException(404, "Candidate not found")

    ir_result = await db.execute(
        select(InterviewResult).where(InterviewResult.candidate_id == candidate_id)
    )
    interview = ir_result.scalar_one_or_none()

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
            "questions": interview.questions,
            "keyword_coverage": interview.keyword_coverage,
            "global_score": interview.global_score,
            "recommendation": interview.recommendation,
            "red_flags": interview.red_flags,
        } if interview else None,
    }


class DecisionRequest(BaseModel):
    decision: str  # "accept" or "reject"


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

    new_status = "accepted" if body.decision == "accept" else "rejected_after_interview"
    c.status = new_status
    await db.commit()

    send_decision_email(c.email, c.first_name, job.title if job else "the position", body.decision)

    return {"status": new_status}


# ── Job Config (for dashboard) ───────────────────────────

@app.get("/api/job-config")
async def get_job_config(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(JobConfig).limit(1))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "No job config")
    return {
        "id": job.id,
        "title": job.title,
        "description": job.description,
        "keywords": job.keywords,
        "mandatory_questions": job.mandatory_questions,
    }


# ── WebSocket Interview Relay ──────────────────────────────

def build_system_instruction(ctx: dict) -> str:
    questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(ctx["mandatory_questions"]))
    keywords = ", ".join(ctx["keywords"])
    return f"""You are an experienced HR recruiter named Sarah conducting a short phone screen for the position of {ctx["job_title"]}.

JOB DESCRIPTION:
{ctx["job_description"]}

MANDATORY QUESTIONS (you MUST ask all of these, in any order):
{questions}

KEYWORDS TO PROBE FOR:
{keywords}

CANDIDATE CONTEXT:
Name: {ctx["candidate_name"]}
CV Summary (truncated):
{ctx["cv_text"]}

CONVERSATION RULES:
- Greet the candidate warmly by name and introduce yourself as Sarah from HappyHR.
- Ask ONE question at a time.
- Keep your responses concise and natural - this should feel like a real phone conversation.
- Listen carefully and ask relevant follow-up questions when appropriate.
- Probe for evidence of the listed keywords through natural conversation.
- After all mandatory questions are covered, ask if the candidate has any questions about the role.
- End the interview gracefully when: (a) all mandatory questions are done AND (b) the candidate has no more questions, OR after {ctx["max_interview_minutes"]} minutes.
- When ending, thank the candidate and let them know they'll hear back soon.

SAFETY/FAIRNESS RULES:
- Do NOT infer or ask about protected attributes (race, religion, health, disability, age, gender, sexual orientation, marital status, etc.).
- Evaluate ONLY job-relevant skills and experience.
- Be equally warm and professional with all candidates."""


def build_gemini_ws_url() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    return (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        f"?key={api_key}"
    )


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


async def transcribe_audio_with_gemini(audio_chunks: list[str]) -> str:
    """Transcribe candidate audio chunks using Gemini STT."""
    api_key = os.getenv("GEMINI_API_KEY")
    stt_model = os.getenv("STT_MODEL", "gemini-2.5-flash")

    if not audio_chunks:
        return ""

    # Concatenate base64 chunks into raw PCM buffer
    pcm_parts = [base64.b64decode(chunk) for chunk in audio_chunks]
    pcm_buffer = b"".join(pcm_parts)

    if len(pcm_buffer) < 6000:  # Too short to be useful
        return ""

    wav_bytes = pcm16_mono_to_wav(pcm_buffer, 16000)
    wav_b64 = base64.b64encode(wav_bytes).decode()

    from google import genai
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=stt_model,
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "Transcribe this audio exactly. Output only the transcript text, no commentary."},
                    {"inline_data": {"mime_type": "audio/wav", "data": wav_b64}},
                ],
            }
        ],
    )
    return response.text.strip() if response.text else ""


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

    ctx = {
        "candidate_name": f"{candidate.first_name} {candidate.last_name}",
        "job_title": job.title,
        "job_description": job.description,
        "keywords": job.keywords,
        "mandatory_questions": job.mandatory_questions,
        "cv_text": candidate.cv_text[:3000],
        "max_interview_minutes": job.max_interview_minutes,
    }

    await ws.accept()

    gemini_model = os.getenv("LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
    gemini_url = build_gemini_ws_url()

    # Track transcript for scoring
    model_text_buffer = ""
    transcript_parts = []  # list of {"role": "ai"|"user", "text": str, "index": int}
    candidate_audio_chunks = []  # current speech segment
    all_candidate_audio = []  # list of {"index": int, "chunks": list[str]}
    turn_index = 0

    try:
        async with websockets.connect(gemini_url) as gemini_ws:
            # Send setup message
            setup_msg = {
                "setup": {
                    "model": f"models/{gemini_model}" if not gemini_model.startswith("models/") else gemini_model,
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                    },
                    "systemInstruction": {
                        "parts": [{"text": build_system_instruction(ctx)}]
                    },
                }
            }
            await gemini_ws.send(json.dumps(setup_msg))

            # Wait for setupComplete (loop in case other messages arrive first)
            while True:
                setup_response = await gemini_ws.recv()
                setup_data = json.loads(setup_response)
                print(f"[WS] Gemini setup response: {list(setup_data.keys())}")
                if setup_data.get("setupComplete") is not None:
                    break

            await ws.send_json({"type": "status", "message": "Connected to Gemini Live API"})
            print("[WS] Setup complete, sending initial trigger to Gemini")

            # Send initial trigger so Gemini starts the interview
            await gemini_ws.send(json.dumps({
                "clientContent": {
                    "turns": [{"role": "user", "parts": [{"text": "Hello, I'm ready to start the interview."}]}],
                    "turnComplete": True,
                }
            }))

            async def gemini_to_client():
                """Forward Gemini messages to the browser client."""
                nonlocal model_text_buffer, turn_index
                try:
                    async for raw in gemini_ws:
                        msg = json.loads(raw)
                        print(f"[WS] Gemini → client: keys={list(msg.keys())}")

                        parts = msg.get("serverContent", {}).get("modelTurn", {}).get("parts", [])
                        for part in parts:
                            inline = part.get("inlineData")
                            if inline and inline.get("data") and inline.get("mimeType", "").startswith("audio/pcm"):
                                print(f"[WS] Sending audio chunk to client ({len(inline['data'])} b64 bytes)")
                                await ws.send_json({
                                    "type": "audio",
                                    "mimeType": inline["mimeType"],
                                    "data": inline["data"],
                                })

                            text = part.get("text")
                            if text:
                                model_text_buffer += (" " if model_text_buffer else "") + text
                                await ws.send_json({"type": "text", "text": text})

                        if msg.get("serverContent", {}).get("turnComplete"):
                            if model_text_buffer.strip():
                                transcript_parts.append({"role": "ai", "text": model_text_buffer.strip(), "index": turn_index})
                                turn_index += 1
                                model_text_buffer = ""
                            await ws.send_json({"type": "turnComplete"})

                        if msg.get("error"):
                            await ws.send_json({"type": "error", "error": msg["error"]})

                except websockets.ConnectionClosed as e:
                    print(f"[WS] Gemini connection closed: {e}")
                except Exception as e:
                    print(f"[WS] gemini_to_client error: {e}")

            async def client_to_gemini():
                """Forward browser client messages to Gemini."""
                nonlocal candidate_audio_chunks, turn_index
                try:
                    while True:
                        data = await ws.receive_text()
                        msg = json.loads(data)

                        if msg.get("type") == "audio" and msg.get("data"):
                            # Store audio for STT
                            candidate_audio_chunks.append(msg["data"])
                            # Forward to Gemini
                            await gemini_ws.send(json.dumps({
                                "realtimeInput": {
                                    "mediaChunks": [{
                                        "mimeType": msg.get("mimeType", "audio/pcm;rate=16000"),
                                        "data": msg["data"],
                                    }]
                                }
                            }))

                        elif msg.get("type") == "speechStart":
                            candidate_audio_chunks = []

                        elif msg.get("type") == "speechEnd":
                            if candidate_audio_chunks:
                                idx = turn_index
                                turn_index += 1
                                all_candidate_audio.append({"index": idx, "chunks": list(candidate_audio_chunks)})
                                # Add placeholder in transcript
                                transcript_parts.append({"role": "user", "text": "", "index": idx})
                            candidate_audio_chunks = []

                        elif msg.get("type") == "text" and msg.get("text"):
                            transcript_parts.append({"role": "user", "text": msg["text"], "index": turn_index})
                            turn_index += 1
                            await gemini_ws.send(json.dumps({
                                "clientContent": {
                                    "turns": [{"role": "user", "parts": [{"text": msg["text"]}]}],
                                    "turnComplete": True,
                                }
                            }))

                except WebSocketDisconnect:
                    print("[WS] Client disconnected")
                except Exception as e:
                    print(f"[WS] client_to_gemini error: {e}")

            # Run both relay tasks concurrently
            await asyncio.gather(
                gemini_to_client(),
                client_to_gemini(),
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

    # Post-interview: transcribe candidate audio and fill in transcript placeholders
    try:
        for audio_entry in all_candidate_audio:
            try:
                stt_text = await transcribe_audio_with_gemini(audio_entry["chunks"])
                if stt_text:
                    # Find and update the placeholder entry
                    for part in transcript_parts:
                        if part.get("index") == audio_entry["index"] and part["role"] == "user":
                            part["text"] = stt_text
                            break
            except Exception as e:
                print(f"STT error: {e}")

        # Sort by index and build transcript string
        transcript_parts.sort(key=lambda p: p.get("index", 0))
        transcript_text = "\n\n".join(
            f"{'Interviewer' if p['role'] == 'ai' else 'Candidate'}: {p['text']}"
            for p in transcript_parts
            if p.get("text")
        )

        if transcript_text.strip():
            async for db in get_db():
                scoring = await score_interview(transcript_text, job, candidate)
                interview_result = InterviewResult(
                    candidate_id=candidate.id,
                    transcript=transcript_text,
                    summary=scoring.get("summary"),
                    questions=scoring.get("questions"),
                    keyword_coverage=scoring.get("keyword_coverage"),
                    global_score=scoring.get("global_score"),
                    recommendation=scoring.get("recommendation"),
                    red_flags=scoring.get("red_flags"),
                    raw_scoring_json=scoring,
                )
                db.add(interview_result)
                candidate.status = "interview_completed"
                candidate.global_score = scoring.get("global_score")
                candidate.recommendation = scoring.get("recommendation")
                await db.commit()
                break

    except Exception as e:
        print(f"Post-interview processing error: {e}")
