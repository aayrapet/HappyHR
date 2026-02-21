from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel
from typing import Optional
import secrets
import os
import json
import fitz  # PyMuPDF
from dotenv import load_dotenv

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

def extract_pdf_text(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def keyword_match(cv_text: str, keywords: list[str]) -> float:
    cv_lower = cv_text.lower()
    matched = sum(1 for kw in keywords if kw.lower() in cv_lower)
    return matched / len(keywords) if keywords else 0.0


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
