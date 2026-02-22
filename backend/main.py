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
import fitz  # PyMuPDF
from dotenv import load_dotenv
import websockets

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
        scoring = json.loads(text.strip())

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

============================================
CONVERSATION STYLE RULES
============================================
- Ask ONE question at a time. Never combine multiple questions.
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
                        sample_rate_hertz=16000,
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
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {
                                    "voiceName": "Aoede"
                                }
                            }
                        },
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
                finally:
                    # Close Gemini WS so gemini_to_client() unblocks and exits
                    print("[WS] Closing Gemini connection to unblock relay task")
                    try:
                        await gemini_ws.close()
                    except Exception:
                        pass

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

    # Flush any AI text that was mid-turn when the client disconnected
    if model_text_buffer.strip():
        transcript_parts.append({"role": "ai", "text": model_text_buffer.strip(), "index": turn_index})
        model_text_buffer = ""

    # Post-interview: transcribe candidate audio and fill in transcript placeholders
    print(f"[POST] Starting post-interview processing for candidate_id={candidate_id}")
    print(f"[POST] Audio segments to transcribe: {len(all_candidate_audio)}, AI transcript parts: {len([p for p in transcript_parts if p['role'] == 'ai'])}")
    try:
        for i, audio_entry in enumerate(all_candidate_audio):
            try:
                print(f"[POST] Transcribing audio segment {i+1}/{len(all_candidate_audio)} (index={audio_entry['index']}, chunks={len(audio_entry['chunks'])})")
                stt_text = await transcribe_audio_with_stt_v2(audio_entry["chunks"])
                print(f"[POST] STT result for segment {i+1}: '{stt_text[:80]}...' " if len(stt_text) > 80 else f"[POST] STT result for segment {i+1}: '{stt_text}'")
                if stt_text:
                    for part in transcript_parts:
                        if part.get("index") == audio_entry["index"] and part["role"] == "user":
                            part["text"] = stt_text
                            break
            except Exception as e:
                import traceback
                print(f"[POST] STT error for segment {i+1}: {e}")
                traceback.print_exc()

        # Sort by index and build transcript string
        transcript_parts.sort(key=lambda p: p.get("index", 0))
        filled_parts = [p for p in transcript_parts if p.get("text")]
        print(f"[POST] Transcript parts after STT: {len(filled_parts)} non-empty (of {len(transcript_parts)} total)")
        transcript_text = "\n\n".join(
            f"{'Interviewer' if p['role'] == 'ai' else 'Candidate'}: {p['text']}"
            for p in filled_parts
        )
        print(f"[POST] Final transcript length: {len(transcript_text)} chars")

        # Score even if transcript is empty (candidate connected but STT failed)
        # Use a fallback placeholder so the candidate still appears in the dashboard
        if not transcript_text.strip():
            if all_candidate_audio or any(p["role"] == "ai" for p in transcript_parts):
                print("[POST] WARNING: transcript is empty but interview happened -using fallback transcript")
                transcript_text = "Interviewer: [Interview audio captured but transcription unavailable]"
            else:
                print("[POST] No interview data at all -skipping scoring")

        if transcript_text.strip():
            print(f"[POST] Starting scoring for candidate_id={candidate_id}")
            async for db in get_db():
                candidate_result = await db.execute(
                    select(Candidate).where(Candidate.id == candidate_id)
                )
                candidate_db = candidate_result.scalar_one_or_none()
                if not candidate_db:
                    print(f"[POST] ERROR: candidate_id={candidate_id} not found in DB")
                    break

                existing_interview = await get_interview_result_for_candidate(db, candidate_id)
                if candidate_db.status == "interview_completed" and existing_interview:
                    print(f"[POST] Already scored -skipping")
                    break

                job_result = await db.execute(
                    select(JobConfig).where(JobConfig.id == candidate_db.job_config_id)
                )
                job_db = job_result.scalar_one_or_none()
                if not job_db:
                    print(f"[POST] ERROR: job_config not found for candidate_id={candidate_id}")
                    break

                scoring = await score_interview(transcript_text, job_db, candidate_db)
                print(f"[POST] Scoring done: global_score={scoring.get('global_score')}, recommendation={scoring.get('recommendation')}")
                await upsert_interview_result(db, candidate_db.id, transcript_text, scoring)
                apply_scoring_to_candidate(candidate_db, scoring)
                await db.commit()
                print(f"[POST] DB commit done -candidate_id={candidate_id} marked as interview_completed")
                break

    except Exception as e:
        import traceback
        print(f"[POST] ERROR in post-interview processing: {e}")
        traceback.print_exc()
