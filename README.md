<p align="center">
  <img src="https://img.shields.io/badge/HappyHR-AI%20Recruiter-6366f1?style=for-the-badge&logo=sparkles" alt="HappyHR" />
  <a href="https://devpost.com/software/happyhr">
    <img src="https://img.shields.io/badge/Devpost-HappyHR-003E54?style=for-the-badge&logo=devpost" alt="Devpost" />
  </a>
</p>

# ✨ HappyHR

**AI-powered voice interviews.** Screen candidates with a conversational AI recruiter—faster, fairer, and available 24/7.

> 🏆 Built at **HackEurope 2026** in ~20 hours.

---

## 🎯 Why HappyHR?

| Old way | HappyHR |
|--------|---------|
| Endless resume triage | **CV parsing + keyword match** → shortlist in seconds |
| Scheduling hell | **Async voice interviews** → candidates interview on their time |
| Inconsistent questions | **Structured AI interviews** → same rubric, every time |
| Ghosting & delays | **Automated invites & decisions** → keep everyone in the loop |

---

## 🏆 About

We wanted to make early-stage hiring faster, fairer, and less stressful for both candidates and recruiters—keeping the process structured while still feeling conversational, especially for voice interviews with personalized questions and follow-ups.

HappyHR lets candidates apply to role-specific job posts, upload a PDF CV, and get screened automatically. If they pass, they receive an email invite to a live AI voice interview with an animated avatar. Recruiters then see structured scoring, transcript summaries, strengths/weaknesses, and can send accept/reject decisions from a dashboard.

---

## 🛠 Tech Stack

| Layer | Tools |
|-------|-------|
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind CSS, Three.js (3D avatar) |
| **Backend** | FastAPI, SQLAlchemy (async), SQLite + aiosqlite, WebSockets |
| **CV Screening** | PyMuPDF (PDF extraction), spaCy (NLP), SentenceTransformers (semantic embeddings) |
| **Voice Interview** | OpenAI Realtime API, WebSocket relay, RMS-based VAD |
| **AI Scoring** | Gemini 2.5 Flash (structured scoring agent) |
| **Speech-to-Text** | Google Cloud Speech-to-Text (STT fallback) |
| **Comms** | SMTP (invite, rejection & decision emails) |

---

## 🔄 Pipeline

```
Candidate applies
      │
      ▼
PDF CV uploaded
      │
      ▼
PyMuPDF extracts text
      │
      ▼
Hybrid screening: spaCy (lexical) + SentenceTransformers (semantic) + gemini flash summary
      │
  ┌───┴───┐
Below     Above threshold
threshold     │
  │           ▼
  │     Email invite sent (SMTP)
  │           │
  ▼           ▼
Reject   Candidate joins WebSocket interview session
         Avatar activates (idle → talking state)
              │
              ▼
         RMS VAD detects speech → 24kHz PCM audio streamed to backend
              │
              ▼
         OpenAI GPT Realtime agent conducts personalized interview
         (CV-aware questions + recruiter mandatory questions + follow-ups)
              │
              ▼
         Post-interview: Gemini 2.5 Flash scoring agent
              │
         ┌────┴────┐
    Live memory    STT fallback
    scoring        (Google Cloud Speech-to-Text)
         └────┬────┘
              │
              ▼
         HR dashboard: scores, transcript, strengths/weaknesses, recommendation
              │
              ▼
         Recruiter sends accept/reject → email to candidate
```

---

## 🤖 Agents

### Interview Agent — OpenAI GPT Realtime
- Runs over a persistent WebSocket connection with a `tool_call` schema
- Reads the candidate's CV and recruiter-configured mandatory questions to drive the conversation
- Calls `assess_answer` per question for real-time live-memory scoring
- Calls `end_interview` to terminate gracefully once all questions are covered
- Handles audio echo cancellation and speech interruption

### Scoring Agent — Gemini 2.5 Flash
- Triggered after interview completion
- Input: full transcript + job config (keywords, expected themes, scoring weights)
- Output: structured JSON containing:
  - Global score (0–100) and recommendation (`strong_yes` / `yes` / `maybe` / `no`)
  - Per-question scores (0–5) with rationale
  - Component scores: experience, technical, communication (0–10 each)
  - Red flags and keyword coverage

> **Note on our journey:** We initially tried Gemini Live for the voice agent, but ran into issues with function calling in turn-based reaction. After extensive debugging we switched the interview pipeline to OpenAI GPT Realtime, which gave us reliable tool-call support.

---

## 🚀 Quick Start

```bash
# Backend
cd backend
pip install -r requirements.txt
cp .env.example .env          # fill in your API keys
uvicorn main:app --reload

# Frontend
cd frontend
npm install
cp .env.example .env.local    # adjust if needed
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Create a job config, share the apply link, and run voice interviews.

---

## 📁 Repo Layout

```
HappyHR/
├── frontend/     # Next.js app (dashboard, apply flow, interview UI)
├── backend/      # FastAPI (apply, jobs, interviews, email, WebSockets)
├── README.md
└── backend/.env.example
```

---

## 📜 License

See [LICENSE](LICENSE).

---

<p align="center">
  <strong>HappyHR</strong> — better hiring, less hassle.<br/>
  🔗 <a href="https://devpost.com/software/happyhr">View on Devpost</a>
</p>
