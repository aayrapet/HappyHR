<p align="center">
  <img src="https://img.shields.io/badge/HappyHR-AI%20Recruiter-6366f1?style=for-the-badge&logo=sparkles" alt="HappyHR" />
</p>

# ✨ HappyHR

**AI-powered voice interviews.** Screen candidates with a conversational AI recruiter—faster, fairer, and available 24/7.

---

## 🎯 Why HappyHR?

| Old way | HappyHR |
|--------|---------|
| Endless resume triage | **CV parsing + keyword match** → shortlist in seconds |
| Scheduling hell | **Async voice interviews** → candidates interview on their time |
| Inconsistent questions | **Structured AI interviews** → same rubric, every time |
| Ghosting & delays | **Automated invites & decisions** → keep everyone in the loop |

---

## 🛠 Stack

- **Frontend:** Next.js, React, TypeScript, Three.js (avatar / 3D)
- **Backend:** FastAPI, SQLAlchemy (async), WebSockets
- **AI / Voice:** Custom voice pipeline (VAD, lip-sync, talking head)
- **Comms:** Email invites, rejection & decision emails

---

## 🚀 Quick start

```bash
# Backend
cd backend
pip install -r requirements.txt
# Set .env (DB, email, etc.)
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

Make sure to have valid Gemini and Open Ai keys valid in .env
scoring model  = gemini-2.5-flash


Open [http://localhost:3000](http://localhost:3000). Create a job, share the apply link, and run voice interviews.

---

## 📁 Repo layout

```
HappyHR/
├── frontend/     # Next.js app (dashboard, apply flow, interview UI)
├── backend/      # FastAPI (apply, jobs, interviews, email, WebSockets)
├── README.md
└── .env.example  # (add and configure)
```

---

## 📜 License

See [LICENSE](LICENSE).

---

<p align="center">
  <strong>HappyHR</strong> — better hiring, less hassle.
</p>
