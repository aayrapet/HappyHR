from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from models import Base, JobConfig
from sqlalchemy import select, text
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./happyhr.db")

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Migrate: add is_active column for existing databases
        # Migrate: add new columns for existing databases
        for stmt in [
            "ALTER TABLE job_configs ADD COLUMN is_active INTEGER DEFAULT 0",
            "ALTER TABLE job_configs ADD COLUMN scoring_weights JSON",
            "ALTER TABLE job_configs ADD COLUMN evaluation_themes JSON",
            "ALTER TABLE job_configs ADD COLUMN tech_stack JSON",
            "ALTER TABLE interview_results ADD COLUMN strengths JSON",
            "ALTER TABLE interview_results ADD COLUMN weaknesses JSON",
            "ALTER TABLE interview_results ADD COLUMN experience_score FLOAT",
            "ALTER TABLE interview_results ADD COLUMN technical_score FLOAT",
            "ALTER TABLE interview_results ADD COLUMN communication_score FLOAT",
            "ALTER TABLE interview_results ADD COLUMN theme_scores JSON",
            "ALTER TABLE interview_results ADD COLUMN tech_stack_match VARCHAR",
        ]:
            try:
                await conn.execute(text(stmt))
            except Exception:
                pass  # column already exists
        # Ensure at least one config is active
        await conn.execute(text(
            """
            UPDATE job_configs SET is_active = 1
            WHERE id = (SELECT MIN(id) FROM job_configs)
              AND NOT EXISTS (SELECT 1 FROM job_configs WHERE is_active = 1)
            """
        ))
        # Keep one interview result per candidate before enforcing uniqueness.
        await conn.execute(text(
            """
            DELETE FROM interview_results
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM interview_results
                GROUP BY candidate_id
            )
            """
        ))
        # Backfill candidates that already have interview results but were never marked completed.
        await conn.execute(text(
            """
            UPDATE candidates
            SET
                status = 'interview_completed',
                global_score = COALESCE(
                    (
                        SELECT ir.global_score
                        FROM interview_results ir
                        WHERE ir.candidate_id = candidates.id
                        ORDER BY ir.id DESC
                        LIMIT 1
                    ),
                    global_score
                ),
                recommendation = COALESCE(
                    (
                        SELECT ir.recommendation
                        FROM interview_results ir
                        WHERE ir.candidate_id = candidates.id
                        ORDER BY ir.id DESC
                        LIMIT 1
                    ),
                    recommendation
                )
            WHERE status = 'invited'
              AND EXISTS (
                  SELECT 1
                  FROM interview_results ir
                  WHERE ir.candidate_id = candidates.id
              )
            """
        ))
        await conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_interview_results_candidate_id ON interview_results(candidate_id)"
        ))

    async with async_session() as session:
        result = await session.execute(select(JobConfig).limit(1))
        if result.scalar_one_or_none() is None:
            seed = JobConfig(
                title="Full-Stack Developer",
                description=(
                    "We are looking for a Full-Stack Developer proficient in React, Node.js, "
                    "Python, and cloud technologies. The ideal candidate has 2+ years of experience "
                    "building web applications, strong problem-solving skills, and experience with "
                    "agile methodologies. Experience with TypeScript, Docker, and CI/CD pipelines is a plus."
                ),
                keywords=[
                    "react", "node.js", "python", "typescript", "javascript",
                    "docker", "aws", "sql", "git", "api", "rest", "agile",
                    "ci/cd", "cloud", "frontend", "backend", "full-stack",
                    "database", "testing", "devops"
                ],
                mandatory_questions=[
                    {
                        "question": "Walk me through a full-stack feature you built end-to-end — from database schema to UI. What technical choices did you make and why?",
                        "expected_themes": ["data modeling / schema design", "API design (REST or GraphQL)", "frontend state management", "trade-offs explained"],
                    },
                    {
                        "question": "Tell me about a time you had to debug a critical issue in production. How did you approach it?",
                        "expected_themes": ["observability tools (logs, metrics, traces)", "systematic isolation of root cause", "impact mitigation while investigating", "post-mortem or preventive measures"],
                    },
                    {
                        "question": "How do you approach code quality and testing in a team environment?",
                        "expected_themes": ["types of tests written (unit, integration, e2e)", "code review process", "CI/CD integration", "handling legacy or untested code"],
                    },
                    {
                        "question": "Describe a project where you had to learn a new technology or framework under time pressure. How did you handle it?",
                        "expected_themes": ["learning strategy (docs, sandbox, community)", "managing delivery risk", "what they retained long-term", "communication with team about uncertainty"],
                    },
                ],
                match_threshold=0.01,
                max_interview_minutes=8,
                is_active=1,
            )
            session.add(seed)
            await session.commit()


async def get_db():
    async with async_session() as session:
        yield session
