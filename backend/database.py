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
        existing_result = await session.execute(select(JobConfig.title))
        existing_titles = {row[0] for row in existing_result.all()}

        defaults = [
            {
                "title": "Full-Stack Developer",
                "description": (
                    "We are looking for a Full-Stack Developer proficient in React, Node.js, "
                    "Python, and cloud technologies. The ideal candidate has 2+ years of experience "
                    "building web applications, strong problem-solving skills, and experience with "
                    "agile methodologies. Experience with TypeScript, Docker, and CI/CD pipelines is a plus."
                ),
                "keywords": [
                    "react", "node.js", "python", "typescript", "javascript",
                    "docker", "aws", "sql", "git", "api", "rest", "agile",
                    "ci/cd", "cloud", "frontend", "backend", "full-stack",
                    "database", "testing", "devops"
                ],
                "mandatory_questions": [
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
                "match_threshold": 0.01,
                "max_interview_minutes": 8,
                "is_active": 1,
                "scoring_weights": {"experience": 0.4, "technical": 0.4, "communication": 0.2},
                "evaluation_themes": ["ownership", "problem_solving", "teamwork", "communication"],
                "tech_stack": ["react", "node.js", "python", "typescript", "docker", "aws", "sql"],
            },
            {
                "title": "Data Scientist",
                "description": (
                    "We are looking for a Data Scientist to build analytical models and generate business "
                    "insights from complex datasets. You will design experiments, build predictive models, "
                    "and communicate findings to stakeholders."
                ),
                "keywords": [
                    "python", "sql", "pandas", "numpy", "scikit-learn", "statistics",
                    "hypothesis testing", "a/b testing", "feature engineering", "data visualization",
                    "tableau", "power bi", "regression", "classification", "mlflow",
                    "etl", "data pipeline", "business insights"
                ],
                "mandatory_questions": [
                    {"question": "Tell me about a data project where your analysis changed a business decision.", "expected_themes": ["business_impact", "communication"]},
                    {"question": "How do you design and interpret an A/B test?", "expected_themes": ["statistics", "experimentation"]},
                    {"question": "How do you handle missing data, outliers, and data quality issues?", "expected_themes": ["data_quality", "robustness"]},
                    {"question": "Explain a model you built, how you evaluated it, and what trade-offs you made.", "expected_themes": ["modeling", "evaluation"]},
                    {"question": "How do you communicate uncertain results to non-technical stakeholders?", "expected_themes": ["communication", "decision_making"]},
                ],
                "match_threshold": 0.12,
                "max_interview_minutes": 10,
                "is_active": 1,
                "scoring_weights": {"experience": 0.35, "technical": 0.4, "communication": 0.25},
                "evaluation_themes": ["statistics", "business_impact", "experimentation", "communication", "model_evaluation"],
                "tech_stack": ["python", "sql", "pandas", "scikit-learn", "tableau", "mlflow"],
            },
            {
                "title": "Machine Learning Engineer",
                "description": (
                    "We are seeking a Machine Learning Engineer to build, deploy, and monitor ML systems "
                    "in production. You will own model lifecycle from training to serving, optimize "
                    "latency and cost, and ensure reliability at scale."
                ),
                "keywords": [
                    "python", "pytorch", "tensorflow", "scikit-learn", "mlops", "docker",
                    "kubernetes", "model serving", "feature store", "airflow", "mlflow",
                    "monitoring", "drift", "ci/cd", "aws", "gcp", "api", "testing"
                ],
                "mandatory_questions": [
                    {"question": "Describe an ML model you deployed to production and the architecture around it.", "expected_themes": ["deployment", "architecture"]},
                    {"question": "How do you monitor model performance and detect data/model drift?", "expected_themes": ["monitoring", "reliability"]},
                    {"question": "How do you ensure reproducibility across experiments and environments?", "expected_themes": ["mlops", "reproducibility"]},
                    {"question": "What trade-offs do you consider between latency, accuracy, and cost?", "expected_themes": ["system_design", "tradeoffs"]},
                    {"question": "How do you collaborate with data scientists and backend engineers in production ML projects?", "expected_themes": ["collaboration", "delivery"]},
                ],
                "match_threshold": 0.12,
                "max_interview_minutes": 10,
                "is_active": 1,
                "scoring_weights": {"experience": 0.3, "technical": 0.5, "communication": 0.2},
                "evaluation_themes": ["mlops", "deployment", "monitoring", "system_design", "collaboration"],
                "tech_stack": ["python", "pytorch", "tensorflow", "docker", "kubernetes", "mlflow", "airflow"],
            },
            {
                "title": "Quantitative Researcher",
                "description": (
                    "We are hiring a Quantitative Researcher to design and validate alpha signals, build "
                    "statistical models, and evaluate portfolio strategies under realistic constraints. "
                    "The role requires strong mathematical rigor and a systematic research mindset."
                ),
                "keywords": [
                    "python", "statistics", "time series", "stochastic processes", "optimization",
                    "factor modeling", "alpha research", "backtesting", "portfolio construction",
                    "risk management", "sharpe ratio", "drawdown", "slippage", "transaction costs",
                    "numpy", "pandas", "sql", "bayesian", "signal processing"
                ],
                "mandatory_questions": [
                    {"question": "Walk me through a strategy you researched from hypothesis to backtest.", "expected_themes": ["research_process", "rigor"]},
                    {"question": "How do you avoid overfitting and control for look-ahead/data-snooping bias?", "expected_themes": ["validation", "methodology"]},
                    {"question": "How do you evaluate a strategy beyond raw returns?", "expected_themes": ["risk", "metrics"]},
                    {"question": "How do transaction costs and market impact change your conclusions?", "expected_themes": ["execution", "realism"]},
                    {"question": "How do you communicate uncertainty and regime risk to decision-makers?", "expected_themes": ["communication", "risk_thinking"]},
                ],
                "match_threshold": 0.10,
                "max_interview_minutes": 12,
                "is_active": 1,
                "scoring_weights": {"experience": 0.3, "technical": 0.55, "communication": 0.15},
                "evaluation_themes": ["research_rigor", "statistical_validity", "risk_awareness", "market_realism", "communication"],
                "tech_stack": ["python", "numpy", "pandas", "sql", "time series", "optimization", "backtesting"],
            },
        ]

        to_create = [JobConfig(**cfg) for cfg in defaults if cfg["title"] not in existing_titles]
        if to_create:
            session.add_all(to_create)
            await session.commit()


async def get_db():
    async with async_session() as session:
        yield session
