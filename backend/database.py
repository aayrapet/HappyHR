from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from models import Base, JobConfig
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./happyhr.db")

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        from sqlalchemy import select
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
                    "Tell me about yourself and your background in software development.",
                    "Describe a challenging project you worked on recently. What was your role and what technologies did you use?",
                    "How do you approach debugging a complex issue in production?",
                    "Tell me about your experience with team collaboration and agile workflows.",
                    "Where do you see yourself in the next two years, and what are you looking to learn?"
                ],
                match_threshold=0.2,
                max_interview_minutes=8,
            )
            session.add(seed)
            await session.commit()


async def get_db():
    async with async_session() as session:
        yield session
