from sqlalchemy import Column, Integer, String, Float, Text, DateTime, JSON, func
from sqlalchemy.orm import DeclarativeBase
import datetime


class Base(DeclarativeBase):
    pass


class JobConfig(Base):
    __tablename__ = "job_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    keywords = Column(JSON, nullable=False)  # list of strings
    mandatory_questions = Column(JSON, nullable=False)  # list of strings
    match_threshold = Column(Float, default=0.3)
    max_interview_minutes = Column(Integer, default=8)
    created_at = Column(DateTime, default=func.now())


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    cv_text = Column(Text, nullable=False)
    cv_filename = Column(String, nullable=False)
    match_percent = Column(Float, default=0.0)
    status = Column(String, default="applied")  # applied, invited, interview_completed, accepted, rejected
    interview_token = Column(String, unique=True, nullable=True)
    job_config_id = Column(Integer, default=1)
    global_score = Column(Float, nullable=True)
    recommendation = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.now())


class InterviewResult(Base):
    __tablename__ = "interview_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    candidate_id = Column(Integer, nullable=False)
    transcript = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    summary_candidate = Column(Text, nullable=True)
    questions = Column(JSON, nullable=True)  # list of question objects
    keyword_coverage = Column(JSON, nullable=True)
    global_score = Column(Float, nullable=True)
    recommendation = Column(String, nullable=True)
    red_flags = Column(JSON, nullable=True)
    raw_scoring_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.now())
