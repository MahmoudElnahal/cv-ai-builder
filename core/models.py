from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional

class JD(BaseModel):
    title: Optional[str] = None
    text: str

class CVSectionExperience(BaseModel):
    company: Optional[str] = None
    title: Optional[str] = None
    from_date: Optional[str] = None  # YYYY-MM
    to_date: Optional[str] = None
    bullets: List[str] = []

class CV(BaseModel):
    name: str = "Candidate Name"
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    links: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    skills: List[str] = []
    experience: List[CVSectionExperience] = []
    education: List[str] = []
    projects: List[str] = []
    certificates: List[str] = []

class MatchReport(BaseModel):
    score: float = 0.0
    missing_keywords: List[str] = []
    matched_keywords: List[str] = []
    notes: Optional[str] = None
