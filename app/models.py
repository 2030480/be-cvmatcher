from pydantic import BaseModel
from typing import List

class Strength(BaseModel):
    title: str
    description: str

class Weakness(BaseModel):
    title: str
    description: str
    suggestion: str

class AnalysisResult(BaseModel):
    match_percentage: int
    strengths: List[Strength]
    weaknesses: List[Weakness]
    summary: str
    
class CVData(BaseModel):
    skills: List[str]
    experience_years: int
    education: List[str]
    job_titles: List[str]
    
class JobRequirements(BaseModel):
    required_skills: List[str]
    preferred_skills: List[str]
    experience_required: int
    education_requirements: List[str] 