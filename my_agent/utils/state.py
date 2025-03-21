# state.py

from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    """Represents the state of the resume processing workflow."""
    job_description: str  # The job description text
    job_role: str         # The role for which resumes are being processed
    resumes: List[Dict[str, Any]]  # List of extracted resumes (structured data)
    cover_letter: str     # Generated cover letter
    optimized_resume: str # Optimized resume content
    ats_score_before: float  # ATS score before optimization
    ats_score_after: float   # ATS score after optimization
    feedback: str          # Feedback from the evaluator
    improvement_needed: str # Indicates if further improvement is needed
    messages: List[Dict[str, Any]]  # Messages for model interactions

# Example of how to initialize a State instance
initial_state = AgentState(
    job_description="",
    job_role="",
    resumes=[],
    cover_letter="",
    optimized_resume="",
    ats_score_before=0.0,
    ats_score_after=0.0,
    feedback="",
    improvement_needed="no",  # Default value can be "yes" or "no"
    messages=[]
)
