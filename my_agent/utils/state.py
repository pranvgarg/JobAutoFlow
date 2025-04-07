from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
import json

class AgentState(BaseModel):
    job_description: str = Field(..., description="The job description text.")
    original_job_description: Optional[str] = Field(None, description="The original job description as provided by the user.")
    industry: Optional[str] = Field(None, description="The industry of the job (e.g., 'Technology', 'Finance').")
    job_role: Optional[str] = Field(None, description="The title of the job role (e.g., 'Data Scientist', 'Software Engineer').")
    company_name: Optional[str] = Field(None, description="The name of the company offering the job.")
    resumes: Optional[List[Dict[str, Any]]] = Field(None, description="List of resumes (structured data). Each resume is a dictionary.")
    optimized_resume: Optional[str] = Field(None, description="The optimized resume text.")
    cover_letter: Optional[str] = Field(None, description="Generated cover letter text.")
    ats_score_before: Optional[float] = Field(None, description="ATS score of the original resume.")
    ats_score_after: Optional[float] = Field(None, description="ATS score of the optimized resume.")
    gap_analysis_results: Optional[Dict[str, Any]] = Field(None, description="Results of gap analysis between resume and job description.")
    optimization_recommendations: Optional[List[str]] = Field(None, description="Recommendations for resume optimization.")
    gpt_health: Optional[str] = Field(None, description="Status of GPT model health check ('OK' or 'Error').")
    extracted_job_description: Optional[str] = Field(None, description="The raw output from job description extraction (JSON string).")
    company_role_info: Optional[str] = Field(None, description="Additional company/role information from TavilySearch.")
    semantic_match_score: Optional[float] = Field(None, description="Semantic matching score between resume and job description.")
    user_feedback: Optional[str] = Field(None, description="User feedback after final review of the documents.")
    exported_documents: Optional[str] = Field(None, description="Final exported documents (cover letter, cold email, optimized resume).")
    error: Optional[str] = Field(None, description="Error messages if any occur during processing.")
    requirements: Optional[str] = Field(None, description="Job requirements extracted from the job description.")
    responsibilities: Optional[str] = Field(None, description="Job responsibilities extracted from the job description.")
    benefits: Optional[str] = Field(None, description="Job benefits extracted from the job description.")
    location: Optional[str] = Field(None, description="Job location extracted from the job description.")
    salary: Optional[str] = Field(None, description="Job salary extracted from the job description.")
    about_company: Optional[str] = Field(None, description="Information about the company extracted from the job description.")
    embeddings: List[List[float]] = []
    cold_email: Optional[str] = Field(None, description="Generated cold email text.")
    skills: Optional[List[str]] = Field(None, description="List of skills extracted from the job description.")


    @validator("job_description")
    def job_description_cannot_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Job description cannot be empty.")
        return v
    
    @validator("original_job_description", "industry", "job_role", "company_name", "optimized_resume", "cover_letter", "gpt_health", "company_role_info", "user_feedback", "exported_documents", "error", "requirements", "responsibilities", "benefits", "location", "salary", "about_company")
    def string_fields_must_be_strings(cls, v):
        if v is not None and not isinstance(v, str):
            raise ValueError(f"This field must be a string.")
        return v

    @validator("resumes")
    def resumes_must_be_list_of_non_empty_dictionaries(cls, v):
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("Resumes must be a list of dictionaries.")
            for resume in v:
                if not isinstance(resume, dict):
                    raise ValueError("Each resume in the list must be a dictionary.")
                if not resume:
                    raise ValueError("Each resume in the list cannot be empty.")
        return v

    @validator("gap_analysis_results")
    def gap_analysis_results_must_be_dict(cls, v):
        if v is not None and not isinstance(v, dict):
            raise ValueError("Gap analysis results must be a dictionary.")
        return v

    @validator("optimization_recommendations")
    def optimization_recommendations_must_be_list_of_strings(cls, v):
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("Optimization recommendations must be a list of strings.")
            for recommendation in v:
                if not isinstance(recommendation, str):
                    raise ValueError("Each optimization recommendation in the list must be a string.")
        return v
    
    @validator("extracted_job_description")
    def extracted_job_description_must_be_valid_json(cls, v):
        if v is not None:
            try:
                json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("Extracted job description must be a valid JSON string.")
        return v

    @validator("semantic_match_score", "ats_score_before", "ats_score_after")
    def score_must_be_float(cls, v):
        if v is not None and not isinstance(v, float):
            raise ValueError(f"This score must be a float.")
        return v
