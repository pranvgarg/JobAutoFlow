from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator

class AgentState(BaseModel):
    job_description: str = Field(..., description="The job description to process.")
    industry: Optional[str] = Field(None, description="The industry of the job.")
    job_role: Optional[str] = Field(None, description="The job role.")
    company_name: Optional[str] = Field(None, description="The company name.")
    resumes: Optional[List[str]] = Field(None, description="List of resumes (text).")
    optimized_resume: Optional[str] = Field(None, description="The optimized resume.")
    improvement_needed: Optional[str] = Field(None, description="'yes' or 'no' indicating if resume needs improvement.")
    feedback: Optional[str] = Field(None, description="Feedback on the resume.")
    cover_letter: Optional[str] = Field(None, description="Generated cover letter.")
    ats_score_before: Optional[float] = Field(None, description="ATS score before optimization.")
    ats_score_after: Optional[float] = Field(None, description="ATS score after optimization.")
    messages: Optional[List[Dict[str, Any]]] = Field(None, description="List of messages for the LLM chain.")
    embeddings: Optional[List[List[float]]] = Field(None, description="Embeddings for the resumes.")
    pinecone_ids: Optional[List[str]] = Field(None, description="Pinecone IDs for the resume embeddings.")
    gap_analysis_results: Optional[Dict[str, Any]] = Field(None, description="Results of gap analysis.")
    optimization_recommendations: Optional[List[str]] = Field(None, description="Recommendations for resume optimization.")
    gpt_health: Optional[str] = Field(None, description="Status of GPT model health check.")
    extracted_job_description: Optional[str] = Field(None, description="The raw output from job description extraction.")
    company_role_info: Optional[str] = Field(None, description="Additional company/role info from TavilySearch.")
    semantic_match_score: Optional[float] = Field(None, description="Semantic matching score.")
    user_feedback: Optional[str] = Field(None, description="User feedback after final review.")
    exported_documents: Optional[str] = Field(None, description="Final exported documents.")
    error: Optional[str] = Field(None, description="Error messages if any occur during processing.")

    @validator("job_description")
    def job_description_cannot_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Job description cannot be empty.")
        return v

    @validator("resumes")
    def resumes_must_be_list(cls, v):
        if v is not None and not isinstance(v, list):
            raise ValueError("Resumes must be a list of strings.")
        return v

    @validator("messages")
    def messages_must_be_list(cls, v):
        if v is not None and not isinstance(v, list):
            raise ValueError("Messages must be a list of dictionaries.")
        return v

    @validator("embeddings")
    def embeddings_must_be_list(cls, v):
        if v is not None and not isinstance(v, list):
            raise ValueError("Embeddings must be a list of lists.")
        return v

    @validator("pinecone_ids")
    def pinecone_ids_must_be_list(cls, v):
        if v is not None and not isinstance(v, list):
            raise ValueError("Pinecone IDs must be a list of strings.")
        return v

    @validator("gap_analysis_results")
    def gap_analysis_results_must_be_dict(cls, v):
        if v is not None and not isinstance(v, dict):
            raise ValueError("Gap analysis results must be a dictionary.")
        return v

    @validator("optimization_recommendations")
    def optimization_recommendations_must_be_list(cls, v):
        if v is not None and not isinstance(v, list):
            raise ValueError("Optimization recommendations must be a list of strings.")
        return v


        