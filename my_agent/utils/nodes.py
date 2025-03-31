from functools import lru_cache
import concurrent.futures
from typing import Literal, List, Dict, Any, Optional
from dotenv import load_dotenv
import os
load_dotenv()  # Load environment variables from .env file
from my_agent.utils.state import AgentState
import PyPDF2
from pydantic import BaseModel, Field, validator
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search.tool import TavilySearchResults

# ------------------ Resume Feedback Schema ------------------
class ResumeFeedback(BaseModel):
    improvement_needed: Literal['yes', 'no'] = Field(
        description="Determine if the optimized resume needs further improvement. Return 'yes' or 'no'."
    )
    feedback: str = Field(
        description="Provide constructive feedback if improvements are needed."
    )

parser = PydanticOutputParser(pydantic_object=ResumeFeedback)

# ------------------ Tavily Input Model ------------------
class TavilyInput(BaseModel):
    company_name: Optional[str] = Field("", description="Name of the company")
    job_role: Optional[str] = Field("", description="Job role to search for")

# ------------------ Helper Functions ------------------
@lru_cache(maxsize=32)
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a given PDF file and caches the result."""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return ""
    except PyPDF2.errors.PdfReadError:
        print(f"Error: Could not read PDF file at {pdf_path}")
        return ""

# ------------------ Model Initialization ------------------
@lru_cache(maxsize=4)
def _get_model(model_name: str) -> ChatOpenAI:
    """Initializes and caches the model."""
    api_key = os.getenv("OPENAI_API_KEY")  # Get API key from environment
    print("API Key:", api_key)  # Debug: print API key
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Please set it in your environment.")
    if model_name == "openai":
        return ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=api_key)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

def check_gpt_model(state: AgentState) -> AgentState:
    """Checks if the GPT model is working by sending a simple prompt and logging the response."""
    model = _get_model("openai")
    test_message = "Hello, CPT model. I am GPT model. I am here to assist you."
    response = model.invoke([HumanMessage(content=f"Repeat exactly: '{test_message}'")])
    print("GPT model health check response:", response.content)
    state.gpt_health = "OK" if test_message in response.content else "Error"
    return state

# ------------------ Node Functions ------------------

def extract_job_description(state: AgentState) -> AgentState:
    """Extracts structured data from a job description.
    The prompt instructs the LLM to output JSON: {'company': <company>, 'role': <role>, 'skills': <list_of_skills>}.
    """
    model = _get_model("openai")
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an AI assistant specialized in analyzing job descriptions. Extract the following information in JSON format: {\"company\": \"\", \"role\": \"\", \"skills\": []}. If a field is not present, use an empty string."),
        HumanMessage(content=f"Analyze this job description: {state.job_description}")
    ])
    try:
        formatted_prompt = prompt.format_messages(job_description=state.job_description)
        response = model.invoke(formatted_prompt)
        extracted_data = response.content
        
        # Sanitize response: remove markdown formatting (backticks, leading 'json')
        if extracted_data.startswith("```"):
            extracted_data = extracted_data.strip("`").strip()
            if extracted_data.lower().startswith("json"):
                extracted_data = extracted_data[4:].strip()
                
        import json
        parsed_data = json.loads(extracted_data)
        new_state_dict = state.model_dump()
        new_state_dict.update({
            "extracted_job_description": extracted_data,
            "company_name": parsed_data.get("company", ""),
            "job_role": parsed_data.get("role", ""),
            # Optionally, store skills if desired.
        })
        return AgentState(**new_state_dict)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON response from LLM: {extracted_data}")
        new_state_dict = state.model_dump()
        new_state_dict["error"] = f"Invalid JSON response from LLM: {extracted_data}"
        return AgentState(**new_state_dict)
    except Exception as e:
        print(f"Error extracting job description: {e}")
        new_state_dict = state.model_dump()
        new_state_dict["error"] = f"Extract job description failed: {e}"
        return AgentState(**new_state_dict)

def check_company_industry(state: AgentState) -> AgentState:
    """Extracts the company industry from the job description and updates state."""
    model = _get_model("openai")
    # Use the extracted job description if available, else use the raw job description.
    input_text = state.extracted_job_description or state.job_description
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an AI assistant specialized in extracting industry-related information from job descriptions."),
        HumanMessage(content=f"Extract the company industry from this text: {input_text}")
    ])
    formatted_prompt = prompt.format_messages(job_description=input_text)
    response = model.invoke(formatted_prompt)
    s = state.model_dump()
    s["industry"] = response.content.strip()
    return AgentState(**s)

def extract_job_details(state: AgentState) -> AgentState:
    """Extracts the job role and company name from the job description.
    The expected response format is 'Role: <role> Company: <company>'.
    """
    model = _get_model("openai")
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an AI assistant specialized in extracting job details. Return the result in the exact format: 'Role: <role> Company: <company>'"),
        HumanMessage(content=f"Extract the job role and company name from this job description: {state.job_description}")
    ])
    formatted_prompt = prompt.format_messages(job_description=state.job_description)
    response = model.invoke(formatted_prompt)
    try:
        parts = response.content.split("Company:")
        job_role = parts[0].replace("Role:", "").strip()
        company_name = parts[1].strip()
    except Exception:
        job_role = ""
        company_name = ""
    s = state.model_dump()
    s["job_role"] = job_role
    s["company_name"] = company_name
    return AgentState(**s)

def prompt_for_missing_info(state: AgentState) -> AgentState:
    """Ensures that 'job_role' and 'company_name' are not empty. If they are, default values are applied."""
    job_role = state.job_role.strip() if state.job_role else ""
    company_name = state.company_name.strip() if state.company_name else ""
    s = state.model_dump()
    s["job_role"] = job_role if job_role else "Default Role"
    s["company_name"] = company_name if company_name else "Default Company"
    return AgentState(**s)

def process_job(state: AgentState) -> AgentState:
    """Processes the job description by extracting job details and industry.
    This node only performs extractions; missing details are handled downstream.
    """
    state = extract_job_description(state)
    print("Job Description:", state.extracted_job_description)
    state = check_company_industry(state)
    print("Industry:", state.industry)
    state = extract_job_details(state)
    print("Job Details:", f"Role: {state.job_role}, Company: {state.company_name}")
    return state

def extract_resumes(state: AgentState) -> AgentState:
    """Finds and extracts resumes from PDF files using parallel processing."""
    resume_folder = "pdfs/"
    resume_files = [os.path.join(resume_folder, file) for file in os.listdir(resume_folder) if file.endswith(".pdf")]
    extracted_resumes = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(extract_text_from_pdf, resume_files))
    for resume_text in results:
        model = _get_model("openai")
        msg = model.invoke([HumanMessage(content=f"Extract Name, Skills, Experience, Education from this resume: {resume_text}")])
        extracted_resumes.append(msg.content)
    s = state.model_dump()
    s["resumes"] = extracted_resumes
    return AgentState(**s)

def evaluate_optimization(state: AgentState) -> AgentState:
    """Evaluates the optimized resume using a custom output parser."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an AI assistant specialized in evaluating resumes and providing feedback."),
        HumanMessage(content=f"Review the optimized resume and provide feedback: {state.optimized_resume or ''}"),
        AIMessage(content=parser.get_format_instructions())
    ])
    model = _get_model("openai")
    output = model.invoke(prompt)
    parsed_output = parser.parse(output.content)
    s = state.model_dump()
    s["improvement_needed"] = parsed_output.improvement_needed
    s["feedback"] = parsed_output.feedback
    return AgentState(**s)

def route_optimization(state: AgentState) -> str:
    """Returns the name of the next node based on the ATS evaluation."""
    improvement_needed = state.improvement_needed or "no"
    return "apply_targeted_improvements" if improvement_needed == "yes" else "generate_cover_letter_and_cold_email"

def generate_cover_letter(state: AgentState) -> AgentState:
    """Generates a cover letter based on the first resume and job description."""
    model = _get_model("openai")
    response = model.invoke([HumanMessage(content=f"Write a cover letter for {state.resumes[0] if state.resumes else ''} using the job description: {state.job_description}")])
    s = state.model_dump()
    s["cover_letter"] = response.content
    return AgentState(**s)

def optimize_resume(state: AgentState) -> AgentState:
    """Refines resume bullet points for better alignment with job description skills."""
    model = _get_model("openai")
    response = model.invoke([HumanMessage(content=f"Refine resume bullets for better alignment with job description skills: {state.resumes[0] if state.resumes else ''}")])
    s = state.model_dump()
    s["optimized_resume"] = response.content
    return AgentState(**s)

def evaluate_ats_score(state: AgentState) -> AgentState:
    """Computes ATS scores for the original and optimized resumes."""
    model = _get_model("openai")
    msg_before = model.invoke([HumanMessage(content=f"Evaluate ATS score (out of 100) for this resume. Return only a number: {state.resumes[0] if state.resumes else ''}")])
    msg_after = model.invoke([HumanMessage(content=f"Evaluate ATS score (out of 100) for this optimized resume. Return only a number: {state.optimized_resume or ''}")])
    s = state.model_dump()
    s["ats_score_before"] = float(msg_before.content)
    s["ats_score_after"] = float(msg_after.content)
    return AgentState(**s)

def convert_resume_to_embeddings(state: AgentState) -> AgentState:
    """Splits each resume into overlapping chunks and simulates converting them into embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    resumes = state.resumes or []
    all_embeddings = []
    all_chunks = []
    for resume in resumes:
        chunks = text_splitter.split_text(resume)
        all_chunks.append(chunks)
        for chunk in chunks:
            # Dummy conversion: create a vector of length 768 using the chunk's length.
            embedding = [float(len(chunk))] * 768
            all_embeddings.append(embedding)
    s = state.model_dump()
    s["resume_chunks"] = all_chunks
    s["embeddings"] = all_embeddings
    return AgentState(**s)

def store_in_pinecone(state: AgentState) -> AgentState:
    """Stores the generated embeddings in Pinecone using the gRPC API."""
    from pinecone.grpc import PineconeGRPC as Pinecone
    from pinecone import ServerlessSpec
    api_key = os.getenv("PINECODE_API_KEY")
    if not api_key:
        raise ValueError("PINECODE_API_KEY is not set in the environment.")
    pc = Pinecone(api_key=api_key)
    index_name = "dense-index"
    dimension = 768
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(index_name)
    embeddings = state.embeddings or []
    vectors = []
    for i, emb in enumerate(embeddings):
        # Dummy conversion: create a vector of length 768 using the embedding's length.
        vector = [float(len(emb))] * dimension
        vectors.append((f"id_{i}", vector, {}))
    try:
        result = index.upsert(vectors=vectors, namespace="default")
    except Exception as e:
        print(f"Error during upsert: {e}")
        raise
    s = state.model_dump()
    s["pinecone_ids"] = [f"id_{i}" for i in range(len(embeddings))]
    return AgentState(**s)

@tool(description="Uses Tavily to retrieve additional company and role information.")
def tavily_search_company_role(state: AgentState) -> AgentState:
    """Uses the Tavily library to search for and retrieve additional company and role information."""
    input_data = TavilyInput(
        company_name=state.company_name or "",
        job_role=state.job_role or ""
    )
    search_tool = TavilySearchResults(max_results=5, include_answer=True, include_raw_content=True, include_images=True)
    query = f"Company: {input_data.company_name}, Role: {input_data.job_role} - provide key insights on company culture, industry trends, and role expectations."
    results = search_tool.run(query)
    s = state.model_dump()
    s["company_role_info"] = results
    return AgentState(**s)

def semantic_matching(state: AgentState) -> AgentState:
    """Assigns a simulated semantic matching score between resume embeddings and the job description."""
    s = state.model_dump()
    s["semantic_match_score"] = 0.85
    return AgentState(**s)

def gap_analysis(state: AgentState) -> AgentState:
    """Performs a simulated gap analysis between resume skills and job description requirements."""
    s = state.model_dump()
    s["gap_analysis_results"] = {"summary": "Identified gaps in required skills."}
    return AgentState(**s)

def generate_optimization_recommendations(state: AgentState) -> AgentState:
    """Generates optimization recommendations based on gap analysis results."""
    model = _get_model("openai")
    prompt = f"Generate optimization recommendations based on the following gap analysis: {(state.gap_analysis_results.get('summary', '') if state.gap_analysis_results else '')}. Provide clear, actionable suggestions to improve the resume."
    response = model.invoke(prompt)
    s = state.model_dump()
    s["optimization_recommendations"] = [response.content]
    return AgentState(**s)

def generate_optimized_resume_draft(state: AgentState) -> AgentState:
    """Generates an optimized resume draft using the optimization recommendations."""
    model = _get_model("openai")
    recs = ", ".join(state.optimization_recommendations or [])
    prompt = f"Generate an optimized resume draft using these recommendations: {recs}. Emphasize improvements in clarity and relevance to the job description."
    response = model.invoke(prompt)
    s = state.model_dump()
    s["optimized_resume"] = response.content
    return AgentState(**s)

def apply_targeted_improvements(state: AgentState) -> AgentState:
    """Applies targeted improvements to the optimized resume if needed."""
    model = _get_model("openai")
    prompt = f"Apply targeted improvements to this resume: {state.optimized_resume or ''}. Enhance clarity, formatting, and relevance to the job description."
    response = model.invoke(prompt)
    s = state.model_dump()
    s["optimized_resume"] = response.content
    return AgentState(**s)

def generate_cover_letter_and_cold_email(state: AgentState) -> AgentState:
    """Generates a cover letter and cold email based on company research and job details."""
    model = _get_model("openai")
    prompt = [HumanMessage(content=f"Generate a cover letter and a cold email for company {state.company_name or ''} and role {state.job_role or ''} using available company information: {state.company_role_info or ''}. The cover letter should be professional and tailored, and the cold email should be concise and engaging. Use '||' as a delimiter between the two.")]
    response = model.invoke(prompt)
    parts = response.content.split("||")
    s = state.model_dump()
    s["cover_letter"] = parts[0].strip() if parts else ""
    s["cold_email"] = parts[1].strip() if len(parts) > 1 else ""
    return AgentState(**s)

def user_feedback_final_review(state: AgentState) -> AgentState:
    """Simulates collecting user feedback and final review of the documents."""
    s = state.model_dump()
    s["user_feedback"] = "User approved the documents."
    return AgentState(**s)

def export_final_documents(state: AgentState) -> AgentState:
    """Exports the final documents by combining cover letter, cold email, and optimized resume."""
    s = state.model_dump()
    s["exported_documents"] = f"Cover Letter: {s.get('cover_letter', '')}\nCold Email: {s.get('cold_email', '')}\nOptimized Resume: {s.get('optimized_resume', '')}"
    return AgentState(**s)