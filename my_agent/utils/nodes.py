from functools import lru_cache
import concurrent.futures
import logging
from typing import Literal, List, Dict, Any, Optional
from dotenv import load_dotenv
import os
from my_agent.utils.state import AgentState
from my_agent.utils.model_manager import get_model
import PyPDF2
import json
from pydantic import BaseModel, Field, validator
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
load_dotenv()  # Load environment variables from .env file


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
model = get_model("openai")

def check_gpt_model(state: AgentState) -> AgentState:
    """Checks if the GPT model is working by sending a simple prompt and logging the response."""
    if not model:
        print("Error: GPT model is not initialized.")
        state.gpt_health = "Error"
        return state
    test_message = "Hello, CPT model. I am GPT model. I am here to assist you."
    response = model.invoke([HumanMessage(content=f"Repeat exactly: '{test_message}'")])
    print("GPT model health check response:", response.content)
    state.gpt_health = "OK" if test_message in response.content else "Error"
    return state

# ------------------ Node Functions ------------------


def extract_job_data(state: AgentState) -> Dict[str, Any]:
    """
    Extracts comprehensive structured data from a job description, including company, role,
    skills, industry, requirements, responsibilities, benefits, location, salary, and
    company information. Handles missing data by providing default values.
    """
    if not model:
        print("Error: GPT model is not initialized.")
        state.gpt_health = "Error"
        return state
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
        You are a highly skilled AI assistant specializing in analyzing job descriptions. Your task is to extract key information from the provided job description and structure it in JSON format.
 
        Follow these guidelines meticulously:
 
        -   **Output Format:** Return the extracted information in strict JSON format:
            ```json
            {
                "company": "",
                "role": "",
                "job_id": "",
                "posting_date": "",
                "employment_type": "",
                "application_links": [],
                "job_highlights": [],
                "skills": [],
                "industry": "",
                "requirements": "",
                "qualifications": [],
                "preferred_qualifications": [],
                "responsibilities": [],
                "benefits": [],
                "salary_range": {"min": "", "max": ""},
                "location": "",
                "remote": "",
                "about_company": ""
            }
            ```
 
        -   **Field Definitions:**
            -   **company:** The name of the company offering the job. If not explicitly stated, use an empty string.
            -   **role:** The exact title of the job role. If not explicitly stated, use an empty string.
            -   **job_id:** A unique identifier for the job posting. If not provided, use an empty string.
            -   **posting_date:** The date when the job was posted. If not provided, use an empty string.
            -   **employment_type:** The type of employment (e.g., Full-time, Part-time). If not provided, use an empty string.
            -   **application_links:** A list of URLs where the job can be applied for. If none, use an empty list.
            -   **job_highlights:** A list of key highlights or unique selling points of the job. If none, use an empty list.
            -   **skills:** A list of technical and soft skills required for the job. If none, use an empty list.
            -   **industry:** The industry in which the company operates. If not able to search, use an empty string.
            -   **requirements:** A summary of the mandatory qualifications and experience needed. If not provided, use an empty string.
            -   **qualifications:** A list of required qualifications. If none, use an empty list.
            -   **preferred_qualifications:** A list of preferred qualifications. If none, use an empty list.
            -   **responsibilities:** A list of main duties and tasks associated with the role. If none, use an empty list.
            -   **benefits:** A list of benefits offered by the company. If none, use an empty list.
            -   **salary_range:** An object with "min" and "max" and "bumped_up" keys representing the salary range and if the salary coulde be negiciated and bumped up. If not provided, use empty strings.
            -   **location:** The location of the job. If not provided, use an empty string.
            -   **remote:** Indicates if the job is remote. If not provided, use an empty string then say "Not Sure".
            -   **about_company:** A brief description of the company. If not provided, use an empty string.
 
        -   **Data Integrity:**
            -   If a field is not present in the job description, use an empty string or an empty list as appropriate.
            -   Do not invent information; only extract what is explicitly stated or clearly implied.
            -   Be concise and avoid unnecessary details.
            -   Only extract information that is directly related to the job description.
            -   Do not include any extra text or explanation outside of the JSON format.
 
        -   **Formatting:**
            -   Ensure the output is valid JSON.
            -   Remove any markdown formatting (e.g., backticks) from the final output.
        """),
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

        # Provide default values if data is missing
        job_data = {
            "extracted_job_description": extracted_data,
            "company_name": parsed_data.get("company", "Default Company"),
            "job_role": parsed_data.get("role", "Default Role"),
            "job_id": parsed_data.get("job_id", ""),
            "posting_date": parsed_data.get("posting_date", ""),
            "employment_type": parsed_data.get("employment_type", ""),
            "application_links": parsed_data.get("application_links", []),
            "job_highlights": parsed_data.get("job_highlights", []),
            "skills": parsed_data.get("skills", []),
            "industry": parsed_data.get("industry", ""),
            "requirements": parsed_data.get("requirements", ""),
            "qualifications": parsed_data.get("qualifications", []),
            "preferred_qualifications": parsed_data.get("preferred_qualifications", []),
            "responsibilities": ", ".join(parsed_data.get("responsibilities", [])), # Convert to string
            "benefits": ", ".join(parsed_data.get("benefits", [])), # Convert to string
            "salary_range": parsed_data.get("salary_range", {"min": "", "max": "", "bumped_up": ""}),
            "location": parsed_data.get("location", ""),
            "remote": parsed_data.get("remote", ""),
            "about_company": parsed_data.get("about_company", "")
        }
        return job_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from LLM: {extracted_data}. Error: {e}")
    except Exception as e:
        raise RuntimeError(f"Error extracting job description: {e}")


def process_job(state: AgentState) -> AgentState:
    """
    Processes the job description by extracting job details and industry.
    This node now stores both the original and extracted job descriptions.
    """
    if not model:
        logging.error("Error: GPT model is not initialized.")
        state.gpt_health = "Error"
        state.error = "Error: GPT model is not initialized."
        return state
    try:
        job_data = extract_job_data(state)
        new_state_dict = state.model_dump()
        new_state_dict.update(job_data)
        new_state_dict["original_job_description"] = state.job_description

        # Prepare debug information
        debug_info = {
            "Original Job Description": state.job_description,
            "Extracted Job Description": job_data["extracted_job_description"],
            "Job ID": job_data.get("job_id", ""),
            "Posting Date": job_data.get("posting_date", ""),
            "Employment Type": job_data.get("employment_type", ""),
            "Application Links": job_data.get("application_links", []),
            "Job Highlights": job_data.get("job_highlights", []),
            "Industry": job_data.get("industry", ""),
            "Job Details": f"Role: {job_data['job_role']}, Company: {job_data['company_name']}",
            "Requirements": job_data.get("requirements", ""),
            "Qualifications": job_data.get("qualifications", []),
            "Preferred Qualifications": job_data.get("preferred_qualifications", []),
            "Responsibilities": job_data.get("responsibilities", []),
            "Benefits": job_data.get("benefits", []),
            "Salary Range": job_data.get("salary_range", {"min": "", "max": ""}),
            "Location": job_data.get("location", ""),
            "Remote": job_data.get("remote", ""),
            "About Company": job_data.get("about_company", "")
        }
        
        # Log the debug information
        for key, value in debug_info.items():
            logging.info(f"{key}: {value}")

        return AgentState(**new_state_dict)
    except (ValueError, RuntimeError) as e:
        logging.error(f"Error processing job: {e}")
        new_state_dict = state.model_dump()
        new_state_dict["error"] = f"{type(e).__name__}: {e}"
        new_state_dict["gpt_health"] = "Error"
        return AgentState(**new_state_dict)

# Helper function to extract structured data from a single PDF
def extract_structured_data_from_pdf(pdf_path: str, model) -> Dict[str, Any]:
    """
    Extracts structured data (name, skills, experience, education) from a PDF resume.

    Args:
        pdf_path: The path to the PDF file.
        model: The LLM model to use for extraction.

    Returns:
        A dictionary containing the extracted data, or None if an error occurred.
    """
    try:
        resume_text = extract_text_from_pdf(pdf_path)
        if not resume_text.strip():
            logging.warning(f"Resume at {pdf_path} is empty.")
            return None

        prompt = f"""
        You are a highly skilled AI assistant specializing in analyzing resumes. 
        Your task is to extract key information from the provided resume and structure it in JSON format.
        Extract the following information from the resume:
        - Name
        - Skills (list of skills)
        - Experience
        - Education
        Return the result in JSON format: {{"name": "", "skills": [], "experience": "", "education": ""}}
        """
        msg = model.invoke([HumanMessage(content=f"{prompt} Resume: {resume_text}")])
        
        # Sanitize response: remove markdown formatting (backticks, leading 'json')
        extracted_data = msg.content
        if extracted_data.startswith("```"):
            extracted_data = extracted_data.strip("`").strip()
            if extracted_data.lower().startswith("json"):
                extracted_data = extracted_data[4:].strip()
        
        return json.loads(extracted_data)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON response from LLM for resume at {pdf_path}: {msg.content}. Error: {e}")
        return None
    except Exception as e:
        logging.error(f"Error extracting data from resume at {pdf_path}: {e}")
        return None


def extract_resumes(state: AgentState, resume_folder: str = "pdfs/") -> AgentState:
    """
    Finds and extracts structured data from resumes in PDF files using parallel processing.

    Args:
        state: The current AgentState.
        resume_folder: The folder containing the PDF resumes.

    Returns:
        The updated AgentState.
    """
    model = get_model()
    if not model:
        logging.error("Error: GPT model is not initialized.")
        state.gpt_health = "Error"
        state.error = "Error: GPT model is not initialized."
        return state

    resume_files = [os.path.join(resume_folder, file) for file in os.listdir(resume_folder) if file.endswith(".pdf")]
    extracted_resumes: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_pdf = {executor.submit(extract_structured_data_from_pdf, pdf_file, model): pdf_file for pdf_file in resume_files}
        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf_file = future_to_pdf[future]
            try:
                data = future.result()
                if data:
                    extracted_resumes.append(data)
            except Exception as e:
                logging.error(f"Error processing resume at {pdf_file}: {e}")
                state.error = f"Error processing resume at {pdf_file}: {e}"

    new_state_dict = state.model_dump()
    new_state_dict["resumes"] = extracted_resumes
    return AgentState(**new_state_dict)


def convert_resume_to_embeddings(state: AgentState) -> AgentState:
    """
    Splits each resume into overlapping chunks and simulates converting them into embeddings.
    Now uses the structured resume data (name, skills, experience, education) for better embeddings.
    """
    if not state.resumes:
        logging.warning("No resumes found to convert to embeddings.")
        new_state_dict = state.model_dump()
        new_state_dict["resume_chunks"] = []  # Initialize as empty list
        new_state_dict["embeddings"] = []  # Initialize as empty list
        return AgentState(**new_state_dict)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_embeddings: List[List[float]] = []
    all_chunks: List[List[str]] = []

    for resume_data in state.resumes:
        if not isinstance(resume_data, dict):
            logging.error(f"Invalid resume data format: {resume_data}")
            state.error = f"Invalid resume data format: {resume_data}"
            continue

        # Combine structured data into a single text for embedding
        resume_text_parts = []
        if "name" in resume_data and resume_data["name"]:
            resume_text_parts.append(f"Name: {resume_data['name']}")
        if "skills" in resume_data and resume_data["skills"]:
            resume_text_parts.append(f"Skills: {', '.join(resume_data['skills'])}")
        if "experience" in resume_data and resume_data["experience"]:
            resume_text_parts.append(f"Experience: {resume_data['experience']}")
        if "education" in resume_data and resume_data["education"]:
            resume_text_parts.append(f"Education: {resume_data['education']}")

        resume_text = " ".join(resume_text_parts)

        if not resume_text.strip():
            logging.warning(f"Resume data is empty after combining structured data: {resume_data}")
            continue

        chunks = text_splitter.split_text(resume_text)
        all_chunks.append(chunks)
        for chunk in chunks:
            # Dummy conversion: create a vector of length 768 using the chunk's length.
            embedding = [float(len(chunk))] * 768
            all_embeddings.append(embedding)

    new_state_dict = state.model_dump()
    new_state_dict["resume_chunks"] = all_chunks
    new_state_dict["embeddings"] = all_embeddings
    return AgentState(**new_state_dict)



def store_in_pinecone(state: AgentState, index_name: str = "dense-index", dimension: int = 768) -> AgentState:
    """
    Stores the generated embeddings in Pinecone using the gRPC API.

    Args:
        state: The current AgentState.
        index_name: The name of the Pinecone index.
        dimension: The dimension of the embeddings.

    Returns:
        The updated AgentState.
    """
    api_key = os.getenv("PINECODE_API_KEY")
    if not api_key:
        raise ValueError("PINECODE_API_KEY is not set in the environment.")

    pc = Pinecone(api_key=api_key)

    if index_name not in pc.list_indexes().names():
        logging.info(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(index_name)
    embeddings = state.embeddings or []
    vectors: List[tuple[str, List[float], dict]] = []

    for i, emb in enumerate(embeddings):
        if not isinstance(emb, list) or len(emb) != dimension:
            logging.error(f"Invalid embedding format at index {i}: {emb}")
            state.error = f"Invalid embedding format at index {i}: {emb}"
            continue
        vectors.append((f"id_{i}", emb, {}))

    try:
        logging.info(f"Upserting {len(vectors)} vectors to Pinecone index: {index_name}")
        result = index.upsert(vectors=vectors, namespace="default")
        logging.info(f"Upsert result: {result}")
    except Exception as e:
        logging.error(f"Error during upsert to Pinecone index {index_name}: {e}")
        state.error = f"Error during upsert to Pinecone index {index_name}: {e}"
        raise

    new_state_dict = state.model_dump()
    new_state_dict["pinecone_ids"] = [f"id_{i}" for i in range(len(vectors))]
    return AgentState(**new_state_dict)


def semantic_matching(state: AgentState) -> AgentState:
    """
    Calculates a semantic matching score between the resume and the job description using the LLM.

    Args:
        state: The current AgentState.

    Returns:
        The updated AgentState with the semantic_match_score.
    """
    model = get_model()
    if not model:
        logging.error("Error: GPT model is not initialized.")
        state.gpt_health = "Error"
        state.error = "Error: GPT model is not initialized."
        return state

    if not state.resumes:
        logging.warning("No resumes found for semantic matching.")
        state.semantic_match_score = 0.0
        return state
    
    if not state.job_description:
        logging.warning("No job description found for semantic matching.")
        state.semantic_match_score = 0.0
        return state

    try:
        resume_text = ""
        if isinstance(state.resumes[0], dict):
            resume_text_parts = []
            resume_data = state.resumes[0]
            if "name" in resume_data and resume_data["name"]:
                resume_text_parts.append(f"Name: {resume_data['name']}")
            if "skills" in resume_data and resume_data["skills"]:
                resume_text_parts.append(f"Skills: {', '.join(resume_data['skills'])}")
            if "experience" in resume_data and resume_data["experience"]:
                resume_text_parts.append(f"Experience: {resume_data['experience']}")
            if "education" in resume_data and resume_data["education"]:
                resume_text_parts.append(f"Education: {resume_data['education']}")
            resume_text = " ".join(resume_text_parts)
        else:
            resume_text = state.resumes[0]

        prompt = f"""
        You are a highly skilled AI assistant specializing in evaluating the semantic similarity between resumes and job descriptions.
        Your task is to assess how well the provided resume matches the job description.
        Consider the skills, experience, and overall content of the resume in relation to the requirements and responsibilities outlined in the job description.
        Provide a semantic matching score between 0 and 1, where 0 indicates no match and 1 indicates a perfect match.
        Return only a number between 0 and 1.
        Resume: {resume_text}
        Job Description: {state.job_description}
        """
        response = model.invoke([HumanMessage(content=prompt)])
        semantic_match_score = float(response.content)
        if semantic_match_score < 0 or semantic_match_score > 1:
            raise ValueError("Semantic match score is not between 0 and 1")
        new_state_dict = state.model_dump()
        new_state_dict["semantic_match_score"] = semantic_match_score
        return AgentState(**new_state_dict)
    except ValueError as e:
        logging.error(f"Error in semantic matching: {e}")
        state.error = f"Error in semantic matching: {e}"
        state.semantic_match_score = 0.0
        return state
    except Exception as e:
        logging.error(f"Error in semantic matching: {e}")
        state.error = f"Error in semantic matching: {e}"
        state.semantic_match_score = 0.0
        return state


def gap_analysis(state: AgentState) -> AgentState:
    """
    Performs a gap analysis between resume data and job description details.
    Identifies missing skills and suggests areas for improvement.
    """
    model = get_model()
    if not model:
        logging.error("Error: GPT model is not initialized.")
        state.gpt_health = "Error"
        state.error = "Error: GPT model is not initialized."
        return state

    if not state.resumes:
        logging.warning("No resumes found for gap analysis.")
        state.gap_analysis_results = {"summary": "No resumes provided."}
        return state

    if not state.job_description:
        logging.warning("No job description found for gap analysis.")
        state.gap_analysis_results = {"summary": "No job description provided."}
        return state

    try:
        resume_skills = ""
        resume_data = state.resumes[0]
        if isinstance(resume_data, dict):
            if "skills" in resume_data and resume_data["skills"]:
                resume_skills = ", ".join(resume_data["skills"])
        else:
            try:
                resume_skills_response = model.invoke([HumanMessage(content=f"Extract skills from this resume: {state.resumes[0]}")])
                resume_skills = resume_skills_response.content
            except Exception as e:
                logging.error(f"Error extracting skills from resume: {e}")
                state.error = f"Error extracting skills from resume: {e}"
                state.gap_analysis_results = {"summary": f"Error extracting skills from resume: {e}"}
                return state

        prompt = f"""
        You are a highly skilled AI assistant specializing in performing gap analysis between resumes and job descriptions.
        Your task is to analyze the provided resume data and job description details to identify gaps in skills and experience.
        Consider the job requirements, responsibilities, and the candidate's skills and experience.

        Job Description Details:
        - Company: {state.company_name or 'Not specified'}
        - Role: {state.job_role or 'Not specified'}
        - Industry: {state.industry or 'Not specified'}
        - Requirements: {state.requirements or 'Not specified'}
        - Responsibilities: {state.responsibilities or 'Not specified'}

        Resume Data:
        - Skills: {resume_skills or 'Not specified'}

        Analyze the job description details and the resume data to identify gaps in skills and experience.
        Provide a detailed summary of the gaps and suggest specific areas for improvement in the resume.
        Focus on how the candidate can better align their experience and skills with the job requirements and responsibilities.
        """
        response = model.invoke([HumanMessage(content=prompt)])

        new_state_dict = state.model_dump()
        new_state_dict["gap_analysis_results"] = {"summary": response.content}
        return AgentState(**new_state_dict)
    except Exception as e:
        logging.error(f"Error in gap analysis: {e}")
        state.error = f"Error in gap analysis: {e}"
        state.gap_analysis_results = {"summary": f"Error in gap analysis: {e}"}
        return state


def generate_optimization_recommendations(state: AgentState) -> AgentState:
    """
    Generates detailed optimization recommendations for the resume based on gap analysis,
    job description details, and the provided instructions.
    """
    model = get_model()
    if not model:
        logging.error("Error: GPT model is not initialized.")
        state.gpt_health = "Error"
        state.error = "Error: GPT model is not initialized."
        return state

    if not state.gap_analysis_results or "summary" not in state.gap_analysis_results:
        logging.warning("No gap analysis results found. Cannot generate optimization recommendations.")
        state.optimization_recommendations = ["No gap analysis results available."]
        return state

    if not state.resumes:
        logging.warning("No resumes found for optimization recommendations.")
        state.optimization_recommendations = ["No resumes provided."]
        return state

    gap_summary = state.gap_analysis_results["summary"]
    job_requirements = state.requirements or "Not specified"
    job_responsibilities = state.responsibilities or "Not specified"
    job_skills = state.skills or "Not specified"
    resume_text = ""
    if isinstance(state.resumes[0], dict):
        resume_text_parts = []
        resume_data = state.resumes[0]
        if "name" in resume_data and resume_data["name"]:
            resume_text_parts.append(f"Name: {resume_data['name']}")
        if "skills" in resume_data and resume_data["skills"]:
            resume_text_parts.append(f"Skills: {', '.join(resume_data['skills'])}")
        if "experience" in resume_data and resume_data["experience"]:
            resume_text_parts.append(f"Experience: {resume_data['experience']}")
        if "education" in resume_data and resume_data["education"]:
            resume_text_parts.append(f"Education: {resume_data['education']}")
        resume_text = " ".join(resume_text_parts)
    else:
        resume_text = state.resumes[0]

    prompt = f"""
    You are a world-class resume optimization expert. Your task is to provide detailed, actionable recommendations to improve a resume based on a job description and a gap analysis.

    Here's the information you have:
    - Job Description Details:
        - Company: {state.company_name or 'Not specified'}
        - Role: {state.job_role or 'Not specified'}
        - Industry: {state.industry or 'Not specified'}
        - Requirements: {job_requirements}
        - Responsibilities: {job_responsibilities}
        - MUST-HAVE Skills: {job_skills}
    - Resume: {resume_text}
    - Gap Analysis Summary: {gap_summary}

    Follow these instructions meticulously:

    1.  Identify the top 3-5 MUST-HAVE skills from the job description.
    2.  When listing your accomplishments, try to use this framing: "Accomplished [X] as measured by [Y] by doing [Z]" X = The accomplishment or result, Y = The specific metric or measurement, Z = The actions you took to achieve it. Take this as a reference and use synonyms rather than the exact same words in the resume.
    3.  For each bullet point on my resume, pinpoint the specific skill it demonstrates. If a bullet point isn't relevant to the job, please let me know.
    4.  For each bullet point on my base resume, keep the information retained and add the relevant keywords while keeping it small and short (under 2 lines per bullet point).
    5.  If my resume doesn't strongly reflect a MUST-HAVE skill, suggest specific rewrites to my bullet points to better showcase it. Use keywords and phrases from the job description.
    6.  Keep it authentic to the original (Don't change the meaning too much) and Just try to add the keywords (bold them were added) and fit them in the resume (making it relevant to the JD). At the end, tell which keywords you already fit and which keywords are left out.
    7.  Tailor the given resume to the job you're applying to.
    8.  Give the ATS Official scores out of 10 before and after the changes.
    9.  Provide a prioritized list of 2-3 key areas where I should focus my editing efforts for maximum impact.
    10. Include industry-standard terminology for this role where appropriate.
    11. Provide specific feedback on my education section and if it needs adjustments to better align with the job requirements.

    Provide your recommendations in a clear, well-organized format.
    """
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        new_state_dict = state.model_dump()
        new_state_dict["optimization_recommendations"] = [response.content]
        return AgentState(**new_state_dict)
    except Exception as e:
        logging.error(f"Error generating optimization recommendations: {e}")
        state.error = f"Error generating optimization recommendations: {e}"
        state.optimization_recommendations = [f"Error generating optimization recommendations: {e}"]
        return state


def generate_optimized_resume_draft(state: AgentState) -> AgentState:
    """
    Generates an optimized resume draft using the optimization recommendations.
    """
    model = get_model()
    if not model:
        logging.error("Error: GPT model is not initialized.")
        state.gpt_health = "Error"
        state.error = "Error: GPT model is not initialized."
        return state

    if not state.optimization_recommendations:
        logging.warning("No optimization recommendations found. Cannot generate optimized resume draft.")
        state.optimized_resume = "No optimization recommendations available."
        return state
    
    if not state.resumes:
        logging.warning("No resumes found for optimization.")
        state.optimized_resume = "No resumes provided."
        return state

    recs = ", ".join(state.optimization_recommendations)
    resume_text = ""
    if isinstance(state.resumes[0], dict):
        resume_text_parts = []
        resume_data = state.resumes[0]
        if "name" in resume_data and resume_data["name"]:
            resume_text_parts.append(f"Name: {resume_data['name']}")
        if "skills" in resume_data and resume_data["skills"]:
            resume_text_parts.append(f"Skills: {', '.join(resume_data['skills'])}")
        if "experience" in resume_data and resume_data["experience"]:
            resume_text_parts.append(f"Experience: {resume_data['experience']}")
        if "education" in resume_data and resume_data["education"]:
            resume_text_parts.append(f"Education: {resume_data['education']}")
        resume_text = " ".join(resume_text_parts)
    else:
        resume_text = state.resumes[0]

    prompt = f"""
    You are a world-class resume writer. Your task is to generate an optimized resume draft based on the provided resume and the following optimization recommendations.

    Here's the information you have:
    - Resume: {resume_text}
    - Optimization Recommendations: {recs}
    - Job Description Details:
        - Company: {state.company_name or 'Not specified'}
        - Role: {state.job_role or 'Not specified'}
        - Industry: {state.industry or 'Not specified'}
        - Requirements: {state.requirements or 'Not specified'}
        - Responsibilities: {state.responsibilities or 'Not specified'}

    Follow these instructions meticulously:
    1.  Incorporate the provided optimization recommendations into the resume.
    2.  Emphasize improvements in clarity, impact, and relevance to the job description.
    3.  Use keywords and phrases from the job description where appropriate.
    4.  Maintain the original meaning and authenticity of the resume content.
    5.  Ensure the optimized resume is well-formatted and easy to read.
    6.  Focus on making the resume ATS-friendly.
    7.  Use industry-standard terminology for this role where appropriate.
    8.  Do not include any extra text or explanation outside of the resume.

    Provide the optimized resume draft.
    """
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        new_state_dict = state.model_dump()
        new_state_dict["optimized_resume"] = response.content
        return AgentState(**new_state_dict)
    except Exception as e:
        logging.error(f"Error generating optimized resume draft: {e}")
        state.error = f"Error generating optimized resume draft: {e}"
        state.optimized_resume = f"Error generating optimized resume draft: {e}"
        return state


def evaluate_ats_score(state: AgentState) -> AgentState:
    """
    Computes ATS scores for the original and optimized resumes using the LLM.
    """
    model = get_model()
    if not model:
        logging.error("Error: GPT model is not initialized.")
        state.gpt_health = "Error"
        state.error = "Error: GPT model is not initialized."
        return state

    if not state.resumes:
        logging.warning("No resumes found for ATS evaluation.")
        state.ats_score_before = 0.0
        state.ats_score_after = 0.0
        return state

    if not state.optimized_resume:
        logging.warning("No optimized resume found for ATS evaluation.")
        state.ats_score_before = 0.0
        state.ats_score_after = 0.0
        return state

    try:
        resume_text = ""
        if isinstance(state.resumes[0], dict):
            resume_text_parts = []
            resume_data = state.resumes[0]
            if "name" in resume_data and resume_data["name"]:
                resume_text_parts.append(f"Name: {resume_data['name']}")
            if "skills" in resume_data and resume_data["skills"]:
                resume_text_parts.append(f"Skills: {', '.join(resume_data['skills'])}")
            if "experience" in resume_data and resume_data["experience"]:
                resume_text_parts.append(f"Experience: {resume_data['experience']}")
            if "education" in resume_data and resume_data["education"]:
                resume_text_parts.append(f"Education: {resume_data['education']}")
            resume_text = " ".join(resume_text_parts)
        else:
            resume_text = state.resumes[0]
        prompt_before = f"""
        You are a highly skilled AI assistant specializing in evaluating resumes for Applicant Tracking System (ATS) compatibility.
        Your task is to assess the ATS score of the following resume.
        Provide an ATS score between 0 and 100, where 0 indicates no match and 100 indicates a perfect match.
        Return only a number between 0 and 100.
        Resume: {resume_text}
        Job Description: {state.job_description}
        """
        msg_before = model.invoke([HumanMessage(content=prompt_before)])
        ats_score_before = float(msg_before.content)
        if ats_score_before < 0 or ats_score_before > 100:
            raise ValueError("ATS score before is not between 0 and 100")

        prompt_after = f"""
        You are a highly skilled AI assistant specializing in evaluating resumes for Applicant Tracking System (ATS) compatibility.
        Your task is to assess the ATS score of the following optimized resume.
        Provide an ATS score between 0 and 100, where 0 indicates no match and 100 indicates a perfect match.
        Return only a number between 0 and 100.
        Optimized Resume: {state.optimized_resume}
        Job Description: {state.job_description}
        """
        msg_after = model.invoke([HumanMessage(content=prompt_after)])
        ats_score_after = float(msg_after.content)
        if ats_score_after < 0 or ats_score_after > 100:
            raise ValueError("ATS score after is not between 0 and 100")

        new_state_dict = state.model_dump()
        new_state_dict["ats_score_before"] = ats_score_before
        new_state_dict["ats_score_after"] = ats_score_after
        return AgentState(**new_state_dict)
    except ValueError as e:
        logging.error(f"Error evaluating ATS score: {e}")
        state.error = f"Error evaluating ATS score: {e}"
        state.ats_score_before = 0.0
        state.ats_score_after = 0.0
        return state
    except Exception as e:
        logging.error(f"Error evaluating ATS score: {e}")
        state.error = f"Error evaluating ATS score: {e}"
        state.ats_score_before = 0.0
        state.ats_score_after = 0.0
        return state


def apply_targeted_improvements(state: AgentState) -> AgentState:
    """
    Applies targeted improvements to the optimized resume if needed.
    """
    model = get_model()
    if not model:
        logging.error("Error: GPT model is not initialized.")
        state.gpt_health = "Error"
        state.error = "Error: GPT model is not initialized."
        return state

    if not state.optimized_resume:
        logging.warning("No optimized resume found for targeted improvements.")
        state.optimized_resume = "No optimized resume available."
        return state

    if not state.optimization_recommendations:
        logging.warning("No optimization recommendations found for targeted improvements.")
        state.optimized_resume = "No optimization recommendations available."
        return state
    
    job_requirements = state.requirements or "Not specified"
    job_responsibilities = state.responsibilities or "Not specified"
    job_skills = state.skills or "Not specified"

    prompt = f"""
    You are a world-class resume refinement expert. Your task is to apply targeted improvements to an already optimized resume, making it even more compelling and aligned with the job description.

    Here's the information you have:
    - Optimized Resume: {state.optimized_resume}
    - Optimization Recommendations: {", ".join(state.optimization_recommendations)}
    - Job Description Details:
        - Company: {state.company_name or 'Not specified'}
        - Role: {state.job_role or 'Not specified'}
        - Industry: {state.industry or 'Not specified'}
        - Requirements: {job_requirements}
        - Responsibilities: {job_responsibilities}
        - MUST-HAVE Skills: {job_skills}

    Follow these instructions meticulously:

    1.  Carefully review the optimized resume and the provided optimization recommendations.
    2.  Enhance the clarity, impact, and relevance of the resume to the job description.
    3.  Ensure that the resume effectively showcases the candidate's skills and experience in relation to the job requirements and responsibilities.
    4.  Use keywords and phrases from the job description where appropriate.
    5.  Maintain the original meaning and authenticity of the resume content.
    6.  Ensure the improved resume is well-formatted and easy to read.
    7.  Focus on making the resume ATS-friendly.
    8.  Use industry-standard terminology for this role where appropriate.
    9. Do not include any extra text or explanation outside of the resume.

    Provide the improved resume.
    """
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        new_state_dict = state.model_dump()
        new_state_dict["optimized_resume"] = response.content
        return AgentState(**new_state_dict)
    except Exception as e:
        logging.error(f"Error applying targeted improvements: {e}")
        state.error = f"Error applying targeted improvements: {e}"
        state.optimized_resume = f"Error applying targeted improvements: {e}"
        return state


def generate_cover_letter_and_cold_email(state: AgentState) -> AgentState:
    """
    Generates a cover letter and cold email based on company research, job details,
    and extracted information.
    """
    model = get_model()
    if not model:
        logging.error("Error: GPT model is not initialized.")
        state.gpt_health = "Error"
        state.error = "Error: GPT model is not initialized."
        return state

    if not state.resumes:
        logging.warning("No resumes found. Cannot generate cover letter and cold email.")
        state.cover_letter = "No resume available."
        state.cold_email = "No resume available."
        return state

    try:
        resume_text = ""
        if isinstance(state.resumes[0], dict):
            resume_text_parts = []
            resume_data = state.resumes[0]
            if "name" in resume_data and resume_data["name"]:
                resume_text_parts.append(f"Name: {resume_data['name']}")
            if "skills" in resume_data and resume_data["skills"]:
                resume_text_parts.append(f"Skills: {', '.join(resume_data['skills'])}")
            if "experience" in resume_data and resume_data["experience"]:
                resume_text_parts.append(f"Experience: {resume_data['experience']}")
            if "education" in resume_data and resume_data["education"]:
                resume_text_parts.append(f"Education: {resume_data['education']}")
            resume_text = " ".join(resume_text_parts)
        else:
            resume_text = state.resumes[0]

        prompt = f"""
        You are a world-class expert in crafting compelling cover letters and cold emails. Your task is to generate a tailored cover letter and a concise, engaging cold email for a job application.

        Here's the information you have:
        - Company: {state.company_name or 'Not specified'}
        - Role: {state.job_role or 'Not specified'}
        - About Company: {state.about_company or 'Not specified'}
        - Job Requirements: {state.requirements or 'Not specified'}
        - Job Responsibilities: {state.responsibilities or 'Not specified'}
        - Job Benefits: {state.benefits or 'Not specified'}
        - Job Location: {state.location or 'Not specified'}
        - Job Salary: {state.salary or 'Not specified'}
        - Available company information: {state.company_role_info or 'Not specified'}
        - Resume: {resume_text}

        Instructions:
        1.  **Cover Letter:**
            -   Write a professional and tailored cover letter.
            -   Highlight the candidate's relevant skills and experience.
            -   Express genuine interest in the role and the company.
            -   Connect the candidate's qualifications to the specific job requirements and responsibilities.
            -   Use a formal and engaging tone.
            - Use industry-standard terminology for this role where appropriate.
        2.  **Cold Email:**
            -   Craft a concise and engaging cold email.
            -   Express interest in the role and company.
            -   Briefly highlight the candidate's key qualifications.
            -   Include a call to action (e.g., suggest a brief introductory call).
            -   Maintain a professional yet approachable tone.
            - Use industry-standard terminology for this role where appropriate.
        3. **Delimiter:**
            - Separate the cover letter and the cold email with the delimiter: '||' (two pipe symbols).
        4. **No Extra Text:**
            - Do not include any extra text or explanation outside of the cover letter and cold email.

        Provide the cover letter and cold email, separated by the delimiter.
        """
        response = model.invoke([HumanMessage(content=prompt)])
        parts = response.content.split("||")
        new_state_dict = state.model_dump()
        new_state_dict["cover_letter"] = parts[0].strip() if parts else ""
        new_state_dict["cold_email"] = parts[1].strip() if len(parts) > 1 else ""
        return AgentState(**new_state_dict)
    except Exception as e:
        logging.error(f"Error generating cover letter and cold email: {e}")
        state.error = f"Error generating cover letter and cold email: {e}"
        state.cover_letter = f"Error generating cover letter and cold email: {e}"
        state.cold_email = f"Error generating cover letter and cold email: {e}"
        return state


def user_feedback_final_review(state: AgentState) -> AgentState:
    """
    Collects user feedback on the final documents (cover letter, cold email, optimized resume).
    If no user feedback is provided, it checks if the ATS score improved and suggests
    key skills to add if the score did not improve.
    """
    model = get_model()
    if not model:
        logging.error("Error: GPT model is not initialized.")
        state.gpt_health = "Error"
        state.error = "Error: GPT model is not initialized."
        return state

    if state.user_feedback:
        logging.info(f"User feedback received: {state.user_feedback}")
        new_state_dict = state.model_dump()
        return AgentState(**new_state_dict)

    logging.warning("No user feedback provided. Checking ATS score improvement.")

    if state.ats_score_after is None or state.ats_score_before is None:
        logging.warning("ATS scores are not available. Cannot determine if improvement was made.")
        state.user_feedback = "ATS scores not available."
        new_state_dict = state.model_dump()
        return AgentState(**new_state_dict)

    if state.ats_score_after > state.ats_score_before:
        logging.info("ATS score improved. No further action needed.")
        state.user_feedback = "ATS score improved."
        new_state_dict = state.model_dump()
        return AgentState(**new_state_dict)
    
    if not state.resumes:
        logging.warning("No resumes found for optimization.")
        state.user_feedback = "No resumes provided."
        new_state_dict = state.model_dump()
        return AgentState(**new_state_dict)

    try:
        resume_text = ""
        if isinstance(state.resumes[0], dict):
            resume_text_parts = []
            resume_data = state.resumes[0]
            if "name" in resume_data and resume_data["name"]:
                resume_text_parts.append(f"Name: {resume_data['name']}")
            if "skills" in resume_data and resume_data["skills"]:
                resume_text_parts.append(f"Skills: {', '.join(resume_data['skills'])}")
            if "experience" in resume_data and resume_data["experience"]:
                resume_text_parts.append(f"Experience: {resume_data['experience']}")
            if "education" in resume_data and resume_data["education"]:
                resume_text_parts.append(f"Education: {resume_data['education']}")
            resume_text = " ".join(resume_text_parts)
        else:
            resume_text = state.resumes[0]
        prompt = f"""
        You are a world-class resume optimization expert. The ATS score of the optimized resume did not improve compared to the original resume.
        Your task is to identify the key skills missing from the resume that are present in the job description.
        Provide a list of 3-5 key skills that are missing from the resume.
        Resume: {resume_text}
        Job Description: {state.job_description}
        """
        response = model.invoke([HumanMessage(content=prompt)])
        state.user_feedback = f"ATS score did not improve. Missing key skills: {response.content}"
        logging.info(f"ATS score did not improve. Missing key skills: {response.content}")
        new_state_dict = state.model_dump()
        return AgentState(**new_state_dict)
    except Exception as e:
        logging.error(f"Error suggesting missing skills: {e}")
        state.error = f"Error suggesting missing skills: {e}"
        state.user_feedback = f"Error suggesting missing skills: {e}"
        new_state_dict = state.model_dump()
        return AgentState(**new_state_dict)


def export_final_documents(state: AgentState) -> AgentState:
    """
    Exports the final documents by combining cover letter, cold email, and optimized resume.
    """
    logging.info("Exporting final documents...")
    
    cover_letter = state.cover_letter or "No cover letter generated."
    cold_email = state.cold_email or "No cold email generated."
    optimized_resume = state.optimized_resume or "No optimized resume generated."

    exported_documents = f"""
    ==================================================
    FINAL DOCUMENTS
    ==================================================

    --- Cover Letter ---
    {cover_letter}

    --- Cold Email ---
    {cold_email}

    --- Optimized Resume ---
    {optimized_resume}

    ==================================================
    END OF DOCUMENTS
    ==================================================
    """

    new_state_dict = state.model_dump()
    new_state_dict["exported_documents"] = exported_documents
    logging.info("Final documents exported successfully.")
    return AgentState(**new_state_dict)


def evaluate_optimization(state: AgentState) -> AgentState:
    """
    Evaluates the optimized resume and provides feedback based on ATS scores.

    Args:
        state: The current AgentState.

    Returns:
        The updated AgentState with feedback and improvement_needed flag.
    """
    model = get_model()
    if not model:
        print("Error: GPT model is not initialized.")
        state.gpt_health = "Error"
        return state

    if state.ats_score_before is None or state.ats_score_after is None:
        print("Warning: ATS scores are not available for comparison.")
        state.improvement_needed = "no"
        state.feedback = "ATS scores are not available for comparison."
        return state

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an AI assistant specialized in evaluating resumes and providing feedback."),
        HumanMessage(content=f"""
            Review the optimized resume and provide feedback: {state.optimized_resume or ''}
            Original ATS Score: {state.ats_score_before}
            Optimized ATS Score: {state.ats_score_after}
            Based on the ATS score improvement, determine if the optimized resume needs further improvement.
            If the optimized ATS score is higher than the original ATS score, then no further improvement is needed.
            If the optimized ATS score is lower than or equal to the original ATS score, then further improvement is needed.
        """),
        AIMessage(content=parser.get_format_instructions())
    ])

    formatted_prompt = prompt.format_messages(
        optimized_resume=state.optimized_resume or "",
        ats_score_before=state.ats_score_before,
        ats_score_after=state.ats_score_after
    )

    output = model.invoke(formatted_prompt)
    try:
        parsed_output = parser.parse(output.content)
    except Exception as e:
        print(f"Error parsing output: {e}")
        state.improvement_needed = "no"
        state.feedback = f"Error parsing output: {e}"
        return state

    new_state_dict = state.model_dump()
    new_state_dict["improvement_needed"] = parsed_output.improvement_needed
    new_state_dict["feedback"] = parsed_output.feedback
    return AgentState(**new_state_dict)




def route_optimization(state: AgentState) -> str:
    """Returns the name of the next node based on the ATS evaluation."""
    improvement_needed = getattr(state, "improvement_needed", "no")
    return "apply_targeted_improvements" if improvement_needed == "yes" else "generate_cover_letter_and_cold_email"


def generate_cover_letter(state: AgentState) -> AgentState:
    """Generates a cover letter based on the first resume and job description."""
    if not model:
        print("Error: GPT model is not initialized.")
        state.gpt_health = "Error"
        return state
    response = model.invoke([HumanMessage(content=f"Write a cover letter for {state.resumes[0] if state.resumes else ''} using the job description: {state.job_description}")])
    s = state.model_dump()
    s["cover_letter"] = response.content
    return AgentState(**s)

def optimize_resume(state: AgentState) -> AgentState:
    """Refines resume bullet points for better alignment with job description skills."""
    if not model:
        print("Error: GPT model is not initialized.")
        state.gpt_health = "Error"
        return state
    response = model.invoke([HumanMessage(content=f"Refine resume bullets for better alignment with job description skills: {state.resumes[0] if state.resumes else ''}")])
    s = state.model_dump()
    s["optimized_resume"] = response.content
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

