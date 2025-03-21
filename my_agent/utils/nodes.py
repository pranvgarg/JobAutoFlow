# nodes.py
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import TypedDict, Literal
import PyPDF2
import os
# ------------------ Resume Feedback Schema ------------------
# ------------------ Evaluator Optimizer Schema ------------------
class ResumeFeedback(BaseModel):
    improvement_needed: Literal["yes", "no"] = Field(
        description="Determine if the optimized resume needs further improvement. Return 'yes' or 'no'."
    )
    feedback: str = Field(
        description="Provide constructive feedback if improvements are needed."
    )

parser = PydanticOutputParser(pydantic_object=ResumeFeedback)


# ------------------ Helper Functions ------------------
def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text


# ------------------ Model Initialization ------------------
@lru_cache(maxsize=4)
def _get_model(model_name: str):
    """Initializes and caches the model."""
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    return model


# ------------------ Node Functions ------------------
def extract_job_description(state):
    """Extracts structured data from a job description."""
    model = _get_model("openai")
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an AI assistant specialized in analyzing job descriptions."),
        HumanMessage(content="Extract Company, Role, and Must-Have Skills from this job description: {job_description}")
    ])
    formatted_prompt = prompt.format_messages(job_description=state["job_description"])
    response = model.invoke(formatted_prompt)
    return {"job_description": response.content}


def extract_resumes(state):
    """Finds and extracts the resume matching the job role."""
    resume_folder = "pdfs/"  # Folder containing resumes
    resume_files = [os.path.join(resume_folder, file) for file in os.listdir(resume_folder) if file.endswith(".pdf")]
    extracted_resumes = []
    
    for file in resume_files:
        resume_text = extract_text_from_pdf(file)
        model = _get_model("openai")
        msg = model.invoke(f"Extract Name, Skills, Experience, Education from this resume: {resume_text}")
        extracted_resumes.append(msg.content)
    
    return {"resumes": extracted_resumes}


def evaluate_optimization(state):
    """Evaluates the optimized resume using a custom output parser."""
    prompt = f"""
    Review the optimized resume and provide feedback:
    {state['optimized_resume']}

    {parser.get_format_instructions()}
    """
    model = _get_model("openai")
    output = model.invoke(prompt)
    parsed_output = parser.parse(output.content)
    return {"improvement_needed": parsed_output.improvement_needed, "feedback": parsed_output.feedback}

# ------------------ Route Based on Evaluator Feedback ------------------
def route_optimization(state):
    """Route back to resume optimization if improvement is needed."""
    if state["improvement_needed"] == "yes":
        return "Rejected + Feedback"
    else:
        return "Accepted"

def generate_cover_letter(state):
    """Generates a personalized cover letter based on the selected resume and JD."""
    model = _get_model("openai")
    response = model.invoke(f"Write a cover letter for {state['resumes'][0]} using the JD {state['job_description']}")
    return {"cover_letter": response.content}


def optimize_resume(state):
    """Refines resume bullet points based on JD skills."""
    model = _get_model("openai")
    response = model.invoke(f"Refine resume bullets for better alignment with JD skills: {state['resumes'][0]}")
    return {"optimized_resume": response.content}


def evaluate_ats_score(state):
    """Computes ATS score before & after resume optimization."""
    model = _get_model("openai")
    
    msg_before = model.invoke(f"Evaluate ATS score (out of 100) for this resume. Return only a number: {state['resumes'][0]}")
    msg_after = model.invoke(f"Evaluate ATS score (out of 100) for this optimized resume. Return only a number: {state['optimized_resume']}")
    
    return {
        "ats_score_before": float(msg_before.content),
        "ats_score_after": float(msg_after.content)
    }


# ------------------ Tool Execution ------------------
tools = []  # Add tools if necessary (e.g., TavilySearchResults)
tool_node = ToolNode(tools)


# ------------------ Routing Function ------------------
def should_continue(state):
    """Determines whether to continue or end the workflow."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are no tool calls, finish; otherwise, continue
    if not last_message.get("tool_calls"):
        return "end"
    else:
        return "continue"


# ------------------ Model Calling Function ------------------
system_prompt = """Be a helpful assistant"""

def call_model(state, config):
    """Calls the LLM with system and user messages."""
    messages = state["messages"]
    
    # Add system prompt to the messages
    messages = [{"role": "system", "content": system_prompt}] + messages
    
    model_name = config.get('configurable', {}).get("model_name", "openai")
    
    # Get the appropriate model
    model = _get_model(model_name)
    
    # Invoke the model and return its response
    response = model.invoke(messages)
    
    # Return as a list to append to existing messages
    return {"messages": [response]}
