from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from my_agent.utils.nodes import (
    process_job,
    extract_resumes,
    convert_resume_to_embeddings,
    store_in_pinecone,
    tavily_search_company_role,
    semantic_matching,
    gap_analysis,
    generate_optimization_recommendations,
    generate_optimized_resume_draft,
    evaluate_ats_score,
    apply_targeted_improvements,
    generate_cover_letter_and_cold_email,
    user_feedback_final_review,
    export_final_documents,
    route_optimization,
    check_gpt_model
)
from langchain.output_parsers import PydanticOutputParser
from my_agent.utils.state import AgentState

# Define the config schema for the workflow
class GraphConfig(TypedDict):
    model_name: Literal["openai"]

# ------------------ Evaluator Optimizer Schema ------------------
class ResumeFeedback(BaseModel):
    improvement_needed: Literal["yes", "no"] = Field(
        description="Determine if the optimized resume needs further improvement. Return 'yes' or 'no'."
    )
    feedback: str = Field(
        description="Provide constructive feedback if improvements are needed."
    )

parser = PydanticOutputParser(pydantic_object=ResumeFeedback)

# Build the workflow graph using AgentState as state type
workflow = StateGraph(AgentState, config_schema=GraphConfig)

workflow.add_node("check_gpt_model", check_gpt_model)
workflow.add_node("process_job", process_job)
workflow.add_node("extract_resumes", extract_resumes)
workflow.add_node("convert_resume_to_embeddings", convert_resume_to_embeddings)
workflow.add_node("store_in_pinecone", store_in_pinecone)
workflow.add_node("tavily_search_company_role", tavily_search_company_role)
workflow.add_node("semantic_matching", semantic_matching)
workflow.add_node("perform_gap_analysis", gap_analysis)
workflow.add_node("generate_optimization_recommendations", generate_optimization_recommendations)
workflow.add_node("generate_optimized_resume_draft", generate_optimized_resume_draft)
workflow.add_node("evaluate_ats_score", evaluate_ats_score)
workflow.add_node("apply_targeted_improvements", apply_targeted_improvements)
workflow.add_node("generate_cover_letter_and_cold_email", generate_cover_letter_and_cold_email)
workflow.add_node("user_feedback_final_review", user_feedback_final_review)
workflow.add_node("export_final_documents", export_final_documents)

# Define edges for the execution flow
workflow.add_edge(START, "check_gpt_model")
workflow.add_edge("check_gpt_model", "process_job")
workflow.add_edge("process_job", "extract_resumes")
workflow.add_edge("extract_resumes", "convert_resume_to_embeddings")
workflow.add_edge("convert_resume_to_embeddings", "store_in_pinecone")
# Optionally, if you want to enrich with Tavily search before semantic matching, uncomment:
# workflow.add_edge("store_in_pinecone", "tavily_search_company_role")
# workflow.add_edge("tavily_search_company_role", "semantic_matching")
workflow.add_edge("store_in_pinecone", "semantic_matching")
workflow.add_edge("semantic_matching", "perform_gap_analysis")
workflow.add_edge("perform_gap_analysis", "generate_optimization_recommendations")
workflow.add_edge("generate_optimization_recommendations", "generate_optimized_resume_draft")
workflow.add_edge("generate_optimized_resume_draft", "evaluate_ats_score")

workflow.add_conditional_edges(
    "evaluate_ats_score",
    route_optimization,
    {
        "apply_targeted_improvements": "apply_targeted_improvements",
        "generate_cover_letter_and_cold_email": "generate_cover_letter_and_cold_email"
    }
)

workflow.add_edge("apply_targeted_improvements", "generate_optimized_resume_draft")
workflow.add_edge("generate_cover_letter_and_cold_email", "user_feedback_final_review")
workflow.add_edge("user_feedback_final_review", "export_final_documents")
workflow.add_edge("export_final_documents", END)

graph = workflow.compile()

# Example usage:
job_description = """
company expedia

Data Governance Engineer III page is loaded
Data Governance Engineer III
Apply
Data Governance Engineer III
Apply
locations
USA - Illinois - Chicago
time type
Full time
posted on
Posted Yesterday
time left to apply
End Date: April 30, 2025 (30+ days left to apply)
job requisition id
R-94611

15%
Resume Match
2 of 13 keywords
Simplify
Simplify
Expedia Group brands power global travel for everyone, everywhere. We design cutting-edge tech to make travel smoother and more memorable, and we create groundbreaking solutions for our partners. Our diverse, vibrant, and welcoming community is essential in driving our success.

Why Join Us?

To shape the future of travel, people must come first. Guided by our Values and Leadership Agreements, we foster an open culture where everyone belongs, differences are celebrated and know that when one of us wins, we all win.

We provide a full benefits package, including exciting travel perks, generous time-off, parental leave, a global hybrid work setup (with some pretty cool offices), and career development resources, all to fuel our employees' passion for travel and ensure a rewarding career journey. We’re building a more open world. Join us.

Introduction to the team

Expedia’s Data Management and Governance (EDMG) organization is an enterprise function chartered with enabling and maintaining metadata management, data quality, and data architecture; facilitating a community of data stewards; and providing oversight and monitoring into effectiveness of our capabilities and compliance to standards and regulations. Our goal is to simplify the data landscape by making data easy to discover while protecting the security and integrity of our most valued data assets. We aim to provide practical and scalable solutions that empower data-driven business, moving the innovation and progress of the enterprise forward.

As the Data Governance Engineer, you will use your strong engineering skills to lead development of capabilities and implementation to enable our Data Governance and Compliance programs. You’ll work to define requirements and business processes as well as engineer and implement controls to manage, govern, and protect our data ensuring we achieve our internal data management and governance standards while complying to various regulations.  This job includes working with our larger EDMG team as well as Compliance, Security, Legal, Privacy, Engineering, and Product teams across the organization.

In this role you will:
Develop frameworks and implement solutions to manage critical data at-scale ensuring regulatory requirements are adhered to and embedded in data processes; critical data might include business critical, PCI, PII, personal data
In partnership with Security and data stewards, develop a process to monitor, identify, and remediate data handling concerns
Work with cross-functional teams to build and integrate Data Governance controls in existing business and technical processes
Develop mechanisms to gain insight into data access and implement processes to manage data access at-scale
Build repeatable processes and systems to support Data Governance compliance reporting, including data protection and data privacy insights
Identify, build, and implement areas of continuous improvement and automation in data governance space
Contribute to a culture of innovation and trust by evangelizing and adopting best practices with well-defined goals and metrics
Foster cross-functional and cross-team collaboration by building trust and relationships with Security, Legal, Privacy, Product and Engineering teams
Experience and qualifications:
Bachelor’s degree in a Computer Science, Computer Engineering, MIS, Data Engineering or a related field
5+ years of professional experience in data security and data governance
A passion for Data Governance combined with strong knowledge of Data Governance principles and best practices
Experience building and implementing technical frameworks for effective data handling, data sharing, and data change management
Experience in tools such as Collibra, Sentra, or OneTrust is beneficial
Experience with data security (including PCI, PII data), as well as CCPA and GDPR privacy laws and regulations
Experience designing, building, and implementing technical solutions to handle data in-line with governance and regulations
Technical experience in cloud and big data technologies, as well as building scripting and automation solutions
Excellent communications skills (verbal and written) and interpersonal skills to effectively communicate with both business and technical teams
The total cash range for this position in Chicago is $128,000.00 to $179,500.00. Employees in this role have the potential to increase their pay up to $205,000.00, which is the top of the range, based on ongoing, demonstrated, and sustained performance in the role.
Starting pay for this role will vary based on multiple factors, including location, available budget, and an individual’s knowledge, skills, and experience.
Accommodation requests:
If you need assistance with any part of the application or recruiting process due to a disability, please reach out to our Recruiting Accommodations Team.
"""

initial_state = AgentState(job_description=job_description, messages=[], gpt_health="Not Checked")
final_state = graph.invoke(initial_state)
print(final_state)