from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph,START, END
from my_agent.utils.nodes import (
        extract_job_description,
        extract_resumes,
        generate_cover_letter,
        optimize_resume,
        evaluate_ats_score,
        evaluate_optimization,
        route_optimization,
        tool_node,
        call_model,
        should_continue
    )
from langchain.output_parsers import PydanticOutputParser
from my_agent.utils.state import AgentState


# Define the config
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

# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)


workflow.add_node("extract_job_description", extract_job_description)
workflow.add_node("extract_resumes", extract_resumes)
workflow.add_node("generate_cover_letter", generate_cover_letter)
workflow.add_node("optimize_resume", optimize_resume)
workflow.add_node("evaluate_ats_score", evaluate_ats_score)
workflow.add_node("evaluate_optimization", evaluate_optimization)
    
workflow.add_node("call_model", call_model)
    
workflow.add_node("tool_execution", tool_node)

    # Define edges for execution flow
workflow.add_edge(START, "extract_job_description")
    
workflow.add_edge("extract_job_description", "extract_resumes")
    
workflow.add_edge("extract_resumes", "generate_cover_letter")
    
workflow.add_edge("extract_resumes", "optimize_resume")
    
workflow.add_edge("optimize_resume", "evaluate_optimization")
    
workflow.add_conditional_edges(
        "evaluate_optimization",
        route_optimization,
        {
            "Accepted": "evaluate_ats_score",
            "Rejected + Feedback": "optimize_resume"
        }
)
workflow.add_edge("evaluate_ats_score", "call_model")
    
workflow.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "continue": "tool_execution",
            "end": END
        }
    )
    
workflow.add_edge("tool_execution", "call_model")

    # return workflow.compile()
# # Define the two nodes we will cycle between
# workflow.add_node("agent", call_model)
# workflow.add_node("action", tool_node)

# # Set the entrypoint as `agent`
# # This means that this node is the first one called
# workflow.set_entry_point("agent")

# # We now add a conditional edge
# workflow.add_conditional_edges(
#     # First, we define the start node. We use `agent`.
#     # This means these are the edges taken after the `agent` node is called.
#     "agent",
#     # Next, we pass in the function that will determine which node is called next.
#     should_continue,
#     # Finally we pass in a mapping.
#     # The keys are strings, and the values are other nodes.
#     # END is a special node marking that the graph should finish.
#     # What will happen is we will call `should_continue`, and then the output of that
#     # will be matched against the keys in this mapping.
#     # Based on which one it matches, that node will then be called.
#     {
#         # If `tools`, then we call the tool node.
#         "continue": "action",
#         # Otherwise we finish.
#         "end": END,
#     },
# )

# # We now add a normal edge from `tools` to `agent`.
# # This means that after `tools` is called, `agent` node is called next.
# workflow.add_edge("action", "agent")

# # Finally, we compile it!
# # This compiles it into a LangChain Runnable,
# # meaning you can use it as you would any other runnable
# meaning you can use it as you would any other runnable
graph = workflow.compile()

# # ------------------ Workflow Definition ------------------
# def create_agent_workflow():
    

#     workflow.add_node("extract_job_description", extract_job_description)
#     workflow.add_node("extract_resumes", extract_resumes)
#     workflow.add_node("generate_cover_letter", generate_cover_letter)
#     workflow.add_node("optimize_resume", optimize_resume)
#     workflow.add_node("evaluate_ats_score", evaluate_ats_score)
#     workflow.add_node("evaluate_optimization", evaluate_optimization)
    
#     workflow.add_node("call_model", call_model)
    
#     workflow.add_node("tool_execution", tool_node)

#     # Define edges for execution flow
#     workflow.add_edge(START, "extract_job_description")
    
#     workflow.add_edge("extract_job_description", "extract_resumes")
    
#     workflow.add_edge("extract_resumes", "generate_cover_letter")
    
#     workflow.add_edge("extract_resumes", "optimize_resume")
    
#     workflow.add_edge("optimize_resume", "evaluate_optimization")
    
#     workflow.add_conditional_edges(
#         "evaluate_optimization",
#         route_optimization,
#         {
#             "Accepted": "evaluate_ats_score",
#             "Rejected + Feedback": "optimize_resume"
#         }
#     )
    
#     workflow.add_edge("evaluate_ats_score", "call_model")
    
#     workflow.add_conditional_edges(
#         "call_model",
#         should_continue,
#         {
#             "continue": "tool_execution",
#             "end": END
#         }
#     )
    
#     workflow.add_edge("tool_execution", "call_model")

#     return workflow.compile()


# # ------------------ Example Usage ------------------
# if __name__ == "__main__":
    
#     initial_state = AgentState(
#         job_description="Analyze this job description for a Business Analyst role.",
#         job_role="BA",
#         resumes=[],
#         cover_letter="",
#         optimized_resume="",
#         ats_score_before=0.0,
#         ats_score_after=0.0,
#         feedback="",
#         improvement_needed="no",
#         messages=[]
#     )

#     compiled_workflow = create_agent_workflow()
    
#     graph = compiled_workflow.invoke(initial_state)

#     print("\n🚀 Final Output:\n")
    
#     print(f"📌 Job Description:\n{graph['job_description']}")
    
#     print("\n📌 Extracted Resumes:")
    
#     for resume in graph["resumes"]:
#         print(resume)
        
