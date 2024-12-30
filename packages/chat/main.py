from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START
from .custom_types import GraphState
from .nodes import disease_classifier, simple_conversation, heart_expert, report_router, choose_reports, base_router,analyze_report, deciding_expert, human_input, redefine_reports, has_user_chosen_report
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

workflow = StateGraph(GraphState)



workflow.add_node("disease_classifier_node", disease_classifier)
workflow.add_node("heart_expert_node", heart_expert)
workflow.add_node("choose_reports_node", choose_reports)
workflow.add_node("analyze_report_node", analyze_report)
workflow.add_node("deciding_expert_node", deciding_expert)
workflow.add_node("redefine_reports_node", redefine_reports)
workflow.add_node("has_user_chosen_report_node", has_user_chosen_report)
workflow.add_node("simple_conversation_node", simple_conversation)

workflow.add_edge(START, "disease_classifier_node")
workflow.add_edge("disease_classifier_node", "has_user_chosen_report_node")
workflow.add_conditional_edges("has_user_chosen_report_node", base_router, {
  "redefine_reports_node": "redefine_reports_node",
  "heart": "heart_expert_node",
  "other": "simple_conversation_node",
  # "diabetes": "diabetes_expert_node",

})

workflow.add_conditional_edges("heart_expert_node", report_router, {
    "continue": "choose_reports_node",
    "end": "simple_conversation_node"
})

workflow.add_edge("choose_reports_node", "human_input_node")

workflow.add_edge("redefine_reports_node", "analyze_report_node")
workflow.add_edge("analyze_report_node", "deciding_expert_node")
workflow.add_edge("deciding_expert_node", END)
workflow.add_edge("simple_conversation_node", END)

checkpointer_memory = MemorySaver()

graph = workflow.compile(checkpointer=checkpointer_memory, interrupt_before=["human_input_node"])

# class inputType(BaseModel):
#   user:str
#   query: str
#   chat_id: str

chatbot_chain = graph

image = chatbot_chain.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)

# Save the image to a file
with open("~/Desktop/mermaid_diagram.png", "wb") as file:
    file.write(image)