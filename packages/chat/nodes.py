from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain_core.messages import  AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import END
import boto3
import os
import json
from .custom_types import GraphState, DiseaseSchema, HeartDiseaseSchema, YesNoSchema
from .tools import get_reports_tool
from .helpers import  get_chosen_report_url, call_model, get_chat_memory


UPSTASH_URL = os.environ.get("UPSTASH_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
AWS_REGION = os.environ.get("AWS_REGION")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-exp")


textract_client = boto3.client(
    'textract',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


def disease_classifier(state: GraphState) -> GraphState:
  template = """You are an expert in diagnosing diseases based on conversation analysis. Carefully analyze the conversation for any symptoms mentioned. If any symptom matches a specific disease, classify the situation into that disease category. Always prioritize classification based on symptoms provided.

You must classify the situation into one of the following categories:
1. heart: If any symptom indicates a heart-related condition (e.g., chest pain, shortness of breath, irregular heartbeat, etc.).
2. diabetes: If any symptom indicates a diabetes-related condition (e.g., frequent urination, excessive thirst, fatigue, etc.).


Do not classify into 'Other' if any symptoms mentioned are indicative of the listed diseases, even if they are uncommon or not explicitly provided in the examples.
Conversation: {conversation}


  """
 
  prompt = PromptTemplate(
      template=template,
  )
  llm_chain = prompt | llm.with_structured_output(DiseaseSchema)
  result = llm_chain.invoke({"conversation": state["conversation"]})
  chat_memory = get_chat_memory(state.get("user")+":"+state.get("chat_id"))
  chat_memory.add_user_message(state["query"])
  disease = result.disease
  return {"disease": disease.strip()}

def disease_router(state: GraphState):
  if state["disease"] == "heart":
    return "heart"
  elif state["disease"] == "diabetes":
    return "diabetes"
  elif state["disease"] == "no disease":
    return "no_disease"
  else:
    return END


def heart_expert(state: GraphState) -> GraphState:
  template = """ 
  You are an expert in analyzing heart related reports. You have to understand what types of problems does the patient is experiencing.
  if yes, get the reports of the heart by calling the tools provided to you or by returning the name of report by analyzing conversation. else, request the user to upload the reports.
  
  Don't let user know that you are calling any tool
  
  Conversation: {conversation}


  user email : {email}"""
 
  prompt = PromptTemplate(template=template)

  llm_with_tools = llm.bind_tools(tools=[get_reports_tool])
  llm_chain = prompt | llm_with_tools
  chat_memory = get_chat_memory(state.get("user")+":"+state.get("chat_id"))
  result = llm_chain.invoke({"conversation": chat_memory.messages, "email": state["user"]})
  if(result.tool_calls):
    for tool_call in result.tool_calls:
      tool_msg = get_reports_tool.invoke(tool_call)

      print("tool_msg")
      tool_msg_content = json.loads(tool_msg.content)
      report_names = [report["name"] for report in tool_msg_content]
      if(len(report_names) == 0):
        print("report names empty")
        return {"conversation" : [result], "reports": [], "report_names": []}
      else:
        print("report names present")
        return {"conversation" : [result, tool_msg], "report_names": [report_names], "reports": [tool_msg_content]}
  else:
    chat_memory.add_ai_message(result.content)
    return {"conversation" : [result], "reports": [], "report_names": []}

def report_router(state: GraphState) -> GraphState:
  if len(state["reports"]) == 0:
    print("No reports present. Ending...")
    return "end"
  else:
    print("Reports present.")
    return "continue"


def choose_reports(state: GraphState) -> GraphState:

  template = """
    Give user options to choose from the available reports: {reports}. 

    Previous conversation with the user : {conversation}
  """

  prompt = PromptTemplate(
      template=template,
  )
 
  llm_chain = prompt | llm
  result = llm_chain.invoke({"reports": state["reports"], "conversation": state["conversation"]})
  chat_memory = get_chat_memory(state.get("user")+":"+state.get("chat_id"))
  chat_memory.add_ai_message(message=result)
  return {"conversation" : [result]}


def human_input():
  pass

def has_user_chosen_report(state: GraphState) -> GraphState:
    template = """
    Carefully analyze the conversation between ai and user and determine if ai asked to choose report and then user has replied with any report name.
    Respond with:
    1. yes : if the user has respond a report and result of the report is not told
    2. no :  if the user hasn't chosen report and result of the report is not told.
    3. other:  If the result of reports is told to the user, respond with "other". Carefully analyze other response,
    Use the following format for output:
    {{
        "chosen": "yes", "no" or "other"
    }}
    Conversation: {conversation}
    """

    prompt = PromptTemplate(
        template=template,
    )
    llm_chain = prompt | llm.with_structured_output(YesNoSchema)
    chat_memory = get_chat_memory(state.get("user") + ":" + state.get("chat_id"))    
    messages = [message.content for message in chat_memory.messages]


    result = llm_chain.invoke({"conversation": messages})
    
    
    print("Has user chosen a report? :", result.chosen)
    
    return {"has_chosen_report": result.chosen.strip() , "conversation": chat_memory.messages}

def base_router(state: GraphState) -> str:
  if(state["has_chosen_report"] == "yes"):
    print("moving to redefine_reports_node")
    return "redefine_reports_node"
  elif(state["has_chosen_report"] == "other"):
    print("moving to simple_conversation_node")
    return "other"
  else:
    if(state["disease"] == "heart"):
      print("moving to heart_expert_node")
      return "heart"
    if(state["disease"] == "diabetes"):
      print("moving to diabetes_expert_node")
      return "diabetes"
    else:
      return END


def redefine_reports(state: GraphState) -> GraphState:
    reports = get_reports_tool.invoke({
      "email": state["user"]
    })
    report_names = [report["name"] for report in reports]
    template = """
    The user has chosen the report. Your task is to analyze the conversation and determine the most relevant report name from the following options: {report_names}.
    Conversation: {conversation}

    Important: Provide your answer as a single word from the given options only. Do not include any other text, symbols, or newlines.
"""


    prompt = PromptTemplate.from_template(template)

    llm_chain = prompt | llm
    result = llm_chain.invoke({"report_names": report_names, "conversation": state["conversation"]})
    chosen_report = result.content.strip()  

    return {"chosen_report": chosen_report, "reports": reports, "report_names": report_names}



def analyze_report(state: GraphState) -> GraphState:
  template = """
   You are an expert in analyzing reports and patient conversations. You have to extract maximum information from the user conversation and report provided. Use the following json format for your output:
\{{
    age: int,
    sex: str,
    cp: str, Type of chest pain: Typical Angina, Atypical Angina, Non-anginal Pain, or Asymptomatic,
    trestbps: int, Resting blood pressure in mmHg,
    chol: int, Serum cholesterol in mg/dl,
    fbs: bool, Fasting blood sugar > 120 mg/dl (True or False),
    restecg: str, Resting ECG results: Normal, ST-T wave abnormality, or Left ventricular hypertrophy,
    thalach: int, Maximum heart rate achieved in bpm,
    exang: bool, Exercise induced angina (True or False),
    oldpeak: float, ST depression induced by exercise relative to rest,
    slope: str, Slope of the peak exercise ST segment: Upsloping, Flat, or Downsloping,
    thal: str, Thalassemia status: Normal, Fixed defect, or Reversible defect A
\}}

conversation : {conversation}
report : {report}
  """
  reports = state["reports"]
  chosen_report_url = [report["url"] for report in reports if report["name"] == state["chosen_report"]][0]
  print("chosen_report_url", chosen_report_url)

  report = AmazonTextractPDFLoader(chosen_report_url, client=textract_client).load()
  print("report", report)
  report_content = [doc.page_content for doc in report]
  prompt = PromptTemplate(
      template=template,
  )

  llm_chain = prompt | llm.with_structured_output(HeartDiseaseSchema)

  result = llm_chain.invoke({"report": report_content, "conversation": state["conversation"]})
  ai_message = AIMessage(content=str(result))
  state["conversation"].append(ai_message)
  return {"conversation" : [ai_message], "extracted_report_data": result}



def deciding_expert(state: GraphState) -> GraphState:
  is_disease_present = call_model(state["extracted_report_data"])
  template = """
  You are an expert {disease} reports analyzer. You have read the reports of the user. the disease : {disease} is {is_disease_present} according to the reports
   Don't let the user know that you are a assistant. Clearly mention the disease and the result of the report. Answer in 2-3 lines only.


   Conversation : {conversation} 
  """
  prompt = PromptTemplate(
      template=template,
  )

  llm_chain = prompt | llm
  result = llm_chain.invoke({"disease": state["disease"], "is_disease_present": is_disease_present, "conversation": state["conversation"]})
  chat_memory = get_chat_memory(state.get("user")+":"+state.get("chat_id"))
  chat_memory.add_ai_message(result)
  return {"conversation" : [result]}

def simple_conversation(state: GraphState):
  template = """
  You are a report analyzing expert. Your task is to respond to the user queries. 
  Advise the patient to seek medical attention, wherever necessary. 
   Query : {query}

  Previous Conversation : {conversation}

  """
  prompt = PromptTemplate(template=template)

  llm_chain = prompt | llm
  result = llm_chain.invoke({"conversation": state["conversation"], "query": state["query"]})
  return {"conversation" : [result]}











