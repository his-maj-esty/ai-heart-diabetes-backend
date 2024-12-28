from typing import List, Optional, TypedDict, Literal, Annotated, Sequence, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage, AIMessage
from langgraph.graph.message import add_messages

class Report(TypedDict):
  name: str
  url: str

class GraphState(TypedDict, total=False):
  chosen_report: str
  reports: List[Report]
  report_names: List[str]
  disease: str
  conversation: Annotated[Sequence[BaseMessage], add_messages]
  extracted_report_data: Any
  has_chosen_report: bool
  query: str
  chat_id: str
  user: str

class DiseaseSchema(BaseModel):
  disease: Literal['heart', 'diabetes']

class HeartDiseaseSchema(BaseModel):
    age: int = Field(..., description="The patient's age in years")
    sex: str = Field(..., description="The patient's sex, either Male or Female")
    cp: str = Field(..., description="Type of chest pain: Typical Angina, Atypical Angina, Non-anginal Pain, or Asymptomatic")
    trestbps: int = Field(..., description="Resting blood pressure in mmHg")
    chol: int = Field(..., description="Serum cholesterol in mg/dl")
    fbs: bool = Field(..., description="Fasting blood sugar > 120 mg/dl (True or False)")
    restecg: str = Field(..., description="Resting ECG results: Normal, ST-T wave abnormality, or Left ventricular hypertrophy")
    thalach: int = Field(..., description="Maximum heart rate achieved in bpm")
    exang: bool = Field(..., description="Exercise induced angina (True or False)")
    oldpeak: float = Field(..., description="ST depression induced by exercise relative to rest")
    slope: str = Field(..., description="Slope of the peak exercise ST segment: Upsloping, Flat, or Downsloping")
    thal: str = Field(..., description="Thalassemia status: Normal, Fixed defect, or Reversible defect")

class YesNoSchema(BaseModel):
    chosen: Literal["yes", "no", "other"]
