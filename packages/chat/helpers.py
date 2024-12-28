import  os
import random
from langchain_community.chat_message_histories import (
    UpstashRedisChatMessageHistory
)

UPSTASH_URL = os.environ.get("UPSTASH_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_TOKEN")

def get_chat_memory(session_id: str):
  chat_memory = UpstashRedisChatMessageHistory(url=UPSTASH_URL, token=UPSTASH_TOKEN, session_id=session_id)
  return chat_memory


def call_model(extracted_report):
  print("extracted report : ", extracted_report)
  return random.choice(["present", "absent"])

def get_chosen_report_url(chosen_report):
  reports = get_reports()
  for report in reports:
    if report["name"] == chosen_report:
      return report["url"]
  return None

def get_reports():
  """tool to get heart reports of the user"""
  return [{"name": "report 1","url":"s3://disease-reports/complex_heart_disease_report1.pdf"}, {"name" : "report 2", "url": "s3://disease-reports/complex_heart_disease_report2.pdf"}]

