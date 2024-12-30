from langchain.tools import  tool
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
from typing import Annotated
import os

load_dotenv()

# tools
@tool
def get_reports_tool(email: str = Annotated[str, "email of the patient"] ):
  """tool to get reports of the user"""
  email_stripped = email.strip()
  reports = fetch_reports_by_email(email_stripped)
  print("reports", reports)
  formatted_reports = [
    {"name": report["name"], "url": f"s3://disease-reports/{email_stripped}/{report['storageKey']}.pdf"}
    for report in reports
]
  return formatted_reports
  
@tool
def ask_human_tool():
  """useful to ask human a question"""
  return "report_heart_1.pdf"



def fetch_reports_by_email( email: str ):
    """
    Fetches reports of the user. You can use this tool to fetch health reports of the user.
    """
    try:
        DATABASE_URL=os.environ.get("DATABASE_URL")
        print("DATABASE_URL", DATABASE_URL)

        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor()
        

        query = sql.SQL("""
            SELECT * FROM "Report" WHERE email = %s;
        """)
        cursor.execute(query, (email,))

        rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]

        reports = [dict(zip(column_names, row)) for row in rows]

        return reports

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return None
