from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool

from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()


class StudentDatabase(BaseModel):
    student: str = Field(..., description="Student name")


class StudentData(BaseTool):
    name: str = "student_data"
    description: str = "A tool to get student data from the database."

    def _run(self, input: str) -> str:
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.7
        )

        prompt = ChatPromptTemplate.from_template(
            """
            You are a database assistant. Use the content below as your only source of truth:
            {db}

            Question: {question}
            """
        )

        chain = prompt | llm | StrOutputParser()
        return chain.invoke({
            "db": input,
            "question": "What is the student's name?"
        })


if __name__ == "__main__":
    student_data = StudentData()
    student = StudentDatabase(student="John Doe")
    result = student_data._run(student.model_dump_json())
    print(result)
