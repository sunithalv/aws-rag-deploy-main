from dataclasses import dataclass
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from src.rag_app.get_chroma_db import get_chroma_db
import boto3
import os

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

#BEDROCK_MODEL_ID = "amazon.titan-text-premier-v1:0"
BEDROCK_MODEL_ID = "meta.llama3-70b-instruct-v1:0"

@dataclass
class QueryResponse:
    query_text: str
    response_text: str
    sources: List[str]


def query_rag(query_text: str) -> QueryResponse:
    db = get_chroma_db()
    #print(f"Total documents in DB: {len(db.get()['ids'])}")
    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    session = boto3.Session(region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"), aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID"),
                            aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY"))

    model = ChatBedrock(model_id=BEDROCK_MODEL_ID, region_name="us-east-1", client=session.client('bedrock-runtime'))
    response = model.invoke(prompt)
    response_text = response.content

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    print(f"Response: {response_text}\nSources: {sources}")

    return QueryResponse(
        query_text=query_text, response_text=response_text, sources=sources
    )


if __name__ == "__main__":
    #query_rag("How much does a landing page cost to develop?")
    #query_rag("What is self attention?")
    query_rag("What is imagenet?")