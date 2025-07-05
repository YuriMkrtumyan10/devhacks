import logging
import os
import re
import asyncio

from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from openai import OpenAI

from qdrant_client import QdrantClient
from qdrant_client.http import models
from database import get_db
from models import Company

router = APIRouter(tags=["assistant"])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI & Qdrant setup
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "companies"
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# 🔹 Embed function
def embed(text: str):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# 🔹 Check if question is about offerings/services
def question_is_about_offerings(question: str) -> bool:
    keywords = ["առաջարկ", "ծառայություն", "սպասարկում"]
    return any(word in question.lower() for word in keywords)

# 🔹 Format offerings
def format_offerings(offerings) -> str:
    if not offerings:
        return "Ընկերությունը չունի մատուցվող ծառայություններ։"
    return "\n".join(f"- {offering.name}" for offering in offerings)

def gpt_answer(details: str, question: str) -> str:
    prompt = f"""
Ընկերության տվյալներ՝
{details}

Պատասխանը պատրաստիր միայն վերոնշյալ տվյալների հիման վրա։

Եթե հարցը վերաբերում է հասցեներին կամ վայրերին, խնդրում եմ նշեք բոլոր հասցեները ճիշտ այնպես, ինչպես նշված են, տող առ տող, առանց կրճատման կամ փոփոխության։

Խնդրում եմ պատասխանել հայերեն լեզվով և պահպանել բնօրինակ լեզուն և ոճը։

Խնդրում եմ պատասխանն ավելի կարճ դարձնել՝ մոտ 300 նիշ (մոտ 2-3 նախադասություն)։

Հարցը՝
{question}
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

# 🔹 Webhook route
@router.post("/webhook")
async def vapi_webhook(request: Request, db: Session = Depends(get_db)):
    data = await request.json()

    tool_call_id = "no-id"
    try:
        message = data.get("message", {})
        tool_calls = message.get("toolCalls", [])
        if tool_calls:
            tool_call = tool_calls[0]
            tool_call_id = tool_call.get("toolCallId") or tool_call.get("id") or "no-id"
    except Exception:
        pass

    user_question = (
        data.get("message", {}).get("text") or
        data.get("question") or
        "Tell me about the company"
    )

    async def process_question(question: str):
        company_id = 1  # Hardcoded for now

        if question_is_about_offerings(question):
            company = db.query(Company).filter(Company.id == company_id).first()
            if not company:
                return "Ընկերությունը չի գտնվել։"
            return format_offerings(company.offerings)

        # Otherwise: fallback to Qdrant + GPT
        question_vector = await asyncio.to_thread(embed, question)

        search_result = await asyncio.to_thread(
            lambda: client.search(
                collection_name=COLLECTION_NAME,
                query_vector=question_vector,
                limit=15,
                with_payload=True,
                query_filter=models.Filter(
                    must=[models.FieldCondition(key="company_id", match=models.MatchValue(value=company_id))]
                )
            )
        )

        if not search_result:
            return "Sorry, I couldn't find any relevant information."

        combined_context = "\n\n".join(hit.payload.get("text", "") for hit in search_result)
        logger.info(f"Combined context length: {len(combined_context)}")
        return combined_context
        return await asyncio.to_thread(gpt_answer, combined_context, question)

    # ⬇️ Final result
    result_text = await process_question(user_question)

    return JSONResponse(content={
        "results": [
            {
                "toolCallId": tool_call_id,
                "result": result_text,
            }
        ]
    })
