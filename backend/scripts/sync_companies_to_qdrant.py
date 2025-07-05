import os
import sys
import textwrap
from typing import List
from openai import OpenAI
from database import SessionLocal
from models import Company
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Config
COLLECTION_NAME = "companies"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
#CHUNK_SIZE = 300  # number of characters per chunk
BATCH_SIZE = 20

# Clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(host="localhost", port=6333)

def chunk_text(text: str) -> List[str]:
    return [chunk.strip() for chunk in text.strip().split("\n\n") if chunk.strip()]

def embed_batch(texts: List[str]) -> List[List[float]]:
    print(f"Total chunks to embed: {len(texts)}")
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        print(f"Embedding batch of size {len(batch)}")
        print(f"Sample text: {batch[0][:100]}...")
        response = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


def sync_companies():
    with SessionLocal() as db:
        companies = db.query(Company).all()

        if qdrant_client.collection_exists(COLLECTION_NAME):
            qdrant_client.delete_collection(collection_name=COLLECTION_NAME)

        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest.VectorParams(size=EMBED_DIM, distance=rest.Distance.COSINE)
        )

        points = []
        vector_text_pairs = []

        for company in companies:
            text_chunks = []

            # 1. General description
            general_text = company.details or ""
            general_chunks = chunk_text(general_text)
            text_chunks.extend([(chunk, "general") for chunk in general_chunks])

            # 2. Offerings
            for offering in company.offerings:
                text_chunks.append(('ծառայություն ' + offering.name, "offerings"))

            print(f"Company: {company.name} ({company.id}), chunks: {len(text_chunks)}")
            for i, (chunk, _) in enumerate(text_chunks):
                print(f"Chunk[{i}] (len={len(chunk)}): {chunk[:60]}...")

            vector_text_pairs.extend([
                {
                    "company_id": company.id,
                    "text": chunk,
                    "type": chunk_type
                }
                for chunk, chunk_type in text_chunks
            ])

        all_texts = [item["text"] for item in vector_text_pairs]
        all_vectors = embed_batch(all_texts)

        for idx, (item, vector) in enumerate(zip(vector_text_pairs, all_vectors)):
            points.append(rest.PointStruct(
                id=idx,
                vector=vector,
                payload={
                    "company_id": item["company_id"],
                    "text": item["text"],
                    "type": item["type"]
                }
            ))

        if points:
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"Upserted {len(points)} points into Qdrant.")
        else:
            print("No data to upsert.")


if __name__ == "__main__":
    sync_companies()
