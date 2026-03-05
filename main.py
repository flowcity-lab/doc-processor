import os, uuid, time, io, logging
from contextlib import asynccontextmanager
from typing import Optional
import httpx, openai
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, BackgroundTasks
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    MatchValue, MatchAny, FilterSelector
)

API_SECRET = os.environ.get("API_SECRET", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "http://qdrant:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
TIKA_HOST = os.environ.get("TIKA_HOST", "http://tika:9998")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
COLLECTION = "notebook_documents"
VECTOR_DIM = 1536
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dps")

jobs = {}
qdrant = None
oai = None

AUDIO_EXTENSIONS = ("mp3", "wav", "m4a", "ogg", "webm", "mp4", "mpeg", "mpga")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant, oai
    qdrant = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY, timeout=60)
    oai = openai.OpenAI(api_key=OPENAI_API_KEY)
    if not qdrant.collection_exists(COLLECTION):
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        qdrant.create_payload_index(COLLECTION, "notebook_id", "keyword")
        qdrant.create_payload_index(COLLECTION, "document_id", "keyword")
    log.info("DPS ready. Qdrant=%s Tika=%s", QDRANT_HOST, TIKA_HOST)
    yield

app = FastAPI(title="Document Processing Service", lifespan=lifespan)

def auth(authorization: str = Header(None)):
    if not authorization or authorization != f"Bearer {API_SECRET}":
        raise HTTPException(401, "Unauthorized")

@app.get("/health")
async def health():
    return {"status": "ok", "active_jobs": len([j for j in jobs.values() if j["status"] == "processing"])}

@app.post("/process")
async def process_doc(
    bg: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: str = Form(...),
    notebook_id: str = Form(...),
    callback_url: str = Form(""),
    authorization: str = Header(None)
):
    auth(authorization)
    job_id = str(uuid.uuid4())
    content = await file.read()
    jobs[job_id] = {
        "job_id": job_id, "status": "queued", "step": "",
        "document_id": document_id, "notebook_id": notebook_id,
        "filename": file.filename, "chunks": 0, "error": None,
        "started_at": time.time()
    }
    bg.add_task(pipeline, job_id, content, file.filename or "", document_id, notebook_id, callback_url)
    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
async def get_status(job_id: str, authorization: str = Header(None)):
    auth(authorization)
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]

@app.post("/search")
async def search(
    query: str = Form(...),
    notebook_id: str = Form(...),
    document_ids: str = Form(""),
    top_k: int = Form(5),
    authorization: str = Header(None)
):
    auth(authorization)
    resp = oai.embeddings.create(model=EMBEDDING_MODEL, input=query)
    qvec = resp.data[0].embedding
    must = [FieldCondition(key="notebook_id", match=MatchValue(value=notebook_id))]
    if document_ids:
        ids = [d.strip() for d in document_ids.split(",") if d.strip()]
        if ids:
            must.append(FieldCondition(key="document_id", match=MatchAny(any=ids)))
    results = qdrant.query_points(
        collection_name=COLLECTION, query=qvec,
        query_filter=Filter(must=must), limit=top_k, with_payload=True
    )
    return {"results": [
        {"text": r.payload.get("text",""), "document_id": r.payload.get("document_id",""),
         "chunk_index": r.payload.get("chunk_index",0), "filename": r.payload.get("filename",""),
         "score": r.score}
        for r in results.points
    ]}

@app.delete("/documents/{document_id}")
async def delete_doc(document_id: str, authorization: str = Header(None)):
    auth(authorization)
    qdrant.delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        )
    )
    return {"status": "deleted", "document_id": document_id}

def pipeline(job_id, content, filename, document_id, notebook_id, callback_url):
    try:
        jobs[job_id]["status"] = "processing"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in AUDIO_EXTENSIONS:
            jobs[job_id]["step"] = "transcribing"
            text = transcribe(content, filename)
        else:
            jobs[job_id]["step"] = "extracting"
            text = extract(content)
        if not text or len(text.strip()) < 10:
            raise Exception(f"No text extracted from {filename}")
        log.info("Job %s: %d chars from %s", job_id, len(text), filename)
        jobs[job_id]["step"] = "chunking"
        chunks = chunk_text(text)
        log.info("Job %s: %d chunks", job_id, len(chunks))
        jobs[job_id]["step"] = "embedding"
        embeddings = embed([c["text"] for c in chunks])
        qdrant.delete(
            collection_name=COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
            )
        )
        jobs[job_id]["step"] = "storing"
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=emb, payload={
                "text": ch["text"], "notebook_id": notebook_id,
                "document_id": document_id, "filename": filename, "chunk_index": i
            })
            for i, (ch, emb) in enumerate(zip(chunks, embeddings))
        ]
        for i in range(0, len(points), 100):
            qdrant.upsert(collection_name=COLLECTION, points=points[i:i+100])
        jobs[job_id]["status"] = "ready"
        jobs[job_id]["step"] = "done"
        jobs[job_id]["chunks"] = len(chunks)
        log.info("Job %s: done, %d chunks stored", job_id, len(chunks))
        if callback_url:
            notify(callback_url, job_id, document_id, "ready", len(chunks))
    except Exception as e:
        log.error("Job %s failed: %s", job_id, str(e))
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        if callback_url:
            notify(callback_url, job_id, document_id, "failed", 0, str(e))

def extract(content):
    with httpx.Client(timeout=120) as c:
        r = c.put(f"{TIKA_HOST}/tika", content=content, headers={"Accept": "text/plain"})
        r.raise_for_status()
        return r.text

def transcribe(content, filename):
    f = io.BytesIO(content)
    f.name = filename
    return oai.audio.transcriptions.create(model="whisper-1", file=f).text

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.strip()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, words = [], []
    for para in paragraphs:
        words.extend(para.split())
        while len(words) >= size:
            chunks.append({"text": " ".join(words[:size])})
            words = words[size - overlap:]
    if words:
        chunks.append({"text": " ".join(words)})
    if not chunks:
        chunks.append({"text": text})
    return chunks

def embed(texts):
    all_emb = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        resp = oai.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_emb.extend([d.embedding for d in resp.data])
    return all_emb

def notify(url, job_id, document_id, status, chunks, error=""):
    try:
        with httpx.Client(timeout=10) as c:
            c.post(url, json={"job_id": job_id, "document_id": document_id,
                "status": status, "chunks": chunks, "error": error},
                headers={"Authorization": f"Bearer {API_SECRET}"})
    except Exception as e:
        log.error("Callback failed: %s", str(e))

