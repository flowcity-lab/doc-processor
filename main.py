import os, uuid, time, io, logging, re, asyncio, subprocess, tempfile, json
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import urlparse, urljoin
import httpx, openai
import trafilatura
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, BackgroundTasks, Request
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    MatchValue, MatchAny, FilterSelector, TextIndexParams, TokenizerType,
    PayloadSchemaType
)
from sentence_transformers import CrossEncoder

# Einheitlich mit Laravel: DOC_PROCESSOR_SECRET (Fallback auf API_SECRET für Abwärtskompatibilität)
DOC_PROCESSOR_SECRET = os.environ.get("DOC_PROCESSOR_SECRET", os.environ.get("API_SECRET", ""))
QDRANT_HOST = os.environ.get("QDRANT_HOST", "http://qdrant:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
TIKA_HOST = os.environ.get("TIKA_HOST", "http://tika:9998")

# === Embedding Provider Switch ===
# EMBEDDING_PROVIDER=openai  → OpenAI text-embedding-3-small/large
# EMBEDDING_PROVIDER=google  → Google Gemini Embedding 2 (Vertex AI)
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "1536"))

# OpenAI API Key (für Embeddings bei provider=openai UND für Chat/Contextual Retrieval)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Google Vertex AI (für Embeddings bei provider=google)
GOOGLE_CREDENTIALS_BASE64 = os.environ.get("GOOGLE_CREDENTIALS_BASE64", "")
GOOGLE_PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID", "")
GOOGLE_REGION = os.environ.get("GOOGLE_REGION", "us-central1")

# Chat-Modell für Contextual Retrieval & Research
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4.1-mini")

# Dual-Audio-Embedding: Audiodateien zusätzlich direkt via Gemini embedden (benötigt Google)
DUAL_AUDIO_EMBEDDING = os.environ.get("DUAL_AUDIO_EMBEDDING", "false").lower() == "true"


def get_openai_client():
    """Erstellt OpenAI-Client für Embeddings (bei provider=openai) und Chat."""
    return openai.OpenAI(api_key=OPENAI_API_KEY)


def get_chat_client():
    """Erstellt OpenAI-Client für Chat-Completions (Contextual Retrieval, Research)."""
    return openai.OpenAI(api_key=OPENAI_API_KEY)


def get_google_genai_client():
    """Erstellt Google GenAI-Client für Gemini Embedding 2 (Vertex AI)."""
    import base64, json, tempfile as _tf
    from google import genai

    if GOOGLE_CREDENTIALS_BASE64:
        creds_json = base64.b64decode(GOOGLE_CREDENTIALS_BASE64).decode("utf-8")
        creds_file = _tf.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        creds_file.write(creds_json)
        creds_file.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_file.name

    return genai.Client(
        vertexai=True,
        project=GOOGLE_PROJECT_ID,
        location=GOOGLE_REGION,
    )

COLLECTION = "notebook_documents"
VECTOR_DIM = EMBEDDING_DIMENSIONS
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# === Phase 7: RAG Enhancement Config ===
CONTEXTUAL_RETRIEVAL = os.environ.get("CONTEXTUAL_RETRIEVAL", "true").lower() == "true"
CROSS_ENCODER_MODEL = os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
HYBRID_SEARCH_ENABLED = os.environ.get("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", "20"))  # Anzahl Kandidaten vor Re-Ranking
cross_encoder = None  # wird in lifespan initialisiert

# === Audio Transcription Config ===
TRANSCRIPTION_PROVIDER = os.environ.get("TRANSCRIPTION_PROVIDER", "assemblyai")  # assemblyai | deepgram | whisper
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
ASSEMBLYAI_SPEECH_MODEL = os.environ.get("ASSEMBLYAI_SPEECH_MODEL", "")  # leer = default (universal-3-pro)
ASSEMBLYAI_SPEAKER_LABELS = os.environ.get("ASSEMBLYAI_SPEAKER_LABELS", "true").lower() == "true"
ASSEMBLYAI_SPEAKERS_EXPECTED = int(os.environ.get("ASSEMBLYAI_SPEAKERS_EXPECTED", "0"))  # 0 = auto-detect
ASSEMBLYAI_LANGUAGE_DETECTION = os.environ.get("ASSEMBLYAI_LANGUAGE_DETECTION", "true").lower() == "true"
ASSEMBLYAI_LANGUAGE = os.environ.get("ASSEMBLYAI_LANGUAGE", "")  # leer = auto-detect
ASSEMBLYAI_PROMPT = os.environ.get("ASSEMBLYAI_PROMPT", "")  # freier Steuerungs-Prompt für das Modell
ASSEMBLYAI_KEYTERMS = os.environ.get("ASSEMBLYAI_KEYTERMS", "")  # kommagetrennte Fachbegriffe/Eigennamen
ASSEMBLYAI_TEMPERATURE = float(os.environ.get("ASSEMBLYAI_TEMPERATURE", "0"))  # 0 = deterministisch

# === Deepgram Config ===
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL = os.environ.get("DEEPGRAM_MODEL", "nova-3")  # nova-3 | nova-2 | whisper
DEEPGRAM_LANGUAGE = os.environ.get("DEEPGRAM_LANGUAGE", "")  # leer = auto-detect, z.B. "de"
DEEPGRAM_DIARIZE = os.environ.get("DEEPGRAM_DIARIZE", "true").lower() == "true"
DEEPGRAM_PUNCTUATE = os.environ.get("DEEPGRAM_PUNCTUATE", "true").lower() == "true"
DEEPGRAM_SMART_FORMAT = os.environ.get("DEEPGRAM_SMART_FORMAT", "true").lower() == "true"
DEEPGRAM_KEYTERMS = os.environ.get("DEEPGRAM_KEYTERMS", "")  # kommagetrennte Fachbegriffe/Eigennamen

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dps")

jobs = {}
qdrant = None
oai = None
google_client = None

# TTL für abgeschlossene Jobs: nach 1 Stunde aus dem Speicher entfernen
JOB_TTL_SECONDS = 3600

def cleanup_old_jobs():
    """Entfernt abgeschlossene/fehlgeschlagene Jobs die älter als JOB_TTL_SECONDS sind."""
    now = time.time()
    expired = [
        jid for jid, j in jobs.items()
        if j.get("status") in ("ready", "failed")
        and (now - j.get("started_at", now)) > JOB_TTL_SECONDS
    ]
    for jid in expired:
        del jobs[jid]
    if expired:
        log.info("Cleanup: %d alte Jobs entfernt, %d verbleibend", len(expired), len(jobs))

AUDIO_EXTENSIONS = ("mp3", "wav", "m4a", "ogg", "webm")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant, oai, google_client, cross_encoder
    qdrant = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY, timeout=60)
    oai = get_openai_client()

    # Google GenAI Client initialisieren (nur bei provider=google)
    google_client = None
    if EMBEDDING_PROVIDER == "google":
        try:
            google_client = get_google_genai_client()
            log.info("Google GenAI Client initialisiert (Projekt=%s, Region=%s)", GOOGLE_PROJECT_ID, GOOGLE_REGION)
        except Exception as e:
            log.error("Google GenAI Client konnte nicht erstellt werden: %s", str(e))

    log.info("Embedding Provider: %s, Model: %s, Dimensions: %d", EMBEDDING_PROVIDER, EMBEDDING_MODEL, VECTOR_DIM)

    # Cross-Encoder für Re-Ranking laden
    try:
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        log.info("Cross-Encoder geladen: %s", CROSS_ENCODER_MODEL)
    except Exception as e:
        log.warning("Cross-Encoder konnte nicht geladen werden: %s — Re-Ranking deaktiviert", str(e))
        cross_encoder = None

    # Qdrant Collection: Erstellen oder bei Dimensions-Wechsel neu erstellen
    if qdrant.collection_exists(COLLECTION):
        info = qdrant.get_collection(COLLECTION)
        current_dim = info.config.params.vectors.size
        if current_dim != VECTOR_DIM:
            log.warning("VECTOR_DIM geändert: %d → %d — Collection wird neu erstellt!", current_dim, VECTOR_DIM)
            qdrant.delete_collection(COLLECTION)
            qdrant.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
            )
            qdrant.create_payload_index(COLLECTION, "notebook_id", "keyword")
            qdrant.create_payload_index(COLLECTION, "document_id", "keyword")
            log.info("Collection %s neu erstellt mit dim=%d", COLLECTION, VECTOR_DIM)
    else:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        qdrant.create_payload_index(COLLECTION, "notebook_id", "keyword")
        qdrant.create_payload_index(COLLECTION, "document_id", "keyword")

    # Audio-Collection für Dual-Embedding (separate Flat-Vector Collection)
    if DUAL_AUDIO_EMBEDDING and EMBEDDING_PROVIDER == "google":
        audio_col = COLLECTION + "_audio"
        if not qdrant.collection_exists(audio_col):
            qdrant.create_collection(
                collection_name=audio_col,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
            )
            qdrant.create_payload_index(audio_col, "document_id", "keyword")
            qdrant.create_payload_index(audio_col, "notebook_id", "keyword")
            log.info("Audio-Collection %s erstellt (Dual Embedding)", audio_col)

    # Text-Index für Hybrid Search (BM25-artig) erstellen falls noch nicht vorhanden
    try:
        qdrant.create_payload_index(
            collection_name=COLLECTION,
            field_name="text",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.MULTILINGUAL,
                min_token_len=2,
                max_token_len=20,
                lowercase=True,
            )
        )
        log.info("Text-Index für Hybrid Search erstellt")
    except Exception:
        log.info("Text-Index existiert bereits oder konnte nicht erstellt werden")

    log.info("DPS ready. Qdrant=%s Tika=%s Hybrid=%s Contextual=%s",
             QDRANT_HOST, TIKA_HOST, HYBRID_SEARCH_ENABLED, CONTEXTUAL_RETRIEVAL)

    # Periodischer Cleanup-Task für alte Jobs
    async def periodic_cleanup():
        while True:
            await asyncio.sleep(300)  # alle 5 Minuten
            cleanup_old_jobs()

    cleanup_task = asyncio.create_task(periodic_cleanup())
    yield
    cleanup_task.cancel()

app = FastAPI(title="Document Processing Service", lifespan=lifespan)

def auth(authorization: str = Header(None)):
    if not authorization or authorization != f"Bearer {DOC_PROCESSOR_SECRET}":
        raise HTTPException(401, "Unauthorized")

@app.get("/health")
async def health():
    return {"status": "ok", "active_jobs": len([j for j in jobs.values() if j["status"] == "processing"])}

@app.post("/embed-test")
async def embed_test(authorization: str = Header(None)):
    """Minimaler Embedding-Test: Einen kurzen Text embedden und Vektor-Dimension zurückgeben."""
    auth(authorization)
    try:
        vec = embed_single("Embedding test")
        return {
            "ok": True,
            "dimensions": len(vec),
            "provider": EMBEDDING_PROVIDER,
            "model": EMBEDDING_MODEL,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/embed-texts")
async def embed_texts(request: Request, authorization: str = Header(None)):
    """Embeddet eine Liste kurzer Texte (z.B. Memory-Contents) und gibt Vektoren zurück.
    Body: {"texts": ["...", "..."], "task_type": "RETRIEVAL_DOCUMENT"}
    """
    auth(authorization)
    try:
        body = await request.json()
        texts = body.get("texts", [])
        task_type = body.get("task_type", "RETRIEVAL_DOCUMENT")
        if not isinstance(texts, list) or not texts:
            return {"ok": False, "error": "texts muss ein nicht-leeres Array sein"}
        if len(texts) > 200:
            return {"ok": False, "error": "max 200 Texte pro Request"}
        vecs = embed(texts, task_type=task_type)
        return {
            "ok": True,
            "vectors": vecs,
            "dimensions": len(vecs[0]) if vecs else 0,
            "provider": EMBEDDING_PROVIDER,
            "model": EMBEDDING_MODEL,
        }
    except Exception as e:
        log.error("embed-texts Fehler: %s", str(e))
        return {"ok": False, "error": str(e)}

@app.post("/extract")
async def extract_plain(
    file: UploadFile = File(...),
    authorization: str = Header(None),
):
    """
    Leichter Tika-Only Extraktions-Endpoint für Chat-Datei-Uploads.
    Liefert nur extrahierten Text + Mime zurück – kein Qdrant, kein Chunking.
    """
    auth(authorization)
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file payload")

    try:
        text = extract(content)
    except Exception as e:
        log.exception("Plain extract failed for %s", file.filename)
        raise HTTPException(502, f"Extraction failed: {e}")

    return {
        "text": text or "",
        "mime": file.content_type or "application/octet-stream",
        "filename": file.filename or "",
        "size": len(content),
    }


@app.post("/process")
async def process_doc(
    bg: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: str = Form(...),
    notebook_id: str = Form(...),
    callback_url: str = Form(""),
    pdf_mode: str = Form("basic"),
    chunking_strategy: str = Form("fixed"),
    existing_transcript: str = Form(""),
    # Phase 4: PDF-Bildextraktion
    pdf_figure_min_pixels: int = Form(40000),
    # Phase 3: URL-Bildextraktion (hier unbenutzt, nur fürs einheitliche Schema)
    url_image_extraction_enabled: str = Form("false"),
    url_image_min_pixels: int = Form(10000),
    url_image_max_per_page: int = Form(15),
    authorization: str = Header(None)
):
    auth(authorization)
    job_id = str(uuid.uuid4())
    content = await file.read()
    # existing_transcript ist JSON-kodierter String (oder leer)
    parsed_transcript = None
    if existing_transcript:
        try:
            parsed_transcript = json.loads(existing_transcript)
        except Exception:
            parsed_transcript = None
    strategy = chunking_strategy if chunking_strategy in ("fixed", "semantic") else "fixed"
    image_opts = {
        "pdf_figure_min_pixels":      max(1000, int(pdf_figure_min_pixels)),
    }
    jobs[job_id] = {
        "job_id": job_id, "status": "queued", "step": "",
        "document_id": document_id, "notebook_id": notebook_id,
        "filename": file.filename, "chunks": 0, "error": None,
        "pdf_mode": pdf_mode, "chunking_strategy": strategy, "started_at": time.time()
    }
    bg.add_task(pipeline, job_id, content, file.filename or "", document_id, notebook_id,
                callback_url, pdf_mode, parsed_transcript, strategy, image_opts)
    return {"job_id": job_id, "status": "queued"}

@app.post("/process-url")
async def process_url(
    bg: BackgroundTasks,
    url: str = Form(...),
    document_id: str = Form(...),
    notebook_id: str = Form(...),
    callback_url: str = Form(""),
    pdf_mode: str = Form("basic"),
    chunking_strategy: str = Form("fixed"),
    # Phase 4: PDF-Bildextraktion (relevant für heruntergeladene PDFs)
    pdf_figure_min_pixels: int = Form(40000),
    # Phase 3: URL-Bildextraktion
    url_image_extraction_enabled: str = Form("false"),
    url_image_min_pixels: int = Form(10000),
    url_image_max_per_page: int = Form(15),
    authorization: str = Header(None)
):
    auth(authorization)
    job_id = str(uuid.uuid4())
    strategy = chunking_strategy if chunking_strategy in ("fixed", "semantic") else "fixed"
    image_opts = {
        "pdf_figure_min_pixels":        max(1000, int(pdf_figure_min_pixels)),
        "url_image_extraction_enabled": str(url_image_extraction_enabled).lower() in ("true", "1", "yes"),
        "url_image_min_pixels":         max(1000, int(url_image_min_pixels)),
        "url_image_max_per_page":       max(1, int(url_image_max_per_page)),
    }
    jobs[job_id] = {
        "job_id": job_id, "status": "queued", "step": "",
        "document_id": document_id, "notebook_id": notebook_id,
        "filename": url, "chunks": 0, "error": None,
        "pdf_mode": pdf_mode, "chunking_strategy": strategy, "started_at": time.time()
    }
    bg.add_task(url_pipeline, job_id, url, document_id, notebook_id, callback_url, pdf_mode, strategy, image_opts)
    return {"job_id": job_id, "status": "queued"}

@app.post("/research-url")
async def research_url(
    bg: BackgroundTasks,
    url: str = Form(...),
    callback_url: str = Form(""),
    authorization: str = Header(None)
):
    auth(authorization)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id, "status": "queued", "step": "",
        "document_id": "", "notebook_id": "",
        "filename": url, "chunks": 0, "error": None,
        "started_at": time.time(), "result_text": None
    }
    bg.add_task(research_pipeline, job_id, url, callback_url)
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
    qvec = embed_single(query)
    must = [FieldCondition(key="notebook_id", match=MatchValue(value=notebook_id))]
    if document_ids:
        ids = [d.strip() for d in document_ids.split(",") if d.strip()]
        if ids:
            must.append(FieldCondition(key="document_id", match=MatchAny(any=ids)))

    # Phase 7: Hybrid Search — mehr Kandidaten holen für Re-Ranking
    fetch_k = RERANK_TOP_K if cross_encoder else top_k

    # 1. Vektor-Suche (Semantic)
    vector_results = qdrant.query_points(
        collection_name=COLLECTION, query=qvec,
        query_filter=Filter(must=must), limit=fetch_k, with_payload=True
    )

    # 2. Text-Suche (BM25-artig) via Qdrant Full-Text Index
    text_results_points = []
    if HYBRID_SEARCH_ENABLED:
        try:
            from qdrant_client.models import MatchText
            text_filter = Filter(must=[
                *must,
                FieldCondition(key="text", match=MatchText(text=query))
            ])
            text_results = qdrant.scroll(
                collection_name=COLLECTION,
                scroll_filter=text_filter,
                limit=fetch_k,
                with_payload=True,
                with_vectors=False,
            )
            text_results_points = text_results[0] if text_results else []
        except Exception as e:
            log.warning("Text-Suche fehlgeschlagen: %s", str(e))

    # 2b. Audio-Vektor-Suche (Dual Embedding) — sucht im Audio-Chunk auf Dokument-Ebene
    audio_doc_ids = set()
    if DUAL_AUDIO_EMBEDDING and EMBEDDING_PROVIDER == "google":
        try:
            audio_col = COLLECTION + "_audio"
            if qdrant.collection_exists(audio_col):
                audio_must = [FieldCondition(key="notebook_id", match=MatchValue(value=notebook_id))]
                if document_ids:
                    ids = [d.strip() for d in document_ids.split(",") if d.strip()]
                    if ids:
                        audio_must.append(FieldCondition(key="document_id", match=MatchAny(any=ids)))
                audio_hits = qdrant.query_points(
                    collection_name=audio_col, query=qvec,
                    query_filter=Filter(must=audio_must), limit=top_k, with_payload=True
                )
                # document_ids aus Audio-Treffern sammeln → Text-Chunks dieser Dokumente bevorzugen
                audio_doc_ids = {r.payload.get("document_id") for r in audio_hits.points if r.score > 0.7}
                if audio_doc_ids:
                    log.info("Dual-Audio: %d Dokumente via Audio-Vektor gefunden", len(audio_doc_ids))
        except Exception as e:
            log.warning("Audio-Suche fehlgeschlagen: %s", e)

    # 3. Ergebnisse mergen (Reciprocal Rank Fusion)
    scored = {}  # point_id -> {payload, rrf_score}
    # Vector-Ergebnisse
    for rank, r in enumerate(vector_results.points):
        pid = str(r.id)
        # Audio-Dokumente erhalten Bonus-Score
        audio_bonus = 0.3 if r.payload.get("document_id") in audio_doc_ids else 0.0
        scored[pid] = {
            "payload": r.payload,
            "rrf_score": 1.0 / (60 + rank) + audio_bonus,
            "vector_score": r.score,
        }
    # Text-Ergebnisse dazumischen
    for rank, r in enumerate(text_results_points):
        pid = str(r.id)
        rrf_add = 1.0 / (60 + rank)
        if pid in scored:
            scored[pid]["rrf_score"] += rrf_add
        else:
            scored[pid] = {
                "payload": r.payload,
                "rrf_score": rrf_add,
                "vector_score": 0.0,
            }

    # Nach RRF-Score sortieren
    candidates = sorted(scored.values(), key=lambda x: x["rrf_score"], reverse=True)[:fetch_k]

    # 4. Re-Ranking mit Cross-Encoder
    if cross_encoder and len(candidates) > 0:
        pairs = [(query, c["payload"].get("original_text", c["payload"].get("text", ""))) for c in candidates]
        try:
            ce_scores = cross_encoder.predict(pairs)
            for i, score in enumerate(ce_scores):
                candidates[i]["final_score"] = float(score)
            candidates.sort(key=lambda x: x["final_score"], reverse=True)
        except Exception as e:
            log.warning("Re-Ranking fehlgeschlagen: %s", str(e))
            for c in candidates:
                c["final_score"] = c["rrf_score"]
    else:
        for c in candidates:
            c["final_score"] = c.get("vector_score", c["rrf_score"])

    # Top-K zurückgeben. Bild-Felder werden optional mitgeliefert, damit Laravels
    # RAG-Orchestrator/Gateway multimodale Treffer als sichtbare Abbildungen rendern kann.
    final = candidates[:top_k]
    out = []
    for c in final:
        p = c["payload"]
        item = {
            "text":        p.get("original_text", p.get("text", "")),
            "document_id": p.get("document_id", ""),
            "chunk_index": p.get("chunk_index", 0),
            "filename":    p.get("filename", ""),
            "score":       c["final_score"],
        }
        if p.get("has_image"):
            item["has_image"]  = True
            item["image_b64"]  = p.get("image_b64")
            item["page_num"]   = p.get("page_num", 0)
            if p.get("source_url"):
                item["source_url"] = p.get("source_url")
            if p.get("image_hash"):
                item["image_hash"] = p.get("image_hash")
        elif p.get("page_num"):
            item["page_num"] = p.get("page_num")
        out.append(item)
    return {"results": out}

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

def prepare_audio(content, filename):
    """Konvertiert jedes Audio-Format zu MP3 und optimiert für STT via FFmpeg.
    - Rauschunterdrückung (afftdn)
    - Tiefes Brummen entfernen (highpass)
    - Hochfrequentes Rauschen entfernen (lowpass)
    - Lautstärke-Normalisierung (loudnorm, EBU R128)
    - Kanäle bleiben erhalten (wichtig für Speaker-Diarization!)
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "mp3"
    src_path = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as src:
            src.write(content)
            src_path = src.name
        out_path = src_path + ".mp3"

        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-vn",                                          # kein Video
            "-af", ",".join([
                "highpass=f=80",                            # Brummen/Wind < 80 Hz entfernen
                "lowpass=f=8000",                           # Hochfrequentes Rauschen > 8 kHz entfernen
                "afftdn=nf=-25",                            # FFT-basierte Rauschunterdrückung
                "loudnorm=I=-16:TP=-1.5:LRA=11",           # EBU R128 Lautstärke-Normalisierung
            ]),
            "-ar", "16000",                                 # 16 kHz Sample Rate (optimal für STT)
            "-b:a", "128k",                                 # 128 kbps (ausreichend für Sprache)
            "-f", "mp3",
            out_path,
        ]
        log.info("FFmpeg: preparing %s (%d bytes) → %s", filename, len(content), " ".join(cmd[6:]))
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            log.error("FFmpeg failed (rc=%d): %s", result.returncode, stderr[-500:])
            raise Exception(f"FFmpeg conversion failed: {stderr[-200:]}")

        with open(out_path, "rb") as f:
            converted = f.read()
        new_filename = filename.rsplit(".", 1)[0] + ".mp3"
        log.info("FFmpeg: %s → %s (%d → %d bytes)", filename, new_filename, len(content), len(converted))
        return converted, new_filename
    finally:
        if src_path and os.path.exists(src_path):
            os.unlink(src_path)
        if out_path and os.path.exists(out_path):
            os.unlink(out_path)

def pipeline(job_id, content, filename, document_id, notebook_id, callback_url, pdf_mode="basic",
             existing_transcript=None, chunking_strategy="fixed", image_opts=None):
    """
    pdf_mode:
        'basic'              – nur Text via Tika (Standard)
        'vision_description' – Bilder via Vision-LLM beschreiben
        'multimodal'         – zusätzlich Bilder direkt via Gemini Embedding 2 embedden
    existing_transcript: vorhandenes Transkript-Dict (spart Re-Transkription beim Reindex)
    chunking_strategy:
        'fixed'    – feste Wort-Anzahl pro Chunk (Standard)
        'semantic' – Embedding-basierte Splits an inhaltlichen Grenzen
    image_opts (dict, optional):
        pdf_figure_min_pixels: int, Filter für winzige Raster-Assets (Icons/Ornamente)
    """
    image_opts = image_opts or {}
    fig_min_pixels = int(image_opts.get("pdf_figure_min_pixels", 40000))

    try:
        jobs[job_id]["status"] = "processing"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        transcript_data = None
        # Optional extrahierte Multimodal-Punkte (Bild-Vektoren) — werden am Ende zusätzlich
        # zur Text-Pipeline in Qdrant gespeichert, wenn gefüllt.
        image_points: list[PointStruct] = []
        if ext in AUDIO_EXTENSIONS:
            if existing_transcript and existing_transcript.get("full_text"):
                # Vorhandenes Transkript wiederverwenden – keine erneute Transkription
                log.info("Job %s: Vorhandenes Transkript wird wiederverwendet (kein API-Call)", job_id)
                text = existing_transcript["full_text"]
                transcript_data = existing_transcript
            else:
                jobs[job_id]["step"] = "preparing audio"
                content, filename = prepare_audio(content, filename)
                jobs[job_id]["step"] = "transcribing"
                text, transcript_data = transcribe(content, filename)
        else:
            jobs[job_id]["step"] = "extracting"
            ext_lower = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if ext_lower == "pdf" and pdf_mode == "multimodal":
                # Phase 4: Tika-Text IMMER parallel laufen lassen, damit das PDF auch ohne
                # Raster-Bilder durchsuchbar bleibt. Bilder werden zusätzlich embedded.
                jobs[job_id]["step"] = "extracting (multimodal text)"
                text = extract(content)  # Tika-Text-Basis
                jobs[job_id]["step"] = "extracting (multimodal images)"
                # Phase 6: Dedup-Seed aus Qdrant — dasselbe Bild nicht doppelt embedden
                seen_hashes = load_existing_image_hashes(notebook_id, exclude_document_id=document_id)
                image_chunks = extract_pdf_figures(content, job_id, min_pixels=fig_min_pixels, seen_hashes=seen_hashes)
                for p in image_chunks:
                    image_points.append(PointStruct(
                        id=str(uuid.uuid4()), vector=p["vector"], payload={
                            "text":          p["text"],
                            "original_text": p["text"],
                            "context":       "",
                            "notebook_id":   notebook_id,
                            "document_id":   document_id,
                            "filename":      filename,
                            "chunk_index":   -(p["figure_index"] + 1),  # negativer Index → Image-Punkt
                            "has_image":     True,
                            "image_b64":     p["image_b64"],
                            "page_num":      p["page_num"],
                            "image_hash":    p.get("image_hash"),
                        }
                    ))
                if not image_points:
                    log.info("Job %s: keine Raster-Bilder extrahiert, nur Tika-Text", job_id)
            elif ext_lower == "pdf" and pdf_mode == "vision_description":
                jobs[job_id]["step"] = "extracting (vision)"
                text = extract_pdf_with_vision(content, job_id)
            else:
                text = extract(content)
        # Wenn weder Text noch Bild-Punkte vorhanden sind → Fehler. Wenn nur Bilder gefunden
        # wurden (PDF ohne Textlayer), überspringen wir die Text-Pipeline und speichern nur Bilder.
        has_text = bool(text and len(text.strip()) >= 10)
        if not has_text and not image_points:
            raise Exception(f"No text extracted from {filename}")

        # Einmalig alle bisherigen Punkte dieses Dokuments löschen (für Re-Index)
        qdrant.delete(
            collection_name=COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
            )
        )

        chunks = []
        if has_text:
            log.info("Job %s: %d chars from %s", job_id, len(text), filename)
            jobs[job_id]["step"] = "chunking"
            if chunking_strategy == "semantic":
                chunks = semantic_chunk_text(text, job_id=job_id)
            else:
                chunks = chunk_text(text)
            log.info("Job %s: %d chunks (strategy=%s)", job_id, len(chunks), chunking_strategy)

            # Phase 7: Contextual Retrieval — Chunks mit KI-Kontext anreichern
            if CONTEXTUAL_RETRIEVAL:
                jobs[job_id]["step"] = "contextualizing"
                chunks = contextualize_chunks(chunks, text, job_id)

            jobs[job_id]["step"] = "embedding"
            # Embedding auf kontextualisiertem Text (falls vorhanden)
            embed_texts = [c.get("contextualized_text", c["text"]) for c in chunks]
            embeddings = embed(embed_texts, job_id=job_id)

            jobs[job_id]["step"] = "storing"
            points = [
                PointStruct(id=str(uuid.uuid4()), vector=emb, payload={
                    "text": c.get("contextualized_text", c["text"]),
                    "original_text": c["text"],
                    "context": c.get("context", ""),
                    "notebook_id": notebook_id,
                    "document_id": document_id, "filename": filename, "chunk_index": i
                })
                for i, (c, emb) in enumerate(zip(chunks, embeddings))
            ]
            for i in range(0, len(points), 100):
                qdrant.upsert(collection_name=COLLECTION, points=points[i:i+100])

        # Multimodal-Bild-Punkte zusätzlich speichern (Phase 4)
        if image_points:
            jobs[job_id]["step"] = "storing_images"
            for i in range(0, len(image_points), 50):
                qdrant.upsert(collection_name=COLLECTION, points=image_points[i:i+50])
            log.info("Job %s: %d Bild-Punkte gespeichert", job_id, len(image_points))

        # Dual-Audio-Embedding: Gesamt-Audio als einen eigenen Vektor in der Audio-Collection
        if DUAL_AUDIO_EMBEDDING and EMBEDDING_PROVIDER == "google" and ext in AUDIO_EXTENSIONS:
            jobs[job_id]["step"] = "audio_embedding"
            audio_vec = _embed_google_audio(content, mime_type="audio/mp3")
            if audio_vec:
                audio_col = COLLECTION + "_audio"
                qdrant.delete(
                    collection_name=audio_col,
                    points_selector=FilterSelector(
                        filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
                    )
                )
                qdrant.upsert(
                    collection_name=audio_col,
                    points=[PointStruct(
                        id=str(uuid.uuid4()),
                        vector=audio_vec,
                        payload={"document_id": document_id, "notebook_id": notebook_id, "filename": filename}
                    )]
                )
                log.info("Job %s: Audio-Vektor in %s gespeichert", job_id, audio_col)

        total_chunks = len(chunks) + len(image_points)
        jobs[job_id]["status"] = "ready"
        jobs[job_id]["step"] = "done"
        jobs[job_id]["chunks"] = total_chunks
        if transcript_data:
            jobs[job_id]["transcript"] = transcript_data
        log.info("Job %s: done, %d text + %d image chunks stored", job_id, len(chunks), len(image_points))
        if callback_url:
            notify(callback_url, job_id, document_id, "ready", total_chunks,
                   transcript_data=transcript_data, usage=jobs[job_id].get("usage"))
    except Exception as e:
        log.error("Job %s failed: %s", job_id, str(e))
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        if callback_url:
            notify(callback_url, job_id, document_id, "failed", 0, str(e))

# --- URL Processing ---
MAX_CRAWL_PAGES = 20
CRAWL_DELAY = 1.0
URL_TIMEOUT = 15
URL_USER_AGENT = "TrainerBot/1.0"

def normalize_url(url):
    """URL normalisieren: Schema hinzufügen falls nötig."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url

def is_domain_root(url):
    """Prüft ob die URL nur eine Domain ist (kein spezifischer Pfad)."""
    parsed = urlparse(url)
    return parsed.path in ("", "/")

def extract_html_contact_info(html):
    """Kontaktdaten aus Header/Footer/Nav des HTML extrahieren, die trafilatura überspringt."""
    import html as html_module

    contact_parts = []

    # Footer, Header und Kontakt-Sektionen aus dem HTML extrahieren
    # Suche nach <footer>, <header> und Elementen mit typischen Kontakt-IDs/Klassen
    patterns = [
        r'<footer[^>]*>(.*?)</footer>',
        r'<div[^>]*(?:class|id)\s*=\s*["\'][^"\']*(?:contact|kontakt|footer|address|impressum|imprint)[^"\']*["\'][^>]*>(.*?)</div>',
        r'<section[^>]*(?:class|id)\s*=\s*["\'][^"\']*(?:contact|kontakt|footer|address|impressum|imprint)[^"\']*["\'][^>]*>(.*?)</section>',
        r'<address[^>]*>(.*?)</address>',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)
        for match in matches:
            # HTML-Tags entfernen, aber Zeilenumbrüche bei Block-Elementen einfügen
            clean = re.sub(r'<br\s*/?>|</p>|</div>|</li>|</h[1-6]>', '\n', match)
            clean = re.sub(r'<[^>]+>', ' ', clean)
            clean = html_module.unescape(clean)
            # Mehrfache Whitespaces bereinigen
            clean = re.sub(r'[ \t]+', ' ', clean)
            clean = re.sub(r'\n\s*\n', '\n', clean)
            clean = clean.strip()
            if clean and len(clean) > 5:
                contact_parts.append(clean)

    # Auch nach E-Mail-Adressen und Telefonnummern im gesamten HTML suchen
    emails = set(re.findall(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', html))
    phones = set(re.findall(r'(?:tel:|phone:|telefon:|fon:)?\s*[+]?[\d\s\-/().]{7,20}', html, re.IGNORECASE))
    # Nur plausible Telefonnummern behalten (mind. 7 Ziffern)
    phones = {p.strip() for p in phones if sum(c.isdigit() for c in p) >= 7}

    extra_info = []
    if emails:
        extra_info.append("E-Mail: " + ", ".join(sorted(emails)))
    if phones:
        extra_info.append("Telefon: " + ", ".join(sorted(phones)))

    if extra_info:
        contact_parts.append("\n".join(extra_info))

    if not contact_parts:
        return ""

    # Deduplizieren (Footer kann mehrfach matchen)
    seen = set()
    unique = []
    for part in contact_parts:
        normalized = part.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(part)

    return "\n\n--- Kontaktbereich ---\n" + "\n\n".join(unique)


def fetch_page(url):
    """Einzelne Seite laden und Text mit trafilatura extrahieren. Gibt (text, title, html) zurück."""
    with httpx.Client(timeout=URL_TIMEOUT, follow_redirects=True) as c:
        r = c.get(url, headers={"User-Agent": URL_USER_AGENT})
        r.raise_for_status()
        html = r.text
    text = trafilatura.extract(html, include_comments=False, include_tables=True, favor_recall=True)
    # Title aus HTML extrahieren
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    page_title = title_match.group(1).strip() if title_match else url

    # Kontaktdaten aus Header/Footer anhängen (trafilatura überspringt diese)
    contact_info = extract_html_contact_info(html)
    if contact_info:
        text = (text or "") + "\n\n" + contact_info

    return text or "", page_title, html


def extract_url_images(html: str, base_url: str, page_title: str = "",
                       min_pixels: int = 10000, max_per_page: int = 15,
                       seen_hashes: set | None = None, job_id: str = "") -> list[dict]:
    """
    Phase 3: <img>-Tags aus HTML parsen, Bilder herunterladen, nach Größe filtern und
    via Gemini Embedding 2 multimodal embedden.

    Gibt Liste von Dicts zurück: {vector, text, image_b64, page_num, image_hash, source_url,
    alt, figure_index}. Bei fehlendem Google-Provider oder Pillow-Fehler → [].
    """
    if EMBEDDING_PROVIDER != "google" or not google_client:
        return []
    try:
        from bs4 import BeautifulSoup
        from PIL import Image
        import hashlib, base64 as _b64
    except ImportError as e:
        log.warning("extract_url_images: Abhängigkeit fehlt (%s) — übersprungen", e)
        return []

    if seen_hashes is None:
        seen_hashes = set()

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        log.warning("BeautifulSoup Parsing fehlgeschlagen: %s", e)
        return []

    # Kandidaten-URLs einsammeln (mit Duplikat-Schutz)
    candidates: list[tuple[str, str]] = []  # [(abs_url, alt_text)]
    seen_urls: set[str] = set()
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original") or ""
        if not src:
            srcset = img.get("srcset") or ""
            if srcset:
                src = srcset.split(",")[0].strip().split(" ")[0]
        src = (src or "").strip()
        if not src or src.startswith("data:"):
            continue
        abs_url = urljoin(base_url, src)
        parsed = urlparse(abs_url)
        if parsed.scheme not in ("http", "https"):
            continue
        if abs_url in seen_urls:
            continue
        # SVG überspringen (keine sinnvolle Embedding-Quelle via Gemini)
        if parsed.path.lower().endswith(".svg"):
            continue
        seen_urls.add(abs_url)
        alt = (img.get("alt") or "").strip()
        candidates.append((abs_url, alt))
        if len(candidates) >= max_per_page * 3:
            # Hart-Cap gegen extrem bilderlastige Seiten — echter Cap erfolgt nach Filter
            break

    if not candidates:
        return []

    results: list[dict] = []
    figure_index = 0

    with httpx.Client(timeout=URL_TIMEOUT, follow_redirects=True,
                      headers={"User-Agent": URL_USER_AGENT}) as client:
        for abs_url, alt in candidates:
            if len(results) >= max_per_page:
                break
            try:
                r = client.get(abs_url)
                if r.status_code != 200:
                    continue
                content_type = (r.headers.get("content-type") or "").lower()
                if not content_type.startswith("image/") or "svg" in content_type:
                    continue
                img_bytes = r.content
                if not img_bytes or len(img_bytes) < 500:
                    continue

                try:
                    im = Image.open(io.BytesIO(img_bytes))
                    w, h = im.size
                    if w * h < min_pixels:
                        continue
                    if im.mode not in ("RGB", "L"):
                        im = im.convert("RGB")
                    MAX_SIDE = 1400
                    if max(im.size) > MAX_SIDE:
                        im.thumbnail((MAX_SIDE, MAX_SIDE))
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=82, optimize=True)
                    img_bytes = buf.getvalue()
                except Exception as e:
                    log.info("URL-Bild %s übersprungen (Pillow: %s)", abs_url, e)
                    continue

                img_hash = hashlib.sha256(img_bytes).hexdigest()
                if img_hash in seen_hashes:
                    continue
                seen_hashes.add(img_hash)

                vec = _embed_image_google(img_bytes, mime_type="image/jpeg")
                if not vec:
                    continue

                # Text = alt-Text + Seiten-Titel (als Kontext für Text-Matching)
                caption_bits = [p for p in [alt, page_title] if p]
                caption = " — ".join(caption_bits) if caption_bits else f"[Bild von {base_url}]"

                results.append({
                    "vector":       vec,
                    "text":         caption,
                    "image_b64":    _b64.b64encode(img_bytes).decode("utf-8"),
                    "page_num":     0,
                    "image_hash":   img_hash,
                    "source_url":   abs_url,
                    "alt":          alt,
                    "figure_index": figure_index,
                })
                figure_index += 1

                if job_id and job_id in jobs:
                    usage = jobs[job_id].setdefault("usage", {})
                    usage["embedding_tokens"] = usage.get("embedding_tokens", 0) + EMBEDDING_DIMENSIONS
                    usage["embedding_model"]  = EMBEDDING_MODEL
                    usage["url_image_embeds"] = usage.get("url_image_embeds", 0) + 1

            except Exception as e:
                log.info("URL-Bild %s konnte nicht geladen werden: %s", abs_url, e)
                continue

    log.info("Job %s: extract_url_images(%s) → %d Bilder", job_id, base_url, len(results))
    return results

def discover_links(html, base_url):
    """Links auf gleicher Domain aus HTML extrahieren."""
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc.replace("www.", "")
    links = set()
    # Einfaches <a href="..."> Pattern
    for match in re.finditer(r'<a\s+[^>]*href=["\']([^"\'#]+)["\']', html, re.IGNORECASE):
        href = match.group(1)
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        link_domain = parsed.netloc.replace("www.", "")
        # Nur gleiche Domain, nur HTTP(S), keine Dateien
        if link_domain != base_domain:
            continue
        if parsed.scheme not in ("http", "https"):
            continue
        skip_ext = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip", ".mp3",
                     ".mp4", ".wav", ".doc", ".docx", ".xls", ".xlsx", ".css", ".js")
        if any(parsed.path.lower().endswith(ext) for ext in skip_ext):
            continue
        # URL ohne Fragment und Query normalisieren
        clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if clean.endswith("/") and len(clean) > len(f"{parsed.scheme}://{parsed.netloc}/"):
            clean = clean.rstrip("/")
        links.add(clean)
    # Basis-URL entfernen
    base_clean = f"{parsed_base.scheme}://{parsed_base.netloc}{parsed_base.path}".rstrip("/")
    links.discard(base_clean)
    links.discard(base_clean + "/")
    return list(links)[:MAX_CRAWL_PAGES]

def url_pipeline(job_id, url, document_id, notebook_id, callback_url, pdf_mode="basic",
                 chunking_strategy="fixed", image_opts=None):
    """URL(s) laden, Text extrahieren, chunken, embedden, speichern.
    chunking_strategy: 'fixed' (Default) oder 'semantic' (Embedding-basierte Splits).
    image_opts: dict mit URL-Bildextraktions-Flags (Phase 3).
    """
    image_opts = image_opts or {}
    url_img_enabled = bool(image_opts.get("url_image_extraction_enabled", False))
    url_img_min_pixels = int(image_opts.get("url_image_min_pixels", 10000))
    url_img_max_per_page = int(image_opts.get("url_image_max_per_page", 15))

    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["step"] = "fetching"
        url = normalize_url(url)
        log.info("Job %s: fetching %s", job_id, url)

        main_text, main_title, main_html = fetch_page(url)
        all_texts = []
        if main_text and len(main_text.strip()) > 10:
            all_texts.append(f"=== {main_title} – {url} ===\n{main_text}")

        # Bild-Punkte (Phase 3) — werden am Ende zusätzlich in Qdrant gespeichert
        image_points: list[PointStruct] = []
        # Phase 6: Dedup-Seed aus Qdrant, damit dasselbe Bild nicht doppelt embedded wird
        url_seen_hashes: set = load_existing_image_hashes(notebook_id, exclude_document_id=document_id) if url_img_enabled else set()

        if url_img_enabled:
            imgs = extract_url_images(
                main_html, url, page_title=main_title,
                min_pixels=url_img_min_pixels, max_per_page=url_img_max_per_page,
                seen_hashes=url_seen_hashes, job_id=job_id,
            )
            for p in imgs:
                image_points.append(PointStruct(
                    id=str(uuid.uuid4()), vector=p["vector"], payload={
                        "text":          p["text"],
                        "original_text": p["text"],
                        "context":       "",
                        "notebook_id":   notebook_id,
                        "document_id":   document_id,
                        "filename":      url,
                        "chunk_index":   -(p["figure_index"] + 1),
                        "has_image":     True,
                        "image_b64":     p["image_b64"],
                        "page_num":      0,
                        "image_hash":    p.get("image_hash"),
                        "source_url":    p.get("source_url"),
                    }
                ))

        # Bei Domain-Root: Unterseiten crawlen
        if is_domain_root(url):
            jobs[job_id]["step"] = "crawling"
            links = discover_links(main_html, url)
            log.info("Job %s: found %d links to crawl", job_id, len(links))
            for i, link in enumerate(links):
                try:
                    time.sleep(CRAWL_DELAY)
                    page_text, page_title, page_html = fetch_page(link)
                    if page_text and len(page_text.strip()) > 10:
                        all_texts.append(f"=== {page_title} – {link} ===\n{page_text}")
                        log.info("Job %s: crawled %d/%d – %s (%d chars)",
                                 job_id, i + 1, len(links), link, len(page_text))
                    else:
                        log.info("Job %s: skipped %s (no content)", job_id, link)

                    if url_img_enabled:
                        imgs = extract_url_images(
                            page_html, link, page_title=page_title,
                            min_pixels=url_img_min_pixels, max_per_page=url_img_max_per_page,
                            seen_hashes=url_seen_hashes, job_id=job_id,
                        )
                        for p in imgs:
                            image_points.append(PointStruct(
                                id=str(uuid.uuid4()), vector=p["vector"], payload={
                                    "text":          p["text"],
                                    "original_text": p["text"],
                                    "context":       "",
                                    "notebook_id":   notebook_id,
                                    "document_id":   document_id,
                                    "filename":      link,
                                    "chunk_index":   -(p["figure_index"] + 1 + len(image_points)),
                                    "has_image":     True,
                                    "image_b64":     p["image_b64"],
                                    "page_num":      0,
                                    "image_hash":    p.get("image_hash"),
                                    "source_url":    p.get("source_url"),
                                }
                            ))
                except Exception as ex:
                    log.warning("Job %s: failed to fetch %s: %s", job_id, link, str(ex))

        combined_text = "\n\n".join(all_texts)
        has_text = bool(combined_text and len(combined_text.strip()) >= 10)
        if not has_text and not image_points:
            raise Exception(f"No text extracted from {url}")

        log.info("Job %s: %d chars total from %d pages, %d Bilder", job_id,
                 len(combined_text), len(all_texts), len(image_points))

        # Punkte dieses Dokuments entfernen (Re-Index-freundlich)
        qdrant.delete(
            collection_name=COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
            )
        )

        chunks = []
        if has_text:
            jobs[job_id]["step"] = "chunking"
            if chunking_strategy == "semantic":
                chunks = semantic_chunk_text(combined_text, job_id=job_id)
            else:
                chunks = chunk_text(combined_text)
            log.info("Job %s: %d chunks (strategy=%s)", job_id, len(chunks), chunking_strategy)

            # Phase 7: Contextual Retrieval
            if CONTEXTUAL_RETRIEVAL:
                jobs[job_id]["step"] = "contextualizing"
                chunks = contextualize_chunks(chunks, combined_text, job_id)

            jobs[job_id]["step"] = "embedding"
            embed_texts = [c.get("contextualized_text", c["text"]) for c in chunks]
            embeddings = embed(embed_texts, job_id=job_id)

            jobs[job_id]["step"] = "storing"
            points = [
                PointStruct(id=str(uuid.uuid4()), vector=emb, payload={
                    "text": c.get("contextualized_text", c["text"]),
                    "original_text": c["text"],
                    "context": c.get("context", ""),
                    "notebook_id": notebook_id,
                    "document_id": document_id, "filename": url, "chunk_index": i
                })
                for i, (c, emb) in enumerate(zip(chunks, embeddings))
            ]
            for i in range(0, len(points), 100):
                qdrant.upsert(collection_name=COLLECTION, points=points[i:i + 100])

        # URL-Bild-Punkte (Phase 3) zusätzlich speichern
        if image_points:
            jobs[job_id]["step"] = "storing_images"
            for i in range(0, len(image_points), 50):
                qdrant.upsert(collection_name=COLLECTION, points=image_points[i:i + 50])
            log.info("Job %s: %d URL-Bild-Punkte gespeichert", job_id, len(image_points))

        total = len(chunks) + len(image_points)
        jobs[job_id]["status"] = "ready"
        jobs[job_id]["step"] = "done"
        jobs[job_id]["chunks"] = total
        log.info("Job %s: done, %d Text + %d Bild-Chunks gespeichert", job_id, len(chunks), len(image_points))
        if callback_url:
            notify(callback_url, job_id, document_id, "ready", total,
                   usage=jobs[job_id].get("usage"))
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


def extract_pdf_with_vision(content: bytes, job_id: str = "") -> str:
    """
    PDF-Seiten via PyMuPDF als Bilder extrahieren und via Vision-LLM beschreiben.
    pdf_mode='vision_description': Ergebnis sind Textbeschreibungen aller Seiten.
    Fällt bei Fehler auf Tika-Extraktion zurück.
    """
    try:
        import fitz  # PyMuPDF
        import base64 as _b64

        doc = fitz.open(stream=content, filetype="pdf")
        page_texts = []

        for page_num, page in enumerate(doc):
            # Seite als PNG rendern (150 DPI reicht für Vision)
            mat = fitz.Matrix(150 / 72, 150 / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            b64_img = _b64.b64encode(img_bytes).decode("utf-8")

            # Text der Seite (aus PDF-Layer) als Kontext
            page_text = page.get_text().strip()

            try:
                prompt = (
                    "Beschreibe den vollständigen Inhalt dieser PDF-Seite auf Deutsch. "
                    "Extrahiere allen Text, Tabellen, Diagramme und Bilder als strukturierten Text. "
                    "Ignoriere reine Designelemente ohne Informationsgehalt."
                )
                if page_text:
                    prompt += f"\n\nExtrahierter Text der Seite (zur Kontrolle): {page_text[:500]}"

                response = oai.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                            {"type": "text", "text": prompt},
                        ]
                    }],
                    max_tokens=1000,
                )
                description = response.choices[0].message.content.strip()
                page_texts.append(f"[Seite {page_num + 1}]\n{description}")

                # Usage tracken
                if job_id and job_id in jobs:
                    usage = jobs[job_id].setdefault("usage", {})
                    usage["vision_calls"]       = usage.get("vision_calls", 0) + 1
                    usage["context_tokens_in"]  = usage.get("context_tokens_in", 0) + (response.usage.prompt_tokens or 0)
                    usage["context_tokens_out"] = usage.get("context_tokens_out", 0) + (response.usage.completion_tokens or 0)
                    usage["context_model"]      = CHAT_MODEL

            except Exception as e:
                log.warning("Job %s: Vision-Beschreibung Seite %d fehlgeschlagen: %s", job_id, page_num + 1, e)
                if page_text:
                    page_texts.append(f"[Seite {page_num + 1}]\n{page_text}")

        doc.close()
        return "\n\n".join(page_texts) if page_texts else extract(content)

    except ImportError:
        log.warning("PyMuPDF nicht installiert – Fallback auf Tika")
        return extract(content)
    except Exception as e:
        log.error("PDF Vision-Extraktion fehlgeschlagen: %s – Fallback auf Tika", e)
        return extract(content)


def load_existing_image_hashes(notebook_id: str, exclude_document_id: str = "") -> set:
    """
    Phase 6: Liefert alle `image_hash`-Werte, die für dieses Notebook bereits in Qdrant
    gespeichert sind — exklusive des aktuellen Dokuments (damit Re-Index nicht das eigene
    Dokument als "Duplikat" behandelt). Wird als Seed für `seen_hashes` in den
    Extraktionsfunktionen verwendet, um identische Bilder nicht mehrfach zu embedden.
    """
    hashes: set = set()
    try:
        must = [
            FieldCondition(key="notebook_id", match=MatchValue(value=notebook_id)),
            FieldCondition(key="has_image",   match=MatchValue(value=True)),
        ]
        must_not = []
        if exclude_document_id:
            must_not.append(FieldCondition(key="document_id", match=MatchValue(value=exclude_document_id)))
        next_page = None
        while True:
            res = qdrant.scroll(
                collection_name=COLLECTION,
                scroll_filter=Filter(must=must, must_not=must_not or None),
                limit=256,
                offset=next_page,
                with_payload=True,
                with_vectors=False,
            )
            points, next_page = res
            for pt in points:
                h = (pt.payload or {}).get("image_hash")
                if h:
                    hashes.add(h)
            if next_page is None:
                break
    except Exception as e:
        log.warning("load_existing_image_hashes fehlgeschlagen (notebook=%s): %s", notebook_id, e)
    return hashes


def _embed_image_google(img_bytes: bytes, mime_type: str = "image/jpeg") -> list[float] | None:
    """Ein einzelnes Bild via Gemini Embedding 2 embedden. Gibt (normalisierten) Vektor zurück
    oder None bei Fehler / wenn Provider != google."""
    if EMBEDDING_PROVIDER != "google" or not google_client:
        return None
    try:
        from google.genai import types
        part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
        result = google_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[part],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
        vec = result.embeddings[0].values
        if EMBEDDING_DIMENSIONS < 3072:
            import numpy as np
            v = np.array(vec)
            norm = np.linalg.norm(v)
            if norm > 0:
                vec = (v / norm).tolist()
        return vec
    except Exception as e:
        log.warning("Bild-Embedding fehlgeschlagen: %s", e)
        return None


def extract_pdf_figures(content: bytes, job_id: str = "", min_pixels: int = 40000,
                         seen_hashes: set | None = None) -> list[dict]:
    """
    Phase 4: Extrahiert jedes eingebettete Raster-Bild aus einem PDF einzeln und embeddet es via
    Gemini Embedding 2. Vektor-Diagramme ohne Raster-Quelle werden hier nicht erfasst — der
    Textlayer wird aber parallel via Tika verarbeitet, sodass das Dokument trotzdem durchsuchbar bleibt.

    Gibt Liste von Dicts zurück: {vector, text, image_b64, page_num, image_hash, figure_index}.
    Fällt auf leere Liste zurück wenn Provider != google oder PyMuPDF fehlt.
    """
    if EMBEDDING_PROVIDER != "google" or not google_client:
        log.warning("extract_pdf_figures benötigt EMBEDDING_PROVIDER=google — übersprungen")
        return []
    try:
        import fitz  # PyMuPDF
        import hashlib, base64 as _b64
    except ImportError:
        log.warning("PyMuPDF nicht installiert — extract_pdf_figures übersprungen")
        return []

    if seen_hashes is None:
        seen_hashes = set()

    results: list[dict] = []
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as e:
        log.error("PDF konnte nicht geöffnet werden: %s", e)
        return []

    figure_index = 0
    for page_num, page in enumerate(doc, start=1):
        try:
            page_text = (page.get_text() or "").strip()
        except Exception:
            page_text = ""

        try:
            images = page.get_images(full=True)
        except Exception as e:
            log.warning("get_images Seite %d fehlgeschlagen: %s", page_num, e)
            continue

        for img_info in images:
            xref = img_info[0]
            try:
                base_img = doc.extract_image(xref)
            except Exception as e:
                log.warning("extract_image xref=%s Seite %d fehlgeschlagen: %s", xref, page_num, e)
                continue

            img_bytes = base_img.get("image")
            if not img_bytes:
                continue
            ext = (base_img.get("ext") or "jpeg").lower()
            width = int(base_img.get("width") or 0)
            height = int(base_img.get("height") or 0)

            # Größenfilter: Icons, Ornamente, Tracking-Pixel raus
            if width * height < min_pixels:
                continue

            # Alpha/CMYK → RGB normalisieren, dann als JPEG serialisieren (kompakt für Payload)
            try:
                from PIL import Image
                im = Image.open(io.BytesIO(img_bytes))
                if im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
                # Downscale sehr großer Bilder (Payload-Größe begrenzen)
                MAX_SIDE = 1400
                if max(im.size) > MAX_SIDE:
                    im.thumbnail((MAX_SIDE, MAX_SIDE))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=82, optimize=True)
                img_bytes = buf.getvalue()
                ext = "jpeg"
            except Exception as e:
                log.warning("Pillow-Reencoding fehlgeschlagen (Seite %d): %s — Original wird genutzt", page_num, e)

            img_hash = hashlib.sha256(img_bytes).hexdigest()
            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)

            vec = _embed_image_google(img_bytes, mime_type=f"image/{ext}")
            if not vec:
                continue

            # Caption = Seiten-Textauszug (erste ~400 Zeichen) als Kontext für Text-Matching
            caption = (page_text[:400] + ("…" if len(page_text) > 400 else "")) if page_text else f"[Seite {page_num} — Abbildung {figure_index + 1}]"

            results.append({
                "vector":        vec,
                "text":          caption,
                "image_b64":     _b64.b64encode(img_bytes).decode("utf-8"),
                "page_num":      page_num,
                "image_hash":    img_hash,
                "figure_index":  figure_index,
            })
            figure_index += 1

            if job_id and job_id in jobs:
                usage = jobs[job_id].setdefault("usage", {})
                usage["embedding_tokens"] = usage.get("embedding_tokens", 0) + EMBEDDING_DIMENSIONS
                usage["embedding_model"]  = EMBEDDING_MODEL
                usage["figure_embeds"]    = usage.get("figure_embeds", 0) + 1

    doc.close()
    log.info("Job %s: extract_pdf_figures → %d Abbildungen", job_id, len(results))
    return results


def transcribe(content, filename):
    """Transkribiert Audio – gibt (text, transcript_data) zurück."""
    if TRANSCRIPTION_PROVIDER == "deepgram":
        return transcribe_deepgram(content, filename)
    elif TRANSCRIPTION_PROVIDER == "assemblyai":
        return transcribe_assemblyai(content, filename)
    else:
        return transcribe_whisper(content, filename)

def transcribe_whisper(content, filename):
    """Fallback: OpenAI Whisper (kein Diarization)."""
    f = io.BytesIO(content)
    f.name = filename
    result = oai.audio.transcriptions.create(model="whisper-1", file=f)
    transcript_data = {
        "full_text": result.text,
        "utterances": [{"speaker": "A", "text": result.text, "start": 0, "end": 0}],
        "speakers_map": {"A": "Sprecher A"},
        "language": None,
        "duration_ms": None,
        "speakers_count": 1,
        "provider": "whisper",
        "provider_job_id": None,
    }
    return result.text, transcript_data

def transcribe_deepgram(content, filename):
    """Deepgram Nova-3: Pre-recorded Audio mit Diarization."""
    import mimetypes
    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type:
        mime_type = "audio/mpeg"

    base_url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": mime_type,
    }

    # Query-Parameter aufbauen
    params = {
        "model": DEEPGRAM_MODEL,
        "punctuate": str(DEEPGRAM_PUNCTUATE).lower(),
        "diarize": str(DEEPGRAM_DIARIZE).lower(),
        "smart_format": str(DEEPGRAM_SMART_FORMAT).lower(),
        "utterances": "true",
    }
    if DEEPGRAM_LANGUAGE:
        params["language"] = DEEPGRAM_LANGUAGE
    else:
        params["detect_language"] = "true"
    if DEEPGRAM_KEYTERMS:
        # Deepgram: keywords werden als wiederholte Query-Parameter übergeben
        # httpx unterstützt das über eine Liste von Tuples
        keywords = [k.strip() for k in DEEPGRAM_KEYTERMS.split(",") if k.strip()]
    else:
        keywords = []

    log.info("Deepgram: transcribing %s (%d bytes), model=%s, lang=%s",
             filename, len(content), DEEPGRAM_MODEL, DEEPGRAM_LANGUAGE or "auto")

    # Für keywords müssen wir die URL manuell bauen (wiederholte params)
    query_parts = [f"{k}={v}" for k, v in params.items()]
    for kw in keywords:
        query_parts.append(f"keywords={kw}")
    url = f"{base_url}?{'&'.join(query_parts)}"

    resp = httpx.post(url, headers=headers, content=content, timeout=300)
    if resp.status_code != 200:
        log.error("Deepgram: request failed: %s %s", resp.status_code, resp.text)
        resp.raise_for_status()

    data = resp.json()

    # Volltext aus channels
    full_text = ""
    channels = data.get("results", {}).get("channels", [])
    if channels:
        alts = channels[0].get("alternatives", [])
        if alts:
            full_text = alts[0].get("transcript", "")

    # Utterances mit Speaker-Info
    utterances_raw = data.get("results", {}).get("utterances", [])

    # Deepgram liefert Speaker-IDs als int (0, 1, 2...) → auf Buchstaben (A, B, C...) mappen
    # damit das Frontend die Farben korrekt zuordnet (speakerColors: {A:..., B:..., C:...})
    _spk_letter_cache = {}
    def _speaker_to_letter(spk_id):
        if spk_id not in _spk_letter_cache:
            idx = len(_spk_letter_cache)
            _spk_letter_cache[spk_id] = chr(ord("A") + idx) if idx < 26 else f"S{idx}"
        return _spk_letter_cache[spk_id]

    speakers = set()
    raw_utterances = []
    for u in utterances_raw:
        speaker = _speaker_to_letter(u.get("speaker", 0))
        speakers.add(speaker)
        raw_utterances.append({
            "speaker": speaker,
            "text": u.get("transcript", ""),
            "start": int(u.get("start", 0) * 1000),  # Sekunden → ms
            "end": int(u.get("end", 0) * 1000),
        })

    # Aufeinanderfolgende Utterances desselben Sprechers zusammenfügen
    utterances = []
    for utt in raw_utterances:
        if utterances and utterances[-1]["speaker"] == utt["speaker"]:
            # Zusammenführen: Text anhängen, End-Zeit aktualisieren
            utterances[-1]["text"] += " " + utt["text"]
            utterances[-1]["end"] = utt["end"]
        else:
            utterances.append(dict(utt))

    # Metadata
    metadata = data.get("metadata", {})
    duration_s = metadata.get("duration", 0)
    duration_ms = int(duration_s * 1000) if duration_s else 0
    request_id = metadata.get("request_id", "")

    # Sprache aus detected_language (wenn auto-detect)
    language = None
    if channels:
        alts = channels[0].get("alternatives", [])
        if alts:
            lang_list = alts[0].get("languages", [])
            if lang_list:
                language = lang_list[0]
    if not language and DEEPGRAM_LANGUAGE:
        language = DEEPGRAM_LANGUAGE

    speakers_map = {s: f"Sprecher {s}" for s in sorted(speakers)}

    transcript_data = {
        "full_text": full_text,
        "utterances": utterances,
        "speakers_map": speakers_map,
        "language": language,
        "duration_ms": duration_ms,
        "speakers_count": len(speakers),
        "provider": "deepgram",
        "provider_job_id": request_id,
    }

    log.info("Deepgram: done. %d utterances (merged from %d), %d speakers, lang=%s",
             len(utterances), len(raw_utterances), len(speakers), language)

    return full_text, transcript_data

def transcribe_assemblyai(content, filename):
    """AssemblyAI: Upload → Transcribe → Poll → Return mit Diarization."""
    base_url = "https://api.assemblyai.com/v2"
    headers = {"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}
    upload_headers = {"authorization": ASSEMBLYAI_API_KEY}

    # 1. Audio hochladen
    log.info("AssemblyAI: uploading %s (%d bytes)", filename, len(content))
    upload_resp = httpx.post(
        f"{base_url}/upload", headers=upload_headers, content=content, timeout=300
    )
    upload_resp.raise_for_status()
    audio_url = upload_resp.json()["upload_url"]
    log.info("AssemblyAI: uploaded → %s", audio_url[:80])

    # 2. Transkription starten
    # speech_models als Array: universal-3-pro zuerst, universal-2 als Fallback
    # (wichtig für Speaker Labels: universal-2 als Fallback sichert Diarization ab)
    primary_model = ASSEMBLYAI_SPEECH_MODEL if ASSEMBLYAI_SPEECH_MODEL else "universal-3-pro"
    speech_models = [primary_model]
    if primary_model == "universal-3-pro":
        speech_models.append("universal-2")

    transcript_config = {
        "audio_url": audio_url,
        "speaker_labels": ASSEMBLYAI_SPEAKER_LABELS,
        "speech_models": speech_models,
    }
    if ASSEMBLYAI_SPEAKER_LABELS and ASSEMBLYAI_SPEAKERS_EXPECTED > 0:
        transcript_config["speakers_expected"] = ASSEMBLYAI_SPEAKERS_EXPECTED
    if ASSEMBLYAI_LANGUAGE_DETECTION:
        transcript_config["language_detection"] = True
    if ASSEMBLYAI_LANGUAGE:
        transcript_config["language_code"] = ASSEMBLYAI_LANGUAGE
        transcript_config.pop("language_detection", None)
    if ASSEMBLYAI_PROMPT:
        transcript_config["prompt"] = ASSEMBLYAI_PROMPT
    if ASSEMBLYAI_KEYTERMS:
        transcript_config["keyterms_prompt"] = [k.strip() for k in ASSEMBLYAI_KEYTERMS.split(",") if k.strip()]
    if ASSEMBLYAI_TEMPERATURE is not None:
        transcript_config["temperature"] = ASSEMBLYAI_TEMPERATURE

    log.info("AssemblyAI: transcript config: %s", transcript_config)
    start_resp = httpx.post(
        f"{base_url}/transcript", headers=headers, json=transcript_config, timeout=30
    )
    if start_resp.status_code != 200:
        log.error("AssemblyAI: transcript start failed: %s %s", start_resp.status_code, start_resp.text)
        start_resp.raise_for_status()
    transcript_id = start_resp.json()["id"]
    log.info("AssemblyAI: transcription started → %s", transcript_id)

    # 3. Polling bis fertig
    poll_url = f"{base_url}/transcript/{transcript_id}"
    while True:
        time.sleep(5)
        poll_resp = httpx.get(poll_url, headers=headers, timeout=30)
        poll_resp.raise_for_status()
        data = poll_resp.json()
        status = data["status"]
        if status == "completed":
            break
        elif status == "error":
            raise Exception(f"AssemblyAI error: {data.get('error', 'Unknown')}")
        log.info("AssemblyAI: polling %s → %s", transcript_id, status)

    # 4. Ergebnis formatieren
    full_text = data.get("text", "")
    utterances_raw = data.get("utterances", [])
    duration_ms = data.get("audio_duration", 0)
    if duration_ms:
        duration_ms = int(duration_ms * 1000)  # Sekunden → ms
    language = data.get("language_code", None)

    # Sprecher sammeln
    speakers = set()
    utterances = []
    for u in utterances_raw:
        speaker = u.get("speaker", "A")
        speakers.add(speaker)
        utterances.append({
            "speaker": speaker,
            "text": u.get("text", ""),
            "start": u.get("start", 0),
            "end": u.get("end", 0),
        })

    # Default-Sprechernamen
    speakers_map = {s: f"Sprecher {s}" for s in sorted(speakers)}

    transcript_data = {
        "full_text": full_text,
        "utterances": utterances,
        "speakers_map": speakers_map,
        "language": language,
        "duration_ms": duration_ms,
        "speakers_count": len(speakers),
        "provider": "assemblyai",
        "provider_job_id": transcript_id,
    }

    log.info("AssemblyAI: done. %d utterances, %d speakers, lang=%s",
             len(utterances), len(speakers), language)

    return full_text, transcript_data

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


# === Semantic Chunking (A/B-Experiment) ===
# Splittet Text an inhaltlichen Grenzen: zwischen aufeinanderfolgenden Sätzen wird
# die Embedding-Distanz berechnet; Sätze werden zu Chunks zusammengefasst bis eine
# "Bruchstelle" (Distanz im oberen Perzentil) oder die max. Länge erreicht ist.
# Fallback auf chunk_text() bei Fehlern oder zu wenig Daten.
_SENTENCE_BOUNDARY = re.compile(r'(?<=[\.\?!])\s+(?=[A-ZÄÖÜ0-9])')

def _split_sentences(text: str) -> list:
    """Einfache regex-basierte Satz-Segmentierung (DE/EN)."""
    text = text.strip()
    if not text:
        return []
    # Erst Absätze, dann Sätze pro Absatz – erhält strukturelle Grenzen
    out = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        parts = _SENTENCE_BOUNDARY.split(para)
        for p in parts:
            p = p.strip()
            if p:
                out.append(p)
    return out


def semantic_chunk_text(text: str, max_words: int = CHUNK_SIZE, min_words: int = 50,
                        breakpoint_percentile: int = 90, job_id=None):
    """Embedding-basierte Chunk-Splits. Fällt bei Fehlern auf chunk_text() zurück.

    breakpoint_percentile: Distanz-Perzentil, ab dem ein harter Split erzwungen wird
    (90 = nur die obersten 10% der Satz-Übergänge bilden Chunk-Grenzen).
    """
    try:
        import numpy as np

        sentences = _split_sentences(text)
        # Zu wenig Sätze → klassisches Chunking ist sinnvoller
        if len(sentences) < 5:
            return chunk_text(text)

        # Embeddings für alle Sätze (batched)
        sent_embeddings = embed(sentences, job_id=job_id)
        vecs = np.array(sent_embeddings, dtype=np.float32)
        # Cosine-Distanz zwischen aufeinanderfolgenden Sätzen
        norms = np.linalg.norm(vecs, axis=1)
        norms[norms == 0] = 1e-9
        unit = vecs / norms[:, None]
        sims = np.sum(unit[:-1] * unit[1:], axis=1)  # Länge = n-1
        distances = 1.0 - sims

        # Breakpoint-Threshold (z.B. oberstes 10%)
        threshold = float(np.percentile(distances, breakpoint_percentile)) if len(distances) > 0 else 1.0

        chunks = []
        current = []
        current_words = 0
        for i, sentence in enumerate(sentences):
            sent_words = len(sentence.split())
            current.append(sentence)
            current_words += sent_words

            is_last = (i == len(sentences) - 1)
            dist_here = distances[i] if i < len(distances) else None
            hard_break = (dist_here is not None and dist_here >= threshold)
            size_break = (current_words >= max_words)

            if is_last or hard_break or size_break:
                chunk_text_combined = " ".join(current).strip()
                if current_words < min_words and chunks:
                    # Zu kurz → mit vorherigem Chunk mergen
                    chunks[-1]["text"] = (chunks[-1]["text"] + " " + chunk_text_combined).strip()
                else:
                    chunks.append({"text": chunk_text_combined})
                current = []
                current_words = 0

        if not chunks:
            chunks = [{"text": text.strip()}]

        log.info("Semantic chunking: %d sentences → %d chunks (threshold=%.4f p%d)",
                 len(sentences), len(chunks), threshold, breakpoint_percentile)
        return chunks
    except Exception as e:
        log.warning("Job %s: semantic chunking failed (%s) – fallback to fixed", job_id, e)
        return chunk_text(text)

def embed_single(text):
    """Einzelnen Text für Search-Query embedden – gibt einen Vektor zurück."""
    vecs = embed([text], task_type="RETRIEVAL_QUERY")
    return vecs[0]


def embed(texts, task_type="RETRIEVAL_DOCUMENT", job_id=None):
    """Texte embedden – unterstützt OpenAI und Google Gemini Embedding 2.
    task_type wird nur für Google genutzt (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY).
    job_id: optional, für Usage-Tracking.
    """
    if EMBEDDING_PROVIDER == "google":
        return _embed_google(texts, task_type)
    return _embed_openai(texts, job_id=job_id)


def _embed_openai(texts, job_id=None):
    """OpenAI Embedding (text-embedding-3-small/large)."""
    all_emb = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        resp = oai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        all_emb.extend([d.embedding for d in resp.data])
        # Usage-Tracking: Embedding-Tokens akkumulieren
        if job_id and job_id in jobs:
            usage = jobs[job_id].setdefault("usage", {})
            usage["embedding_tokens"] = usage.get("embedding_tokens", 0) + (resp.usage.total_tokens or 0)
            usage["embedding_model"]  = EMBEDDING_MODEL
    return all_emb


def _embed_google(texts, task_type="RETRIEVAL_DOCUMENT"):
    """Google Gemini Embedding 2 via Vertex AI.
    task_type: RETRIEVAL_DOCUMENT für Chunks, RETRIEVAL_QUERY für Suchanfragen.
    """
    from google.genai import types
    import numpy as np

    all_emb = []
    # Google API: max 100 Texte pro Request
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        result = google_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
        for emb in result.embeddings:
            vec = emb.values
            # L2-Normalisierung für MRL truncated Dimensionen
            if EMBEDDING_DIMENSIONS < 3072:
                vec_np = np.array(vec)
                norm = np.linalg.norm(vec_np)
                if norm > 0:
                    vec = (vec_np / norm).tolist()
            all_emb.append(vec)
    return all_emb


def _embed_google_audio(audio_bytes: bytes, mime_type: str = "audio/mp3") -> list | None:
    """
    Audio-Bytes direkt via Gemini Embedding 2 embedden (ohne Transkription).
    Gibt einen einzelnen Vektor zurück oder None bei Fehler.
    Nur verfügbar wenn EMBEDDING_PROVIDER=google.
    """
    if not google_client or EMBEDDING_PROVIDER != "google":
        return None
    try:
        from google.genai import types
        import base64 as _b64
        encoded = _b64.b64encode(audio_bytes).decode("utf-8")
        part = types.Part.from_bytes(data=encoded, mime_type=mime_type)
        result = google_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[part],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
        emb = result.embeddings[0]
        vec = emb.values
        if EMBEDDING_DIMENSIONS < 3072:
            import numpy as np
            v = np.array(vec)
            norm = np.linalg.norm(v)
            if norm > 0:
                vec = (v / norm).tolist()
        return vec
    except Exception as e:
        log.warning("Audio-Embedding via Gemini fehlgeschlagen: %s", e)
        return None


# === Phase 7: Contextual Retrieval ===

CONTEXT_PROMPT = """Hier ist ein Dokument:
<document>
{doc_summary}
</document>

Hier ist ein Chunk aus diesem Dokument:
<chunk>
{chunk_text}
</chunk>

Gib einen kurzen, prägnanten Kontext (2-3 Sätze), der erklärt, wo dieser Chunk im Dokument eingeordnet ist und worum es darin geht. Der Kontext soll helfen, den Chunk bei einer Suchanfrage besser zu finden. Antworte NUR mit dem Kontext, ohne Einleitung."""


def contextualize_chunks(chunks, full_text, job_id=""):
    """Reichert jeden Chunk mit KI-generiertem Kontext an (Contextual Retrieval).
    Gibt Chunks mit 'context' und 'contextualized_text' Feldern zurück.
    """
    if not CONTEXTUAL_RETRIEVAL or not chunks:
        return chunks

    # Dokument-Zusammenfassung erstellen (max 2000 Zeichen für den Prompt)
    doc_summary = full_text[:2000]
    if len(full_text) > 2000:
        doc_summary += "..."

    chat_client = get_chat_client()

    for i, chunk in enumerate(chunks):
        try:
            response = chat_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{
                    "role": "user",
                    "content": CONTEXT_PROMPT.format(
                        doc_summary=doc_summary,
                        chunk_text=chunk["text"]
                    )
                }],
                temperature=0.0,
                max_tokens=150,
            )
            context = response.choices[0].message.content.strip()
            chunk["context"] = context
            chunk["contextualized_text"] = f"{context}\n\n{chunk['text']}"
            # Usage-Tracking: Contextual-Retrieval-Tokens akkumulieren
            if job_id and job_id in jobs:
                usage = jobs[job_id].setdefault("usage", {})
                usage["context_tokens_in"]  = usage.get("context_tokens_in", 0)  + (response.usage.prompt_tokens or 0)
                usage["context_tokens_out"] = usage.get("context_tokens_out", 0) + (response.usage.completion_tokens or 0)
                usage["context_model"]      = CHAT_MODEL
        except Exception as e:
            log.warning("Job %s: Kontext für Chunk %d fehlgeschlagen: %s", job_id, i, str(e))
            chunk["context"] = ""
            chunk["contextualized_text"] = chunk["text"]

    log.info("Job %s: %d/%d Chunks kontextualisiert", job_id,
             sum(1 for c in chunks if c.get("context")), len(chunks))
    return chunks


def notify(url, job_id, document_id, status, chunks, error="", transcript_data=None, usage=None):
    try:
        payload = {
            "job_id": job_id, "document_id": document_id,
            "status": status, "chunks": chunks, "error": error,
        }
        if transcript_data:
            payload["transcript"] = transcript_data
        if usage:
            payload["usage"] = usage
        with httpx.Client(timeout=30) as c:
            c.post(url, json=payload,
                headers={"Authorization": f"Bearer {DOC_PROCESSOR_SECRET}"})
    except Exception as e:
        log.error("Callback failed: %s", str(e))


# --- Research Pipeline (Scrape + AI Summary) ---

RESEARCH_SYSTEM_PROMPT = """Du bist ein Experte für die Erstellung von strukturierten Wissensdokumenten.
Deine Aufgabe ist es, aus dem gegebenen Website-Inhalt ein umfangreiches, strukturiertes Wissensdokument zu erstellen.

Das Dokument soll folgende Abschnitte enthalten (soweit Informationen vorhanden sind):

### Über das Unternehmen
Allgemeine Informationen, Geschichte, Mission, Werte

### Leistungen & Angebote
Alle Produkte, Dienstleistungen, Pakete detailliert beschreiben

### Öffnungszeiten
Tagesweise aufschlüsseln wenn vorhanden

### Kontaktdaten
Telefon, E-Mail, Adresse, Social Media

### Häufige Fragen (FAQ)
Typische Kundenfragen und Antworten ableiten

### Preise & Konditionen
Preislisten, Zahlungsbedingungen wenn vorhanden

### Besonderheiten & Alleinstellungsmerkmale
Was macht das Unternehmen besonders?

### Team & Ansprechpartner
Mitarbeiter, Geschäftsführung wenn erwähnt

### Wichtiges Wissen
Alles weitere Wichtige über das Unternehmen

REGELN:
- Verwende Markdown-Formatierung mit ### für Hauptabschnitte
- Schreibe in der gleichen Sprache wie der Website-Inhalt
- Sei so detailliert und umfangreich wie möglich
- Formuliere Informationen klar und präzise, sodass sie direkt als Wissensquelle verwendet werden können
- Lasse Abschnitte weg, für die keine Informationen vorhanden sind
- Erfinde KEINE Informationen – verwende nur was auf der Website steht
- Maximal 8000 Zeichen"""

def research_pipeline(job_id, url, callback_url):
    """URL(s) scrapen, AI-Zusammenfassung erstellen, Ergebnis per Callback senden."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["step"] = "fetching"
        url = normalize_url(url)
        log.info("Research %s: fetching %s", job_id, url)

        # Hauptseite laden
        with httpx.Client(timeout=URL_TIMEOUT, follow_redirects=True) as c:
            r = c.get(url, headers={"User-Agent": URL_USER_AGENT})
            r.raise_for_status()
            main_html = r.text

        main_text, main_title, _main_html = fetch_page(url)
        all_texts = []
        if main_text and len(main_text.strip()) > 10:
            all_texts.append(f"=== {main_title} – {url} ===\n{main_text}")

        # Bei Domain-Root: Unterseiten crawlen
        if is_domain_root(url):
            jobs[job_id]["step"] = "crawling"
            links = discover_links(main_html, url)
            log.info("Research %s: found %d links to crawl", job_id, len(links))
            for i, link in enumerate(links):
                try:
                    time.sleep(CRAWL_DELAY)
                    page_text, page_title, _page_html = fetch_page(link)
                    if page_text and len(page_text.strip()) > 10:
                        all_texts.append(f"=== {page_title} – {link} ===\n{page_text}")
                        log.info("Research %s: crawled %d/%d – %s (%d chars)",
                                 job_id, i + 1, len(links), link, len(page_text))
                except Exception as ex:
                    log.warning("Research %s: failed to fetch %s: %s", job_id, link, str(ex))

        combined_text = "\n\n".join(all_texts)
        if not combined_text or len(combined_text.strip()) < 10:
            raise Exception(f"Keine Inhalte von {url} extrahiert")

        log.info("Research %s: %d chars total from %d pages", job_id, len(combined_text), len(all_texts))

        # AI-Zusammenfassung erstellen
        jobs[job_id]["step"] = "summarizing"
        # Text auf max. 80000 Zeichen begrenzen für API
        input_text = combined_text[:80000]

        chat_client = get_chat_client()
        response = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": f"Erstelle ein umfangreiches Wissensdokument basierend auf folgendem Website-Inhalt:\n\n{input_text}"}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        result_text = response.choices[0].message.content.strip()
        log.info("Research %s: AI summary generated, %d chars", job_id, len(result_text))

        jobs[job_id]["status"] = "ready"
        jobs[job_id]["step"] = "done"
        jobs[job_id]["result_text"] = result_text

        if callback_url:
            notify_research(callback_url, job_id, "ready", result_text)

    except Exception as e:
        log.error("Research %s failed: %s", job_id, str(e))
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        if callback_url:
            notify_research(callback_url, job_id, "failed", "", str(e))


def notify_research(url, job_id, status, result_text="", error=""):
    try:
        with httpx.Client(timeout=30) as c:
            c.post(url, json={
                "job_id": job_id,
                "status": status,
                "result_text": result_text,
                "error": error,
            }, headers={"Authorization": f"Bearer {DOC_PROCESSOR_SECRET}"})
    except Exception as e:
        log.error("Research callback failed: %s", str(e))

