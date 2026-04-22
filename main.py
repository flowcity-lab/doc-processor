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

# === Embedding Configuration (OpenAI) ===
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "1536"))

# OpenAI API Key (für Embeddings und Chat/Contextual Retrieval)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Chat-Modell für Contextual Retrieval & Research
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4.1-mini")


def get_openai_client():
    """Erstellt OpenAI-Client für Embeddings und Chat."""
    return openai.OpenAI(api_key=OPENAI_API_KEY)


def get_chat_client():
    """Erstellt OpenAI-Client für Chat-Completions (Contextual Retrieval, Research)."""
    return openai.OpenAI(api_key=OPENAI_API_KEY)


COLLECTION = "notebook_documents"
VECTOR_DIM = EMBEDDING_DIMENSIONS

# === RAG 3.0 – Chunk-Größen (ADR-0003) ===
# Child-Chunks: kleine, präzise Einheiten für den Vektor-Recall.
# Parent-Windows: größerer Kontext, der dem LLM beim Antworten geliefert wird.
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "250"))            # Child-Chunk-Wörter
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))
PARENT_WINDOW_WORDS = int(os.environ.get("PARENT_WINDOW_WORDS", "1000"))  # Parent-Kontext-Größe

# === Phase 7: RAG Enhancement Config ===
CONTEXTUAL_RETRIEVAL = os.environ.get("CONTEXTUAL_RETRIEVAL", "true").lower() == "true"
# ADR-0003: Full-Doc-Contextual-Retrieval mit GPT-4.1 1M-Context.
CONTEXTUAL_FULL_DOC = os.environ.get("CONTEXTUAL_FULL_DOC", "true").lower() == "true"
CONTEXTUAL_MAX_DOC_CHARS = int(os.environ.get("CONTEXTUAL_MAX_DOC_CHARS", "800000"))  # ~200k Tokens

# ADR-0003: BGE-Reranker-v2-M3 mit Sigmoid-Normalisierung auf [0,1].
CROSS_ENCODER_MODEL = os.environ.get("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_DEVICE = os.environ.get("RERANKER_DEVICE", "cpu")
RERANKER_SIGMOID = os.environ.get("RERANKER_SIGMOID", "true").lower() == "true"
RERANKER_SCORE_FLOOR = float(os.environ.get("RERANKER_SCORE_FLOOR", "0.0"))  # 0 = deaktiviert

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

def _build_qdrant_client() -> QdrantClient:
    """
    qdrant-client fällt bei URLs ohne expliziten Port auf 6333 zurück — auch bei https.
    Hinter einem Reverse-Proxy (443/80) führt das zu Connection refused. Deshalb hier
    Port aus URL parsen und sonst Scheme-Default (443/80) verwenden.
    """
    parsed = urlparse(QDRANT_HOST)
    if parsed.scheme in ("http", "https") and parsed.hostname and not parsed.port:
        return QdrantClient(
            host=parsed.hostname,
            port=443 if parsed.scheme == "https" else 80,
            https=parsed.scheme == "https",
            api_key=QDRANT_API_KEY,
            timeout=60,
        )
    return QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY, timeout=60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant, oai, cross_encoder
    qdrant = _build_qdrant_client()
    oai = get_openai_client()

    log.info("Embedding: OpenAI %s, Dimensions: %d", EMBEDDING_MODEL, VECTOR_DIM)

    # Cross-Encoder für Re-Ranking laden (BGE-Reranker-v2-M3, ADR-0003)
    try:
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=RERANKER_DEVICE)
        log.info("Cross-Encoder geladen: %s (device=%s, sigmoid=%s)",
                 CROSS_ENCODER_MODEL, RERANKER_DEVICE, RERANKER_SIGMOID)
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
            "provider": "openai",
            "model": EMBEDDING_MODEL,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/embed-texts")
async def embed_texts(request: Request, authorization: str = Header(None)):
    """Embeddet eine Liste kurzer Texte (z.B. Memory-Contents) und gibt Vektoren zurück.
    Body: {"texts": ["...", "..."]}
    """
    auth(authorization)
    try:
        body = await request.json()
        texts = body.get("texts", [])
        if not isinstance(texts, list) or not texts:
            return {"ok": False, "error": "texts muss ein nicht-leeres Array sein"}
        if len(texts) > 200:
            return {"ok": False, "error": "max 200 Texte pro Request"}
        vecs = embed(texts)
        return {
            "ok": True,
            "vectors": vecs,
            "dimensions": len(vecs[0]) if vecs else 0,
            "provider": "openai",
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
    jobs[job_id] = {
        "job_id": job_id, "status": "queued", "step": "",
        "document_id": document_id, "notebook_id": notebook_id,
        "filename": file.filename, "chunks": 0, "error": None,
        "pdf_mode": pdf_mode, "chunking_strategy": strategy, "started_at": time.time()
    }
    bg.add_task(pipeline, job_id, content, file.filename or "", document_id, notebook_id,
                callback_url, pdf_mode, parsed_transcript, strategy)
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
    authorization: str = Header(None)
):
    auth(authorization)
    job_id = str(uuid.uuid4())
    strategy = chunking_strategy if chunking_strategy in ("fixed", "semantic") else "fixed"
    jobs[job_id] = {
        "job_id": job_id, "status": "queued", "step": "",
        "document_id": document_id, "notebook_id": notebook_id,
        "filename": url, "chunks": 0, "error": None,
        "pdf_mode": pdf_mode, "chunking_strategy": strategy, "started_at": time.time()
    }
    bg.add_task(url_pipeline, job_id, url, document_id, notebook_id, callback_url, pdf_mode, strategy)
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
    hypo_query: str = Form(""),  # RAG 3.0 – HyDE hypothetical answer (ADR-0003)
    score_floor: float = Form(0.0),  # Reranker-Score-Floor, 0 = deaktiviert
    authorization: str = Header(None)
):
    auth(authorization)
    qvec = embed_single(query)
    # HyDE: hypothetische Antwort separat embedden, später via RRF fusen
    hypo_vec = None
    if hypo_query and hypo_query.strip():
        try:
            hypo_vec = embed_single(hypo_query.strip())
        except Exception as e:
            log.warning("HyDE-Embedding fehlgeschlagen: %s", e)

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

    # 1b. HyDE-Vektor-Suche (parallel) – eigene Kandidaten-Liste
    hypo_results_points = []
    if hypo_vec is not None:
        try:
            hypo_hits = qdrant.query_points(
                collection_name=COLLECTION, query=hypo_vec,
                query_filter=Filter(must=must), limit=fetch_k, with_payload=True
            )
            hypo_results_points = hypo_hits.points
        except Exception as e:
            log.warning("HyDE-Suche fehlgeschlagen: %s", e)

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

    # 3. Ergebnisse mergen (Reciprocal Rank Fusion)
    scored = {}  # point_id -> {payload, rrf_score}
    # Vector-Ergebnisse
    for rank, r in enumerate(vector_results.points):
        pid = str(r.id)
        scored[pid] = {
            "payload": r.payload,
            "rrf_score": 1.0 / (60 + rank),
            "vector_score": r.score,
        }
    # HyDE-Vector-Ergebnisse dazumischen (RAG 3.0)
    for rank, r in enumerate(hypo_results_points):
        pid = str(r.id)
        rrf_add = 1.0 / (60 + rank)
        if pid in scored:
            scored[pid]["rrf_score"] += rrf_add
        else:
            scored[pid] = {
                "payload": r.payload,
                "rrf_score": rrf_add,
                "vector_score": getattr(r, "score", 0.0) or 0.0,
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

    # 4. Re-Ranking mit Cross-Encoder (BGE-Reranker-v2-M3 + Sigmoid, ADR-0003)
    if cross_encoder and len(candidates) > 0:
        pairs = [(query, c["payload"].get("original_text", c["payload"].get("text", ""))) for c in candidates]
        try:
            ce_scores = cross_encoder.predict(pairs)
            if RERANKER_SIGMOID:
                import math
                ce_scores = [1.0 / (1.0 + math.exp(-float(s))) for s in ce_scores]
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

    # Score-Floor (nur bei aktivem, sigmoidiertem Reranker sinnvoll)
    effective_floor = score_floor if score_floor > 0 else RERANKER_SCORE_FLOOR
    if cross_encoder and RERANKER_SIGMOID and effective_floor > 0:
        before = len(candidates)
        candidates = [c for c in candidates if c["final_score"] >= effective_floor]
        if before != len(candidates):
            log.info("Score-Floor %.2f: %d → %d Kandidaten", effective_floor, before, len(candidates))

    # Top-K zurückgeben
    final = candidates[:top_k]
    out = []
    for c in final:
        p = c["payload"]
        item = {
            "text":        p.get("original_text", p.get("text", "")),
            "parent_text": p.get("parent_text", p.get("original_text", p.get("text", ""))),
            "document_id": p.get("document_id", ""),
            "chunk_index": p.get("chunk_index", 0),
            "filename":    p.get("filename", ""),
            "score":       c["final_score"],
        }
        if p.get("page_num"):
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

@app.delete("/collections/reset")
async def reset_collection(authorization: str = Header(None)):
    """Droppt die Qdrant-Collection komplett und legt sie mit aktueller VECTOR_DIM neu an.
    Wird vom Admin-UI nach Embedding-Dim-Wechsel ausgelöst. Alle Dokumente müssen danach
    vom Laravel-Hub neu zum Indexieren eingereiht werden."""
    auth(authorization)
    try:
        if qdrant.collection_exists(COLLECTION):
            qdrant.delete_collection(COLLECTION)
            log.warning("Collection %s per Admin-Request gedroppt", COLLECTION)
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        qdrant.create_payload_index(COLLECTION, "notebook_id", "keyword")
        qdrant.create_payload_index(COLLECTION, "document_id", "keyword")
        log.info("Collection %s neu erstellt mit dim=%d", COLLECTION, VECTOR_DIM)
        return {"status": "reset", "collection": COLLECTION, "dimensions": VECTOR_DIM}
    except Exception as e:
        log.exception("Collection-Reset fehlgeschlagen")
        raise HTTPException(500, f"Reset failed: {e}")

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
             existing_transcript=None, chunking_strategy="fixed"):
    """
    pdf_mode:
        'basic'              – nur Text via Tika (Standard)
        'vision_description' – Bilder via Vision-LLM beschreiben
    existing_transcript: vorhandenes Transkript-Dict (spart Re-Transkription beim Reindex)
    chunking_strategy:
        'fixed'    – feste Wort-Anzahl pro Chunk (Standard)
        'semantic' – Embedding-basierte Splits an inhaltlichen Grenzen
    """
    try:
        jobs[job_id]["status"] = "processing"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        transcript_data = None
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
            if ext_lower == "pdf" and pdf_mode == "vision_description":
                jobs[job_id]["step"] = "extracting (vision)"
                text = extract_pdf_with_vision(content, job_id)
            else:
                text = extract(content)
        has_text = bool(text and len(text.strip()) >= 10)
        if not has_text:
            raise Exception(f"No text extracted from {filename}")

        # Einmalig alle bisherigen Punkte dieses Dokuments löschen (für Re-Index)
        qdrant.delete(
            collection_name=COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
            )
        )

        log.info("Job %s: %d chars from %s", job_id, len(text), filename)
        jobs[job_id]["step"] = "chunking"
        if chunking_strategy == "semantic":
            chunks = semantic_chunk_text(text, job_id=job_id)
        else:
            chunks = chunk_text(text)
        log.info("Job %s: %d chunks (strategy=%s)", job_id, len(chunks), chunking_strategy)

        # RAG 3.0 – Parent-Windows an jeden Child-Chunk anhängen (ADR-0003)
        chunks = attach_parent_windows(chunks, text, window_words=PARENT_WINDOW_WORDS)

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
                "parent_text": c.get("parent_text", c["text"]),
                "context": c.get("context", ""),
                "notebook_id": notebook_id,
                "document_id": document_id, "filename": filename, "chunk_index": i
            })
            for i, (c, emb) in enumerate(zip(chunks, embeddings))
        ]
        for i in range(0, len(points), 100):
            qdrant.upsert(collection_name=COLLECTION, points=points[i:i+100])

        total_chunks = len(chunks)
        jobs[job_id]["status"] = "ready"
        jobs[job_id]["step"] = "done"
        jobs[job_id]["chunks"] = total_chunks
        if transcript_data:
            jobs[job_id]["transcript"] = transcript_data
        log.info("Job %s: done, %d text chunks stored", job_id, len(chunks))
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
                 chunking_strategy="fixed"):
    """URL(s) laden, Text extrahieren, chunken, embedden, speichern.
    chunking_strategy: 'fixed' (Default) oder 'semantic' (Embedding-basierte Splits).
    """
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["step"] = "fetching"
        url = normalize_url(url)
        log.info("Job %s: fetching %s", job_id, url)

        main_text, main_title, main_html = fetch_page(url)
        all_texts = []
        if main_text and len(main_text.strip()) > 10:
            all_texts.append(f"=== {main_title} – {url} ===\n{main_text}")

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
                except Exception as ex:
                    log.warning("Job %s: failed to fetch %s: %s", job_id, link, str(ex))

        combined_text = "\n\n".join(all_texts)
        has_text = bool(combined_text and len(combined_text.strip()) >= 10)
        if not has_text:
            raise Exception(f"No text extracted from {url}")

        log.info("Job %s: %d chars total from %d pages", job_id,
                 len(combined_text), len(all_texts))

        # Punkte dieses Dokuments entfernen (Re-Index-freundlich)
        qdrant.delete(
            collection_name=COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
            )
        )

        jobs[job_id]["step"] = "chunking"
        if chunking_strategy == "semantic":
            chunks = semantic_chunk_text(combined_text, job_id=job_id)
        else:
            chunks = chunk_text(combined_text)
        log.info("Job %s: %d chunks (strategy=%s)", job_id, len(chunks), chunking_strategy)

        # RAG 3.0 – Parent-Windows (ADR-0003)
        chunks = attach_parent_windows(chunks, combined_text, window_words=PARENT_WINDOW_WORDS)

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
                "parent_text": c.get("parent_text", c["text"]),
                "context": c.get("context", ""),
                "notebook_id": notebook_id,
                "document_id": document_id, "filename": url, "chunk_index": i
            })
            for i, (c, emb) in enumerate(zip(chunks, embeddings))
        ]
        for i in range(0, len(points), 100):
            qdrant.upsert(collection_name=COLLECTION, points=points[i:i + 100])

        total = len(chunks)
        jobs[job_id]["status"] = "ready"
        jobs[job_id]["step"] = "done"
        jobs[job_id]["chunks"] = total
        log.info("Job %s: done, %d Text-Chunks gespeichert", job_id, len(chunks))
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


def attach_parent_windows(chunks, full_text, window_words=PARENT_WINDOW_WORDS):
    """RAG 3.0 – Parent-Chunk-Retrieval (ADR-0003).

    Berechnet für jeden Child-Chunk ein größeres Parent-Fenster im Original-Text
    und hängt es als `parent_text` an. Recall läuft weiterhin auf den kleinen
    Child-Embeddings, das LLM sieht aber beim Antworten den breiteren Kontext.

    Fallback: Wenn der Chunk-Text im Original nicht gefunden wird (z.B. bei
    semantischem Chunking mit Normalisierung), bleibt parent_text = chunk.text.
    """
    if not chunks or not full_text or window_words <= 0:
        for c in chunks:
            c.setdefault("parent_text", c["text"])
        return chunks

    full_words = full_text.split()
    total_words = len(full_words)

    # Schneller Lookup: Chunk-Text → Start-Position (First 6 Wörter)
    for c in chunks:
        chunk_words = c["text"].split()
        if not chunk_words:
            c["parent_text"] = c["text"]
            continue

        needle = " ".join(chunk_words[:6]).lower()
        haystack_lower = " ".join(full_words).lower()
        pos_char = haystack_lower.find(needle)
        if pos_char < 0:
            c["parent_text"] = c["text"]
            continue

        # Char → Word-Index
        word_idx = haystack_lower.count(" ", 0, pos_char)
        chunk_len = len(chunk_words)
        pad = max(0, (window_words - chunk_len) // 2)
        start = max(0, word_idx - pad)
        end = min(total_words, word_idx + chunk_len + pad)
        c["parent_text"] = " ".join(full_words[start:end])
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
    vecs = embed([text])
    return vecs[0]


def embed(texts, job_id=None, **_ignored):
    """Texte via OpenAI Embedding (text-embedding-3-small/large) embedden.
    job_id: optional, für Usage-Tracking. Zusätzliche Keyword-Argumente werden ignoriert
    (Abwärtskompatibilität mit alten Call-Sites, die task_type übergeben haben).
    """
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

    RAG 3.0 (ADR-0003):
      - Bei CONTEXTUAL_FULL_DOC=true wird das gesamte Dokument (bis zum
        CONTEXTUAL_MAX_DOC_CHARS-Cap) in den Prompt gelegt. CHAT_MODEL muss ein
        1M-Context-Modell sein (gpt-4.1-mini).
      - Sonst Fallback auf 2000-Zeichen-Summary.

    Gibt Chunks mit 'context' und 'contextualized_text' Feldern zurück.
    """
    if not CONTEXTUAL_RETRIEVAL or not chunks:
        return chunks

    # Dokument-Kontext-Fenster für den Prompt bestimmen
    if CONTEXTUAL_FULL_DOC:
        doc_for_prompt = full_text[:CONTEXTUAL_MAX_DOC_CHARS]
        if len(full_text) > CONTEXTUAL_MAX_DOC_CHARS:
            doc_for_prompt += "\n...[gekürzt]"
            log.info("Job %s: Dokument auf %d Zeichen gekappt (full_text war %d)",
                     job_id, CONTEXTUAL_MAX_DOC_CHARS, len(full_text))
    else:
        doc_for_prompt = full_text[:2000]
        if len(full_text) > 2000:
            doc_for_prompt += "..."

    chat_client = get_chat_client()

    for i, chunk in enumerate(chunks):
        try:
            response = chat_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{
                    "role": "user",
                    "content": CONTEXT_PROMPT.format(
                        doc_summary=doc_for_prompt,
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

    log.info("Job %s: %d/%d Chunks kontextualisiert (full_doc=%s, doc_chars=%d)",
             job_id, sum(1 for c in chunks if c.get("context")), len(chunks),
             CONTEXTUAL_FULL_DOC, len(doc_for_prompt))
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

