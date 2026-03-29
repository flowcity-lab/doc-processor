import os, uuid, time, io, logging, re, asyncio, subprocess, tempfile
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import urlparse, urljoin
import httpx, openai
import trafilatura
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, BackgroundTasks
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
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# === AI Provider Switch ===
# AI_PROVIDER=openai  → OpenAI direkt (Dev/Test, 1 API Key)
# AI_PROVIDER=azure   → Azure OpenAI (Produktion, DSGVO-konform)
AI_PROVIDER = os.environ.get("AI_PROVIDER", "openai")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Azure OpenAI (nur wenn AI_PROVIDER=azure)
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT", "")
AZURE_OPENAI_EMBEDDING_API_KEY = os.environ.get("AZURE_OPENAI_EMBEDDING_API_KEY", "")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
AZURE_OPENAI_CHAT_ENDPOINT = os.environ.get("AZURE_OPENAI_CHAT_ENDPOINT", "")
AZURE_OPENAI_CHAT_API_KEY = os.environ.get("AZURE_OPENAI_CHAT_API_KEY", "")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")


def get_openai_client():
    """Erstellt OpenAI-Client basierend auf AI_PROVIDER Umgebungsvariable."""
    if AI_PROVIDER == "azure":
        from openai import AzureOpenAI
        # Für Embeddings den Embedding-Endpoint nutzen, für Chat den Chat-Endpoint.
        # Da wir einen globalen Client brauchen, nehmen wir den Embedding-Endpoint
        # (Hauptaufgabe des doc-processors). Für Chat-Calls wird ggf. ein separater Client erstellt.
        return AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
            api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
    else:
        return openai.OpenAI(api_key=OPENAI_API_KEY)


def get_chat_client():
    """Erstellt OpenAI-Client für Chat-Completions (z.B. Research-Zusammenfassung)."""
    if AI_PROVIDER == "azure":
        from openai import AzureOpenAI
        return AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
            api_key=AZURE_OPENAI_CHAT_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
    else:
        return openai.OpenAI(api_key=OPENAI_API_KEY)
COLLECTION = "notebook_documents"
VECTOR_DIM = 1536
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
    global qdrant, oai, cross_encoder
    qdrant = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY, timeout=60)
    oai = get_openai_client()
    log.info("AI Provider: %s", AI_PROVIDER)

    # Cross-Encoder für Re-Ranking laden
    try:
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        log.info("Cross-Encoder geladen: %s", CROSS_ENCODER_MODEL)
    except Exception as e:
        log.warning("Cross-Encoder konnte nicht geladen werden: %s — Re-Ranking deaktiviert", str(e))
        cross_encoder = None

    if not qdrant.collection_exists(COLLECTION):
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

@app.post("/process-url")
async def process_url(
    bg: BackgroundTasks,
    url: str = Form(...),
    document_id: str = Form(...),
    notebook_id: str = Form(...),
    callback_url: str = Form(""),
    authorization: str = Header(None)
):
    auth(authorization)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id, "status": "queued", "step": "",
        "document_id": document_id, "notebook_id": notebook_id,
        "filename": url, "chunks": 0, "error": None,
        "started_at": time.time()
    }
    bg.add_task(url_pipeline, job_id, url, document_id, notebook_id, callback_url)
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
    embed_model = AZURE_OPENAI_EMBEDDING_DEPLOYMENT if AI_PROVIDER == "azure" else EMBEDDING_MODEL
    resp = oai.embeddings.create(model=embed_model, input=query)
    qvec = resp.data[0].embedding
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

    # 3. Ergebnisse mergen (Reciprocal Rank Fusion)
    scored = {}  # point_id -> {payload, rrf_score}
    # Vector-Ergebnisse
    for rank, r in enumerate(vector_results.points):
        pid = str(r.id)
        scored[pid] = {
            "payload": r.payload,
            "rrf_score": 1.0 / (60 + rank),  # RRF mit k=60
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

    # Top-K zurückgeben
    final = candidates[:top_k]
    return {"results": [
        {
            "text": c["payload"].get("original_text", c["payload"].get("text", "")),
            "document_id": c["payload"].get("document_id", ""),
            "chunk_index": c["payload"].get("chunk_index", 0),
            "filename": c["payload"].get("filename", ""),
            "score": c["final_score"],
        }
        for c in final
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

def pipeline(job_id, content, filename, document_id, notebook_id, callback_url):
    try:
        jobs[job_id]["status"] = "processing"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        transcript_data = None
        if ext in AUDIO_EXTENSIONS:
            jobs[job_id]["step"] = "preparing audio"
            content, filename = prepare_audio(content, filename)
            jobs[job_id]["step"] = "transcribing"
            text, transcript_data = transcribe(content, filename)
        else:
            jobs[job_id]["step"] = "extracting"
            text = extract(content)
        if not text or len(text.strip()) < 10:
            raise Exception(f"No text extracted from {filename}")
        log.info("Job %s: %d chars from %s", job_id, len(text), filename)
        jobs[job_id]["step"] = "chunking"
        chunks = chunk_text(text)
        log.info("Job %s: %d chunks", job_id, len(chunks))

        # Phase 7: Contextual Retrieval — Chunks mit KI-Kontext anreichern
        if CONTEXTUAL_RETRIEVAL:
            jobs[job_id]["step"] = "contextualizing"
            chunks = contextualize_chunks(chunks, text, job_id)

        jobs[job_id]["step"] = "embedding"
        # Embedding auf kontextualisiertem Text (falls vorhanden)
        embed_texts = [c.get("contextualized_text", c["text"]) for c in chunks]
        embeddings = embed(embed_texts)
        qdrant.delete(
            collection_name=COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
            )
        )
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
        jobs[job_id]["status"] = "ready"
        jobs[job_id]["step"] = "done"
        jobs[job_id]["chunks"] = len(chunks)
        if transcript_data:
            jobs[job_id]["transcript"] = transcript_data
        log.info("Job %s: done, %d chunks stored", job_id, len(chunks))
        if callback_url:
            notify(callback_url, job_id, document_id, "ready", len(chunks), transcript_data=transcript_data)
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
    """Einzelne Seite laden und Text mit trafilatura extrahieren."""
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

    return text or "", page_title

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

def url_pipeline(job_id, url, document_id, notebook_id, callback_url):
    """URL(s) laden, Text extrahieren, chunken, embedden, speichern."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["step"] = "fetching"
        url = normalize_url(url)
        log.info("Job %s: fetching %s", job_id, url)

        # Hauptseite laden
        with httpx.Client(timeout=URL_TIMEOUT, follow_redirects=True) as c:
            r = c.get(url, headers={"User-Agent": URL_USER_AGENT})
            r.raise_for_status()
            main_html = r.text

        main_text, main_title = fetch_page(url)
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
                    page_text, page_title = fetch_page(link)
                    if page_text and len(page_text.strip()) > 10:
                        all_texts.append(f"=== {page_title} – {link} ===\n{page_text}")
                        log.info("Job %s: crawled %d/%d – %s (%d chars)",
                                 job_id, i + 1, len(links), link, len(page_text))
                    else:
                        log.info("Job %s: skipped %s (no content)", job_id, link)
                except Exception as ex:
                    log.warning("Job %s: failed to fetch %s: %s", job_id, link, str(ex))

        combined_text = "\n\n".join(all_texts)
        if not combined_text or len(combined_text.strip()) < 10:
            raise Exception(f"No text extracted from {url}")

        log.info("Job %s: %d chars total from %d pages", job_id, len(combined_text), len(all_texts))

        # Ab hier: gleiche Pipeline wie Datei-Upload
        jobs[job_id]["step"] = "chunking"
        chunks = chunk_text(combined_text)
        log.info("Job %s: %d chunks", job_id, len(chunks))

        # Phase 7: Contextual Retrieval
        if CONTEXTUAL_RETRIEVAL:
            jobs[job_id]["step"] = "contextualizing"
            chunks = contextualize_chunks(chunks, combined_text, job_id)

        jobs[job_id]["step"] = "embedding"
        embed_texts = [c.get("contextualized_text", c["text"]) for c in chunks]
        embeddings = embed(embed_texts)

        qdrant.delete(
            collection_name=COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
            )
        )

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

        jobs[job_id]["status"] = "ready"
        jobs[job_id]["step"] = "done"
        jobs[job_id]["chunks"] = len(chunks)
        log.info("Job %s: done, %d chunks stored from URL", job_id, len(chunks))
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

def embed(texts):
    embed_model = AZURE_OPENAI_EMBEDDING_DEPLOYMENT if AI_PROVIDER == "azure" else EMBEDDING_MODEL
    all_emb = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        resp = oai.embeddings.create(model=embed_model, input=batch)
        all_emb.extend([d.embedding for d in resp.data])
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
    Gibt Chunks mit 'context' und 'contextualized_text' Feldern zurück.
    """
    if not CONTEXTUAL_RETRIEVAL or not chunks:
        return chunks

    # Dokument-Zusammenfassung erstellen (max 2000 Zeichen für den Prompt)
    doc_summary = full_text[:2000]
    if len(full_text) > 2000:
        doc_summary += "..."

    chat_client = get_chat_client()
    chat_model = AZURE_OPENAI_CHAT_DEPLOYMENT if AI_PROVIDER == "azure" else "gpt-4.1-mini"

    for i, chunk in enumerate(chunks):
        try:
            response = chat_client.chat.completions.create(
                model=chat_model,
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
        except Exception as e:
            log.warning("Job %s: Kontext für Chunk %d fehlgeschlagen: %s", job_id, i, str(e))
            chunk["context"] = ""
            chunk["contextualized_text"] = chunk["text"]

    log.info("Job %s: %d/%d Chunks kontextualisiert", job_id,
             sum(1 for c in chunks if c.get("context")), len(chunks))
    return chunks


def notify(url, job_id, document_id, status, chunks, error="", transcript_data=None):
    try:
        payload = {
            "job_id": job_id, "document_id": document_id,
            "status": status, "chunks": chunks, "error": error,
        }
        if transcript_data:
            payload["transcript"] = transcript_data
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

        main_text, main_title = fetch_page(url)
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
                    page_text, page_title = fetch_page(link)
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
        chat_model = AZURE_OPENAI_CHAT_DEPLOYMENT if AI_PROVIDER == "azure" else "gpt-4.1-mini"
        response = chat_client.chat.completions.create(
            model=chat_model,
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

