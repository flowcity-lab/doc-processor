import os, uuid, time, io, logging, re, asyncio
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import urlparse, urljoin
import httpx, openai
import trafilatura
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

# === Audio Transcription Config ===
TRANSCRIPTION_PROVIDER = os.environ.get("TRANSCRIPTION_PROVIDER", "assemblyai")  # assemblyai | whisper
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
ASSEMBLYAI_SPEECH_MODEL = os.environ.get("ASSEMBLYAI_SPEECH_MODEL", "")  # leer = default (universal-3-pro)
ASSEMBLYAI_SPEAKER_LABELS = os.environ.get("ASSEMBLYAI_SPEAKER_LABELS", "true").lower() == "true"
ASSEMBLYAI_LANGUAGE_DETECTION = os.environ.get("ASSEMBLYAI_LANGUAGE_DETECTION", "true").lower() == "true"
ASSEMBLYAI_LANGUAGE = os.environ.get("ASSEMBLYAI_LANGUAGE", "")  # leer = auto-detect

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
        transcript_data = None
        if ext in AUDIO_EXTENSIONS:
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

def fetch_page(url):
    """Einzelne Seite laden und Text mit trafilatura extrahieren."""
    with httpx.Client(timeout=URL_TIMEOUT, follow_redirects=True) as c:
        r = c.get(url, headers={"User-Agent": URL_USER_AGENT})
        r.raise_for_status()
        html = r.text
    text = trafilatura.extract(html, include_comments=False, include_tables=True)
    title = trafilatura.extract(html, output_format="xml")
    # Title aus HTML extrahieren
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    page_title = title_match.group(1).strip() if title_match else url
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
                "document_id": document_id, "filename": url, "chunk_index": i
            })
            for i, (ch, emb) in enumerate(zip(chunks, embeddings))
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
    if TRANSCRIPTION_PROVIDER == "assemblyai":
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
    speech_model = ASSEMBLYAI_SPEECH_MODEL if ASSEMBLYAI_SPEECH_MODEL else "universal-3-pro"
    transcript_config = {
        "audio_url": audio_url,
        "speaker_labels": ASSEMBLYAI_SPEAKER_LABELS,
        "speech_models": [speech_model],
    }
    if ASSEMBLYAI_LANGUAGE_DETECTION:
        transcript_config["language_detection"] = True
    if ASSEMBLYAI_LANGUAGE:
        transcript_config["language_code"] = ASSEMBLYAI_LANGUAGE
        transcript_config.pop("language_detection", None)

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
    all_emb = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        resp = oai.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_emb.extend([d.embedding for d in resp.data])
    return all_emb

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
                headers={"Authorization": f"Bearer {API_SECRET}"})
    except Exception as e:
        log.error("Callback failed: %s", str(e))


# --- Research Pipeline (Scrape + AI Summary) ---

RESEARCH_SYSTEM_PROMPT = """Du bist ein Experte für die Erstellung von Wissensdokumenten für KI-Telefonassistenten.
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
Alles weitere das ein Telefonassistent wissen sollte

REGELN:
- Verwende Markdown-Formatierung mit ### für Hauptabschnitte
- Schreibe in der gleichen Sprache wie der Website-Inhalt
- Sei so detailliert und umfangreich wie möglich
- Formuliere Informationen so, dass ein KI-Telefonassistent sie direkt verwenden kann
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

        response = oai.chat.completions.create(
            model="gpt-4.1-mini",
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
            }, headers={"Authorization": f"Bearer {API_SECRET}"})
    except Exception as e:
        log.error("Research callback failed: %s", str(e))

