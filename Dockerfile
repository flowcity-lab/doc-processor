FROM python:3.12-slim

# ffmpeg     — Audio-/Video-Transcoding fuer Transkriptions-Endpoints
# libreoffice + fonts — DOCX/PDF-Rendering fuer den VGSE-Invoice-Pipeline
#                       (/docx/render-pages, /docx/inspect-fonts) sowie
#                       weitere Office-Konversionen. fontconfig + ein breiter
#                       Font-Satz vermeiden Layout-Drift, falls Trainer-Vorlagen
#                       Schriften referenzieren, die nicht eingebettet sind.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libreoffice \
        libreoffice-writer \
        fontconfig \
        fonts-dejavu \
        fonts-liberation \
        fonts-liberation2 \
        fonts-noto-core \
        fonts-noto-cjk \
        fonts-noto-color-emoji \
        fonts-roboto \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY template_engine/ ./template_engine/
COPY docx_render.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

