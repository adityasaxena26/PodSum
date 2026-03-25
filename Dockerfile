# =============================================================================
# Podsum v2.0 - Production Dockerfile
# =============================================================================
# Layers ordered by change frequency (least → most) for optimal cache reuse.
# torch is intentionally excluded: faster-whisper uses ctranslate2 (not PyTorch)
# and the app already handles ImportError gracefully (falls back to CPU).
# =============================================================================

FROM python:3.11-slim

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Non-root user
# ---------------------------------------------------------------------------
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ---------------------------------------------------------------------------
# 3. Python dependencies (pinned to known-good versions)
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    "youtube-transcript-api==1.2.4" \
    "yt-dlp==2026.2.4" \
    "faster-whisper==1.2.1" \
    "google-genai==1.65.0" \
    "groq==1.0.0" \
    "python-dotenv==1.0.1" \
    "tqdm==4.65.0" \
    "httpx==0.28.1" \
    "gradio==6.8.0" \
    "pydub==0.25.1"

# ---------------------------------------------------------------------------
# 4. Pre-download Whisper 'small' model into the image
#    Eliminates the 30-60s cold-start download on first request.
#    Set HF_HOME before download so the path is predictable and owned by appuser.
# ---------------------------------------------------------------------------
ENV HF_HOME=/app/.cache/huggingface

RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')" \
    && chown -R appuser:appuser /app/.cache

# ---------------------------------------------------------------------------
# 5. Application source (changes most often — last layer)
# ---------------------------------------------------------------------------
COPY --chown=appuser:appuser src/ ./src/

# ---------------------------------------------------------------------------
# 6. Runtime config
# ---------------------------------------------------------------------------
USER appuser

EXPOSE 7860

# Gradio needs a writable tmp directory for audio files.
# /tmp is writable by all users; TMPDIR overrides tempfile.gettempdir() calls.
ENV TMPDIR=/tmp

# YouTube Data API key for fetching captions from cloud/data-center IPs.
# Get one at: https://console.cloud.google.com/apis/credentials
# Enable "YouTube Data API v3" in the GCP console.
# Set via env var at runtime (do NOT bake into the image).
# ENV YOUTUBE_API_KEY=your-key-here

# Health check: Gradio serves its root at / once ready.
# start-period gives the model-loading time before probes count as failures.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -sf http://localhost:7860/ > /dev/null || exit 1

CMD ["python", "-m", "src.app"]
