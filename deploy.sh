#!/usr/bin/env bash
# =============================================================================
# Podsum v2.1 - Deploy to Google Cloud Run
# =============================================================================
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - A GCP project selected (gcloud config set project YOUR_PROJECT)
#   - GEMINI_API_KEY set in your environment or .env file
# =============================================================================
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
REGION="us-central1"
SERVICE_NAME="podsum"
IMAGE_NAME="podsum"
REPO_NAME="podsum-repo"

# Read API keys from .env if not already in environment
if [ -f .env ]; then
  export $(grep -v '^#' .env | grep -v '^\s*$' | xargs)
fi

# ── Validation ───────────────────────────────────────────────────────
if [ -z "$PROJECT_ID" ]; then
  echo "Error: No GCP project set. Run: gcloud config set project YOUR_PROJECT"
  exit 1
fi

if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "Error: GEMINI_API_KEY not set. Export it or add to .env"
  exit 1
fi

echo "============================================="
echo "  Deploying Podsum v2.1 to Cloud Run"
echo "============================================="
echo "  Project:  $PROJECT_ID"
echo "  Region:   $REGION"
echo "  Service:  $SERVICE_NAME"
echo "============================================="
echo ""

# ── Step 1: Enable required APIs ─────────────────────────────────────
echo "[1/5] Enabling required GCP APIs..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --quiet

# ── Step 2: Create Artifact Registry repo (if not exists) ───────────
echo "[2/5] Setting up Artifact Registry..."
gcloud artifacts repositories describe "$REPO_NAME" \
  --location="$REGION" \
  --format="value(name)" 2>/dev/null || \
gcloud artifacts repositories create "$REPO_NAME" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Podsum container images" \
  --quiet

# Configure Docker to use Artifact Registry
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ── Step 3: Build with Cloud Build (no local Docker needed) ─────────
echo "[3/5] Building container image with Cloud Build..."
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"

gcloud builds submit \
  --tag "$IMAGE_URI" \
  --timeout=1200 \
  --quiet

echo "  Image: $IMAGE_URI"

# ── Step 4: Deploy to Cloud Run ─────────────────────────────────────
echo "[4/5] Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE_URI" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --port 7860 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600 \
  --concurrency 10 \
  --min-instances 0 \
  --max-instances 3 \
  --set-env-vars "GEMINI_API_KEY=${GEMINI_API_KEY}" \
  --set-env-vars "YOUTUBE_API_KEY=${YOUTUBE_API_KEY:-}" \
  --set-env-vars "WHISPER_MODEL=small" \
  --set-env-vars "TMPDIR=/tmp/podsum" \
  --quiet

# ── Step 5: Verify deployment ───────────────────────────────────────
echo "[5/5] Verifying deployment..."
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
  --region "$REGION" \
  --format="value(status.url)")

echo ""
echo "============================================="
echo "  Deployment complete!"
echo "============================================="
echo "  URL: $SERVICE_URL"
echo "  Service: $SERVICE_NAME"
echo "  Region: $REGION"
echo "============================================="
echo ""

# Health check
echo "Running health check..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL" --max-time 30 || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
  echo "  Health check passed (HTTP $HTTP_CODE)"
else
  echo "  Health check returned HTTP $HTTP_CODE"
  echo "  The service may still be starting up (Whisper model loading takes ~60-90s)."
  echo "  Check logs: gcloud run services logs read $SERVICE_NAME --region $REGION"
fi
