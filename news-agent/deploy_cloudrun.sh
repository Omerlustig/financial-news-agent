#!/usr/bin/env bash
# ==============================================================================
# Market News Agent - Automated Google Cloud Run & Cloud Scheduler Deployment
#
# Builds the production container image via Google Cloud Build, deploys to
# Google Cloud Run (Jobs/Service), and configures a recurring Cloud Scheduler
# trigger (every 15 minutes) for serverless news monitoring.
# ==============================================================================

set -euo pipefail

# Configuration with configurable defaults
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || echo '')}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-market-news-agent}"
JOB_NAME="${JOB_NAME:-market-news-agent-cron}"
SCHEDULE="${SCHEDULE:-*/15 * * * *}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "=========================================================="
echo " Market News Agent - Google Cloud Run Deployment"
echo "=========================================================="
echo " Project ID:    ${PROJECT_ID}"
echo " Region:        ${REGION}"
echo " Service Name:  ${SERVICE_NAME}"
echo " Job Name:      ${JOB_NAME}"
echo " Schedule:      ${SCHEDULE}"
echo " Container:     ${IMAGE_NAME}"
echo "=========================================================="

if [ -z "${PROJECT_ID}" ]; then
  echo "[-] ERROR: Google Cloud PROJECT_ID could not be determined."
  echo "    Please set the environment variable: export PROJECT_ID=\"your-project-id\""
  echo "    or authenticate using: gcloud config set project <PROJECT_ID>"
  exit 1
fi

command -v gcloud >/dev/null 2>&1 || {
  echo "[-] ERROR: 'gcloud' CLI is not installed or not in PATH."
  echo "    Install Google Cloud SDK from https://cloud.google.com/sdk"
  exit 1
}

echo "[1/4] Enabling required Google Cloud APIs (Cloud Run, Cloud Build, Scheduler, Artifact Registry)..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  cloudscheduler.googleapis.com \
  containerregistry.googleapis.com \
  --project="${PROJECT_ID}"

echo "[2/4] Building container image with Google Cloud Build..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gcloud builds submit "${SCRIPT_DIR}" \
  --tag="${IMAGE_NAME}" \
  --project="${PROJECT_ID}"

echo "[3/4] Deploying Cloud Run Job '${JOB_NAME}'..."
# Deploy as a Cloud Run Job (optimized for scheduled 15-min batch tasks)
if gcloud run jobs describe "${JOB_NAME}" --region="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "      Updating existing Cloud Run Job '${JOB_NAME}'..."
  gcloud run jobs update "${JOB_NAME}" \
    --image="${IMAGE_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --args="python,-m,src.main,run-once" \
    --memory=512Mi \
    --cpu=1 \
    --max-retries=1
else
  echo "      Creating new Cloud Run Job '${JOB_NAME}'..."
  gcloud run jobs create "${JOB_NAME}" \
    --image="${IMAGE_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --args="python,-m,src.main,run-once" \
    --memory=512Mi \
    --cpu=1 \
    --max-retries=1
fi

echo "[4/4] Configuring Cloud Scheduler trigger '${JOB_NAME}-schedule' (${SCHEDULE})..."
if gcloud scheduler jobs describe "${JOB_NAME}-schedule" --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "      Updating existing Cloud Scheduler trigger..."
  gcloud scheduler jobs update http "${JOB_NAME}-schedule" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --schedule="${SCHEDULE}" \
    --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
    --http-method=POST \
    --oauth-service-account-email="${PROJECT_ID}-compute@developer.gserviceaccount.com"
else
  echo "      Creating new Cloud Scheduler trigger..."
  gcloud scheduler jobs create http "${JOB_NAME}-schedule" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --schedule="${SCHEDULE}" \
    --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
    --http-method=POST \
    --oauth-service-account-email="${PROJECT_ID}-compute@developer.gserviceaccount.com"
fi

echo "=========================================================="
echo "[+] SUCCESS: Market News Agent successfully deployed!"
echo "    Cloud Run Job:      ${JOB_NAME}"
echo "    Scheduler Trigger:  ${JOB_NAME}-schedule (${SCHEDULE})"
echo "=========================================================="
