# Automated Firebase Deployment Guide with GitHub Actions

This guide explains how to automatically deploy the Market News Agent to **Firebase Cloud Functions (2nd Gen)** on every `git push` to your GitHub repository.

---

## Architecture Overview

```
GitHub Push (main)
       │
       ▼
GitHub Actions CI/CD
  ├─ 1. Run 310+ Pytest tests
  ├─ 2. Authenticate via Google Service Account
  └─ 3. Run `firebase deploy --only functions`
       │
       ▼
Firebase / Google Cloud Functions (2nd Gen)
  ├─ Trigger: Cloud Scheduler (Every 15 minutes)
  ├─ Storage Sync: Pulls / Pushes `agent.db` to Firebase Storage
  └─ Telegram Alerts: Dispatches institutional alerts to @stocks_hints_bot
```

---

## Step 1: What You Need to Provide

To connect GitHub Actions to your Firebase project, you need to add **5 Secrets** in your GitHub repository:

| Secret Name | What It Is | Example / Source |
| :--- | :--- | :--- |
| `FIREBASE_PROJECT_ID` | Your Firebase Project ID | e.g. `my-stocks-agent-12345` |
| `FIREBASE_SERVICE_ACCOUNT` | JSON key for deployment permissions | Generated in Firebase / GCP Console (see Step 2 below) |
| `TELEGRAM_BOT_TOKEN` | Token for `@stocks_hints_bot` | From `@BotFather` |
| `TELEGRAM_CHAT_ID` | Your Telegram chat ID | `1682918883` |
| `GEMINI_API_KEY` | Google Gemini API Key | From Google AI Studio |

---

## Step 2: How to Generate `FIREBASE_SERVICE_ACCOUNT`

1. Open the [Firebase Console](https://console.firebase.google.com/) and select your project.
2. Click the ⚙️ **Gear icon** (top left next to "Project Overview") ➔ **Project settings**.
3. Go to the **Service accounts** tab.
4. Click the blue button: **"Generate new private key"** ➔ click **"Generate key"**.
5. A `.json` file will download to your computer.
6. Open this `.json` file in any text editor, copy its entire contents, and paste it as the `FIREBASE_SERVICE_ACCOUNT` secret in GitHub.

> **Note on Permissions**: The default Firebase Admin service account has full permissions to deploy functions and manage storage. If using a custom service account, ensure it has:
> - `Cloud Functions Admin`
> - `Cloud Run Admin`
> - `Service Account User`
> - `Storage Admin`
> - `Cloud Scheduler Admin`

---

## Step 3: Enable Firebase Storage (for Persistent Database)

Because serverless functions have ephemeral file systems, the agent automatically backs up `agent.db` to Firebase Cloud Storage at the end of each run and restores it at the beginning of the next run.

1. In the [Firebase Console](https://console.firebase.google.com/), go to **Build** ➔ **Storage**.
2. If Storage is not yet enabled, click **Get started**, choose production or test mode, and click **Done**.
3. That's it! The agent will automatically use your default storage bucket.

---

## Step 4: Add Secrets to Your GitHub Repository

1. On GitHub, navigate to your repository.
2. Click **Settings** ➔ **Secrets and variables** ➔ **Actions**.
3. Click **New repository secret** for each of the 5 secrets listed in Step 1.

---

## Step 5: Deploy!

Simply push a commit to the `main` branch:

```bash
git add .
git commit -m "ci: enable automated Firebase deployment"
git push origin main
```

Or trigger manually:
1. Go to the **Actions** tab on your GitHub repository.
2. Click **Deploy to Firebase Functions** in the left sidebar.
3. Click **Run workflow** ➔ **Run workflow**.

---

## Useful Local Commands (Optional)

If you have `firebase-tools` installed locally (`npm install -g firebase-tools`):

```bash
# Login to Firebase
firebase login

# Set active project
firebase use <YOUR-PROJECT-ID>

# Set runtime secrets for your function
firebase functions:secrets:set TELEGRAM_BOT_TOKEN
firebase functions:secrets:set TELEGRAM_CHAT_ID
firebase functions:secrets:set GEMINI_API_KEY

# Manual local deployment
firebase deploy --only functions
```
