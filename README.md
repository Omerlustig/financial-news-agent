# Market News Agent: Institutional Financial News Intelligence & Alerting Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

An institutional-grade, automated financial news monitoring and analysis agent. The agent ingests breaking news from major financial publishers, SEC feeds, and custom RSS sources, evaluates catalyst significance and price-moving probability using Large Language Models via LiteLLM, filters out noise, and delivers high-conviction market-moving alerts directly to Telegram channels or private chats.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Ingestion ["Ingestion Layer"]
        R1["Reuters Business"] --> RF["RSS Fetcher"]
        R2["CNBC Top News"] --> RF
        R3["Yahoo Finance"] --> RF
        R4["MarketWatch"] --> RF
        R5["SEC Press Releases"] --> RF
    end

    subgraph Deduplication ["Deduplication & Storage"]
        RF --> Dedupe{"SQLite Filter\n(SHA-256 Hash)"}
        Dedupe -->|Already Seen| Skip["Drop / Skip"]
        Dedupe -->|Unseen Articles| Analyzer["LLM Impact Analyzer"]
        DB[("SQLite Storage\nWAL Mode")] <--> Dedupe
    end

    subgraph Intelligence ["LLM Intelligence Engine"]
        Analyzer --> LLM{"LiteLLM Router\nGemini / OpenAI / Claude"}
        LLM --> ImpactAnalysis["Structured Analysis\n- Impact Grade\n- Confidence (1-10)\n- Affected Tickers\n- Catalyst Thesis"]
    end

    subgraph Evaluation ["Threshold Filter & Alerting"]
        ImpactAnalysis --> Filter{"Filter Rule\nScore >= min_confidence\nGrade >= min_impact"}
        Filter -->|Below Threshold| RecordOnly["Record as Routine\n(Alerted = False)"]
        Filter -->|Above Threshold| Telegram["Telegram Notifier\n(HTML Formatted)"]
        Telegram --> Channel["Telegram Subscriber\nChannel / Private Chat"]
        Telegram --> RecordAlerted["Record in SQLite\n(Alerted = True)"]
        RecordOnly --> DB
        RecordAlerted --> DB
    end
```

### Core Architecture Highlights
- **Decoupled Modular Architecture**: Clean separation between ingestion, LLM analysis, deduplication persistence, and notification layers.
- **Multi-Source Event Crossing & Corroboration Engine**: Automatically clusters incoming breaking headlines across independent newsrooms (Reuters, Bloomberg, MarketWatch, Globes, Bizportal, etc.) within a rolling time window. When 2+ independent publishers corroborate a breaking catalyst:
  - Multi-source verification gate (`is_verified`) automatically passes.
  - Confidence score receives an institutional credibility boost (+1 to +2 points).
  - Status is elevated to `OPPORTUNITY_CANDIDATE`.
  - Dispatches **ONE single consolidated Telegram alert** citing all corroborating outlets (`📡 Corroborated by: [Reuters, Bloomberg, MarketWatch]`)—preventing redundant notification spam.
- **Publication Velocity Surge Engine**: Monitors publication velocity and time deltas across reporting outlets. Rapid reporting across 3+ outlets within 30 minutes triggers `⚡ [BREAKING CATALYST SURGE]`, auto-promoting confidence score to >= 8 and escalating impact grade to `HIGH` or `CRITICAL`.
- **Full-Text Article Extraction Engine**: Powered by `trafilatura` with graceful fallback to BeautifulSoup and regex parsing, enabling deep-dive financial reasoning beyond brief RSS teaser snippets.
- **Regulatory Disclosure Taxonomy (SEC 8-K & Israeli MAYA)**: Dedicated classifier mapping SEC 8-K filing items (Item 1.01 M&A/Partnerships, Item 1.02 contract terminations, Item 2.02 earnings, Item 4.02 restatements, Item 5.02 executive changes) and TASE/MAYA corporate disclosures to deterministic baseline impact severities.
- **Interactive Telegram Inline Keyboards**: Actionable one-tap inline buttons on every alert, deep-linking directly to the primary source article, live TradingView technical charts (`https://www.tradingview.com/symbols/{TICKER}/`), and regulatory filings.
- **Executive Daily Catalyst Digest**: Periodic sector-grouped briefing generator aggregating all verified catalysts, directional stances, and transmission channels over any rolling lookback window (e.g. 24h).
- **Resilient Network Ingestion & Auto-Retries**: Configurable automatic retries with exponential backoff on network timeouts, DNS drops, or HTTP 5xx errors, plus HTTP 304 conditional GET (`ETag`/`Last-Modified`) feed caching to skip unchanged feeds.
- **Zero Duplicate Processing (Multi-Tier Deduplication)**: Automatic URL canonicalization (stripping tracking parameters like `utm_*`, `ref`, and `fbclid`), deterministic SHA-256 content hashing (`url + title`), and SQLite WAL persistence of both alerted and non-alerted articles guarantee an article is never re-analyzed or re-alerted.
- **Multi-LLM Support & Auto-Failover**: Powered by `litellm`—primary model delegation (`gemini/gemini-1.5-flash`) with automatic failover to secondary fallback models (e.g. `gpt-4o-mini`) on provider rate limits or downtime.
- **Focused, Actionable Alerts with Recommendations**: Structured Telegram alerts delivering score, impact grade, directional stance, time horizon, causal transmission mechanism, beneficiaries vs. negatively exposed tickers, and a concise actionable recommendation.
- **Robust SQLite Storage**: Thread-safe connection pooling, Write-Ahead Logging (`WAL`), and automatic schema migration.
- **Reliable Telegram Delivery**: Robust HTML escaping, automatic exponential backoff retry on HTTP 429 rate limits, and 4000-character message boundary safeguarding.

---

## Quickstart: Running Locally

### 1. Prerequisites
- Python 3.11 or higher
- Git

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Omerlustig/financial-news-agent.git
cd financial-news-agent/news-agent

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the sample environment file and configure your credentials:
```bash
cp .env.example .env
```
Edit `.env` with your preferred editor:
```ini
# Telegram Configuration
TELEGRAM_BOT_TOKEN="123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ"
TELEGRAM_CHAT_ID="-1001234567890"

# LLM Provider Key (Google Gemini default, or OpenAI / Anthropic)
GEMINI_API_KEY="AIzaSyYourGeminiKeyHere"
# OPENAI_API_KEY="sk-proj-YourOpenAIKeyHere"
# ANTHROPIC_API_KEY="sk-ant-YourAnthropicKeyHere"

# Storage & Logging
DATABASE_PATH="data/agent.db"
LOG_LEVEL="INFO"
```

### 4. Verify System Connectivity
Execute the built-in diagnostic suite to confirm database, API keys, and Telegram credentials:
```bash
python -m src.main test-connections
```

### 5. Run a Single Scan Cycle
Perform an immediate, one-shot scan across all enabled sources:
```bash
python -m src.main run-once
```
> **Tip**: Use `--alert-on-all` to bypass score thresholds during testing so you can verify end-to-end alert formatting on any retrieved article:
> ```bash
> python -m src.main run-once --alert-on-all
> ```

> **Local Mode (No Telegram Setup Required)**:
> If you have not configured `TELEGRAM_BOT_TOKEN` or `TELEGRAM_CHAT_ID` yet, the agent automatically activates **Local Mode** via `ConsoleNotifier`. Market catalyst alerts, sentiment meters, buzz scores, and trend verdicts will be printed directly to your terminal inside formatted Rich panels. You can test and run immediately without registering a Telegram bot!

### 6. Run Continuous Monitoring Daemon
Launch the autonomous background monitoring daemon with a 15-minute polling loop:
```bash
python -m src.main run --daemon --interval 15
```
Press `Ctrl+C` to trigger a clean, graceful shutdown.

---

## Configuration Guide

The agent is driven by two human-readable YAML configuration files located in `config/`.

### 1. Customizing News Sources (`config/sources.yaml`)
Define the news feeds and RSS streams the agent monitors:
```yaml
sources:
  - name: "Reuters Business"
    type: "rss"
    url: "https://feeds.feedburner.com/reuters/businessNews"
    enabled: true
    max_items: 10                 # Optional: Fetch at most 10 articles from Reuters (overrides global batch_size_per_source)
    max_article_age_minutes: 15   # Optional: Only ingest articles from the past 15 min for this feed

  - name: "CNBC Top News"
    type: "rss"
    url: "https://www.cnbc.com/id/100003114/device/rss/rss.html"
    enabled: true
```
- Set `enabled: false` to temporarily disable a source without deleting its entry.
- Set `max_items: N` to limit articles fetched from a specific site.
- Set `max_article_age_minutes: M` to set a custom recency threshold for a specific site.
- Test individual feed ingestion anytime using:
  ```bash
  python -m src.main test-source --name "Reuters Business"
  ```

#### Custom Google Alerts Integration
You can turn Google Alerts into custom real-time RSS feeds for targeted catalysts (e.g. M&A, executive ousters, or your personal stock watchlist) and ingest them with zero API fees.
👉 **See the step-by-step setup guide**: [`docs/google_alerts_setup.md`](docs/google_alerts_setup.md)

### 2. Tuning Engine Settings (`config/settings.yaml`)
Control polling frequency, recency limits, model parameters, and alert sensitivity thresholds:
```yaml
agent:
  polling_interval_minutes: 15    # Default cycle interval for daemon
  batch_size_per_source: 15       # Global limit: max articles fetched per feed per cycle
  max_article_age_minutes: 15     # Global recency filter: only fetch/process news from the past 15 min (set null/0 to disable)
  database_path: "data/agent.db"  # Path to SQLite database
```

#### CLI Overrides for Quick Testing
You can also override these limits dynamically on the command line:
```bash
# Fetch only 5 articles per site, from the last 10 minutes:
python -m src.main run-once --limit-per-source 5 --max-age-minutes 10

# Or short aliases:
python -m src.main run-once -b 5 -m 15
```

filter:
  min_confidence_score: 6         # Minimum LLM confidence score (1-10) to alert
  min_impact_grade: "HIGH"        # Options: LOW, AVERAGE (or MEDIUM), HIGH, CRITICAL
  alert_on_all: false             # Set true to alert on all articles regardless of score

llm:
  model: "gemini/gemini-1.5-flash" # LiteLLM model identifier
  temperature: 0.1                # Deterministic scoring
  max_retries: 3                  # Retry attempts on transient API failures

telegram:
  bot_token: "${TELEGRAM_BOT_TOKEN}"
  chat_id: "${TELEGRAM_CHAT_ID}"
```

### 3. Setting Up Scheduled Cron Jobs

If you prefer scheduled execution rather than a long-running daemon, use `run-once`:

#### Option A: Linux Crontab
Open your crontab editor:
```bash
crontab -e
```
Add the following line to run every 15 minutes:
```cron
*/15 * * * * cd /opt/news-agent && /opt/news-agent/venv/bin/python -m src.main run-once >> /var/log/news-agent.log 2>&1
```

#### Option B: Systemd Timer
Create `/etc/systemd/system/news-agent-cron.service`:
```ini
[Unit]
Description=Market News Agent Single Scan Cycle
After=network-online.target

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/opt/news-agent
EnvironmentFile=/etc/financial-news-agent.env
ExecStart=/opt/news-agent/venv/bin/python -m src.main run-once
```

Create `/etc/systemd/system/news-agent-cron.timer`:
```ini
[Unit]
Description=Run Market News Agent every 15 minutes

[Timer]
OnCalendar=*:0/15
Persistent=true

[Install]
WantedBy=timers.target
```
Enable and start the timer:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now news-agent-cron.timer
```

#### Option C: GitHub Actions Cron
Create `.github/workflows/market-news-cron.yml`:
```yaml
name: Market News Agent Cron Scan

on:
  schedule:
    - cron: '*/15 * * * *'
  workflow_dispatch:

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r news-agent/requirements.txt
      - name: Run single scan cycle
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
        run: |
          cd news-agent
          python -m src.main run-once
```

---

## Telegram Integration Guide

> 📖 **Detailed Guide**: A standalone, illustrated guide with troubleshooting tips and screenshots is available at [`docs/telegram_setup_guide.md`](docs/telegram_setup_guide.md).

The agent delivers high-priority institutional alerts in structured HTML with impact classification tags, confidence scoring, affected tickers, and analytical catalysts.

### Step 1: Create a Bot with @BotFather
1. Open the Telegram app and search for `@BotFather`.
2. Start the chat and send `/newbot`.
3. Choose a display name (e.g., `Market News Intelligence Bot`).
4. Choose a username ending in `bot` (e.g., `MyMarketAlertsBot`).
5. @BotFather will provide an API token formatted like:
   `7123456789:AAF_k2oA8eB9xyz1234567890abcdef`
6. Save this value as `TELEGRAM_BOT_TOKEN` in your `.env`.

### Step 2: Obtain your Telegram Chat ID
You can deliver alerts to a personal direct message, a private group, or a broadcast channel:

#### Method A: Personal Direct Message
1. In Telegram, search for `@userinfobot` and click **Start**.
2. The bot will reply with your personal user ID (e.g. `987654321`).
3. Send any message to your newly created bot to open a dialog.
4. Set `TELEGRAM_CHAT_ID=987654321`.

#### Method B: Telegram Channel or Group
1. Create a channel or group in Telegram.
2. Add your bot as an **Administrator** with permission to **Post Messages**.
3. Send a message to the group/channel (e.g., "Hello Bot").
4. Open your browser and navigate to:
   ```
   https://api.telegram.org/bot<YOUR_TELEGRAM_BOT_TOKEN>/getUpdates
   ```
5. Find the `"chat":{"id": -100xxxxxxxxxx}` object in the JSON response.
   Channel/group IDs begin with `-100`.
6. Set `TELEGRAM_CHAT_ID=-100xxxxxxxxxx` in `.env`.

### Step 3: Test Integration
Run the diagnostic command to transmit a live test alert:
```bash
python -m src.main test-telegram --message "Market News Agent connected and verified."
```

Sample Alert Preview:
```html
🚨 [OPPORTUNITY CANDIDATE | Score: 9/10 | CRITICAL]
📈 Stance: BULLISH | ⏱ Horizon: SHORT_TERM (1-7 days)
🏭 Sectors: Semiconductors, AI Hardware
Headline: Nvidia Announces Record Blackwell Shipments and Major Hyperscaler Deals
Publisher: Reuters | Published: 2026-10-17 19:30 UTC

Executive Summary:
Nvidia announced record Blackwell shipments and billions in multi-year commitments with global cloud providers.

Causal Mechanism:
Surge in hyperscaler data center capex directly expands operating margins and accelerates booked backlog recognition.

🟢 Beneficiaries: $NVDA, $TSM, $AVGO
🔴 Negatively Exposed: $INTC

💡 Recommendation: Prioritize long exposure on $NVDA/$TSM on dips; hedge downside risk via $SMH puts; avoid $INTC until next quarter.

⚖️ Counterarguments: Foundry packaging bottlenecks; risk of tighter trade export licensing.
🛑 Invalidation Condition: Hyperscaler capex guidance reductions in next 10-Q filings.

🔗 https://reuters.com/business/nvidia-blackwell-record-shipments

⚠️ News-derived hypothesis. Market pricing, valuation, and entry points were not checked.
```

---

## Deployment Guide

### Option 1: Docker & Docker Compose (Recommended for EC2 / VPS)

The container runs as an unprivileged user (`appuser`) with a dedicated volume mount for SQLite persistence.

1. **Clone the repository on your server:**
   ```bash
   git clone https://github.com/Omerlustig/financial-news-agent.git /opt/financial-news-agent
   cd /opt/financial-news-agent/news-agent
   ```

2. **Configure production environment:**
   ```bash
   cp .env.example .env
   # Add your production TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, GEMINI_API_KEY
   chmod 600 .env
   ```

3. **Start the service in detached mode:**
   ```bash
   docker compose up -d --build
   ```

4. **Monitor container health & logs:**
   ```bash
   docker compose logs -f
   ```

5. **Stop or restart:**
   ```bash
   docker compose restart
   docker compose down
   ```

### Option 2: Linux VPS / AWS EC2 with Systemd

For bare-metal Ubuntu / Debian / Amazon Linux environments:

1. **Deploy codebase:**
   ```bash
   sudo mkdir -p /opt/news-agent
   sudo chown -R ubuntu:ubuntu /opt/news-agent
   git clone https://github.com/omer/financial-news-agent.git /opt/news-agent
   cd /opt/news-agent/news-agent
   ```

2. **Setup virtual environment:**
   ```bash
   python3 -m venv /opt/news-agent/venv
   /opt/news-agent/venv/bin/pip install --upgrade pip
   /opt/news-agent/venv/bin/pip install -r requirements.txt
   mkdir -p /opt/news-agent/data
   ```

3. **Create secure environment file:**
   ```bash
   sudo nano /etc/financial-news-agent.env
   ```
   Add:
   ```ini
   TELEGRAM_BOT_TOKEN="your_token"
   TELEGRAM_CHAT_ID="your_chat_id"
   GEMINI_API_KEY="your_gemini_api_key"
   DATABASE_PATH="/opt/news-agent/data/agent.db"
   PYTHONUNBUFFERED="1"
   ```
   Secure permissions:
   ```bash
   sudo chmod 600 /etc/financial-news-agent.env
   sudo chown ubuntu:ubuntu /etc/financial-news-agent.env
   ```

4. **Install Systemd service:**
   ```bash
   sudo cp news-agent/financial-news-agent.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now financial-news-agent
   ```

5. **Manage & monitor service:**
   ```bash
   sudo systemctl status financial-news-agent
   journalctl -u financial-news-agent -f
   ```

### Option 3: Serverless Deployment on Vercel

Vercel Cron allows running serverless scheduled executions without persistent servers:

1. **Create an API handler (`api/cron.py`):**
   ```python
   from http.server import BaseHTTPRequestHandler
   from src.pipeline import Pipeline
   from src.config import load_settings, load_sources

   class handler(BaseHTTPRequestHandler):
       def do_GET(self):
           settings = load_settings()
           sources = load_sources()
           with Pipeline(settings, sources) as pipeline:
               summary = pipeline.run_cycle()
           self.send_response(200)
           self.send_header('Content-type', 'text/plain')
           self.end_headers()
           self.wfile.write(f"Scanned {summary.fetched} articles. Alerted {summary.alerted}.".encode())
   ```

2. **Configure `vercel.json` for cron:**
   ```json
   {
     "crons": [
       {
         "path": "/api/cron",
         "schedule": "*/15 * * * *"
       }
     ]
   }
   ```
   > **Note on Storage**: Since Vercel Serverless is ephemeral, configure `DATABASE_PATH` to point to a mounted cloud storage directory or integrate an external remote SQLite provider (e.g. Turso / LibSQL).

### Option 4: Google Cloud Run & Firebase Functions

Deploy as a serverless container job triggered by Cloud Scheduler:

1. **Build and push image to Google Artifact Registry:**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/financial-news-agent:latest .
   ```

2. **Create Cloud Run Job:**
   ```bash
   gcloud run jobs create financial-news-job \
       --image gcr.io/PROJECT_ID/financial-news-agent:latest \
       --command python \
       --args "-m","src.main","run-once" \
       --set-env-vars "TELEGRAM_BOT_TOKEN=...,TELEGRAM_CHAT_ID=...,GEMINI_API_KEY=..."
   ```

3. **Schedule with Cloud Scheduler:**
   ```bash
   gcloud scheduler jobs create http news-agent-trigger \
       --schedule "*/15 * * * *" \
       --uri "https://run.googleapis.com/v1/namespaces/PROJECT_ID/jobs/financial-news-job:run" \
       --http-method POST \
       --oauth-service-account-email "news-agent-invoker@PROJECT_ID.iam.gserviceaccount.com"
   ```

---

## Diagnostic CLI Reference

The agent includes full diagnostic utilities for real-time inspection and verification:

| Command | Description |
|---|---|
| `python -m src.main run-once` | Executes a single scan, analysis, alerting, and persistence cycle. |
| `python -m src.main run-once --alert-on-all` | Bypasses score/impact thresholds; sends alerts for all unseen items. |
| `python -m src.main run --daemon --interval 15` | Runs the continuous polling daemon with specified interval (minutes). |
| `python -m src.main test-connections` | Validates SQLite DB schema, LLM API keys, and Telegram credentials. |
| `python -m src.main test-telegram` | Dispatches an instant test message to the configured Telegram chat. |
| `python -m src.main test-source --name "<name>"` | Fetches and displays the top 5 parsed headlines from a single feed. |
| `python -m src.main stats` | Displays total seen articles, market-moving detections, and alert counts. |
| `python -m src.main metrics` | Displays total LLM inference counts, tokens, estimated spend, and model breakdown. |
| `python -m src.main digest --hours 24` | Generates a 24-hour sector-grouped catalyst executive digest to the console. |
| `python -m src.main digest --hours 24 --send-telegram` | Generates and broadcasts the 24-hour catalyst briefing directly to Telegram. |
| `python -m src.main scanner --hours 24` | Real-time terminal market scanner showing ticker buzz multipliers, sentiment gauges, velocities, and trend verdicts. |

---

## Running Test Suite

The test suite covers configuration loading, RSS ingestion, LLM parsing/fallback handling, HTML formatting, rate limit retries, SQLite deduplication, and CLI commands:

```bash
# Run all tests with pytest
pytest news-agent/tests/ -v

# Run with coverage report
pytest news-agent/tests/ --cov=src --cov-report=term-missing
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
