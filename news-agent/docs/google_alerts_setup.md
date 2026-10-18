# 🔔 How to Configure Google Alerts as Real-Time RSS Feeds for the Market News Agent

This guide walks you through turning [Google Alerts](https://www.google.com/alerts) into real-time, zero-cost RSS feeds that feed directly into the **Financial News Intelligence Agent**.

---

## 🎯 Why Google Alerts + News Agent?

1. **Free Real-Time Ingestion**: Google indexes millions of newsrooms, specialist industry blogs, corporate press releases, and Israeli media in real time.
2. **Zero API Quotas**: Unlike paid commercial APIs (Bloomberg Terminal, Benzinga Pro), Google Alerts RSS feeds are completely free and unmetered.
3. **Noise Filtering & Intelligence**: When Google Alerts finds breaking news, our agent automatically:
   - Removes spam, syndicated duplicates, and boilerplate solicitations.
   - Corroborates the story across other newsrooms via Jaccard clustering.
   - Extracts the **Causal Chain** (Event $\rightarrow$ Economic Mechanism $\rightarrow$ Beneficiary / Negatively Exposed tickers).
   - Enforces **Public vs. Private entity guardrails** (e.g., remapping acquired tickers to active parents and blocking private entities).
   - Delivers actionable Telegram alerts with chart links.

---

## 🛠️ Step-by-Step Setup Guide

### Step 1: Open Google Alerts
Navigate to: **[https://www.google.com/alerts](https://www.google.com/alerts)**
*(Make sure you are logged into your Google account).*

---

### Step 2: Enter Your Search Query
Type a targeted financial search query into the search box at the top. Use Boolean search operators (`OR`, `AND`, `""`, `-`) for high precision:

> **Example Query:**
> `("acquisition" OR "merger" OR "takeover" OR "bought by") AND (cybersecurity OR defense OR semiconductor)`

---

### Step 3: Configure Alert Options (**Crucial Step!**)
Click **"Show options"** right below the query box and configure the dropdowns as follows:

| Setting | Recommended Value | Why |
| :--- | :--- | :--- |
| **How often** | **As-it-happens** | Ensures breaking news is posted to the RSS feed immediately without a 24h delay. |
| **Sources** | **News** (or *Automatic*) | Restricts results to journalistic newsrooms and press wires rather than forum posts. |
| **Language** | **English** (or **Hebrew** for Israeli alerts) | Matches your target market newsrooms. |
| **Region** | **United States** / **Israel** / **Any Region** | Focuses the geography of reporting. |
| **How many** | **Only the best results** | Filters out low-quality spam scrapers. |
| **Deliver to** | ⚠️ **RSS feed** *(DO NOT select email)* | **Select "RSS feed"** from the dropdown menu. This creates an XML feed URL! |

---

### Step 4: Create Alert & Copy the RSS Feed URL
1. Click **"Create Alert"**.
2. Under the **"My alerts"** section on the page, find your newly created alert.
3. You will see an orange **RSS icon** (📡) next to the pencil (edit) icon.
4. **Right-click the RSS icon** and select **"Copy Link Address"**.
   - The URL will look like:
     ```text
     https://www.google.com/alerts/feeds/14829104820194820194/7382910482910482910
     ```

---

### Step 5: Add the Feed to `config/sources.yaml`
Open `news-agent/config/sources.yaml` and add your new alert entry:

```yaml
  # Google Alerts - Custom M&A & Tech Catalysts
  - name: "Google Alert - M&A Tech Catalysts"
    type: "rss"
    url: "https://www.google.com/alerts/feeds/YOUR_USER_ID/YOUR_ALERT_ID"
    enabled: true
    max_items: 20
    max_article_age_minutes: 60
```

> [!TIP]
> You can create multiple Google Alerts (e.g., one for Israeli Tech, one for Defense M&A, one for your personal Stock Watchlist) and add each as a separate source in `sources.yaml`.

---

### Step 6: Test the New Source Immediately
Verify that the agent connects and parses your Google Alert feed without errors:

```bash
cd news-agent
python -m src.main test-source --name "Google Alert - M&A Tech Catalysts"
```

You should see output similar to:
```text
Connecting to 'Google Alert - M&A Tech Catalysts'...
Type: rss | URL: https://www.google.com/alerts/feeds/...
Status: 200 OK | Fetched: 8 articles
[0] Sample: Proofpoint to acquire Varonis in $6.3B transaction
    Published: 2026-10-18 07:15:00+00:00
```

---

## 💡 Recommended Pre-Built Financial Queries

Copy and paste any of these high-signal queries directly into Google Alerts:

### 1. High-Impact M&A & Private Equity Buyouts
```text
("acquisition" OR "merger" OR "takeover" OR "bought by" OR "tender offer" OR "privatization") AND ("billions" OR "deal" OR "cash") -careers -job
```

### 2. Israeli High-Tech & TASE Catalysts (Hebrew)
```text
("רכישה" OR "מיזוג" OR "הצעת רכש" OR "אזהרת רווח" OR "הסכם מהותי" OR "הנפקה") (הייטק OR סייבר OR בורסה OR תל אביב)
```

### 3. Executive Turmoil & Accounting Shocks (SEC 8-K Items)
```text
("resigned abruptly" OR "ousted" OR "accounting error" OR "financial restatement" OR "Item 4.02" OR "SEC probe" OR "subpoena") -job -hiring
```

### 4. Defense, Aerospace & Geopolitical Supply Chains
```text
("defense contract" OR "DoD contract award" OR "missile defense" OR "munitions" OR "arms sale") AND ("Lockheed" OR "Raytheon" OR "Elbit" OR "General Dynamics")
```

### 5. Commercial Airlines & Fuel Catalysts
```text
("flight capacity cuts" OR "fuel surcharge" OR "jet fuel spike" OR "grounded flights" OR "pilot strike") AND ("Delta" OR "United" OR "American Airlines" OR "Boeing")
```

### 6. Specific Stock Watchlist Alerts
```text
("$NVDA" OR "$TSLA" OR "$AAPL" OR "$MSFT" OR "$AMZN") AND ("antitrust" OR "guidance cut" OR "earnings surprise" OR "patent infringement" OR "downgrade")
```

---

## 🛡️ Best Practices & Guardrails

1. **Exclude Noisy Search Words**:
   Always add negative keywords to your query to eliminate hiring noise:
   `-job -jobs -careers -internship -salary -coupon -discount`
2. **Double Quotes for Exact Phrases**:
   Use quotes around terms like `"profit warning"` or `"tender offer"` so Google requires the exact phrase.
3. **Automatic Deduplication in the Agent**:
   Even if Google Alert returns multiple reports for the same event, our agent's **Jaccard similarity clusterer** bundles them into a single consolidated catalyst with elevated confidence scores (`SurgeLevel.SURGE_BREAKING`).
4. **Private Entity Safety**:
   When news alerts trigger for private buyers (e.g. Proofpoint, Thoma Bravo, Citadel), our agent automatically switches to the **M&A Public Peer playbook**, recommending tradeable public peers ($PANW, $CRWD, $CHKP) or sector ETFs ($CIBR) instead of private stocks.
