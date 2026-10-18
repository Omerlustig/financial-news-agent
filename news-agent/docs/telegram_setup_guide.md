# 📱 Complete Telegram Bot & Alert Setup Guide

This guide walks you through generating your **`TELEGRAM_BOT_TOKEN`** and **`TELEGRAM_CHAT_ID`** in under 2 minutes, configuring them in your environment, and testing live market alerts.

## ⚡ Quick Reference

| Credential | What it is | How to get it |
| :--- | :--- | :--- |
| **`TELEGRAM_BOT_TOKEN`** | The API key for your bot | Chat with **`@BotFather`** on Telegram $\rightarrow$ send `/newbot` |
| **Bot Verification** | Confirms token is active & gives bot username | Open in browser: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getMe` |
| **`TELEGRAM_CHAT_ID` (Personal)** | **Your personal account ID** | Chat with **`@userinfobot`** on Telegram $\rightarrow$ click **Start** |
| **`TELEGRAM_CHAT_ID` (Channel)** | Broadcast channel ID (`-100...`) | Add bot as Admin, post message, check `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates` |

---

## 📋 Overview

The Market News Agent uses the official Telegram Bot API to dispatch instant, real-time institutional alerts containing:
- **Breaking News Badges**: `⚡ [BREAKING CATALYST SURGE]`, `⭐ [WATCHLIST FOCUS]`
- **Directional Stance & Scores**: Bullish/Bearish/Neutral meters and confidence ratings (1-10)
- **Causal Mechanism**: Economic transmission channel explaining why the news moves the market
- **Beneficiaries vs. Negatively Exposed**: Specific public tickers and sectors
- **Interactive Inline Buttons**: One-click TradingView chart links and source article links

---

## 🛠️ Step 1: Generate Your `TELEGRAM_BOT_TOKEN` (via `@BotFather`)

BotFather is Telegram's official bot for creating and managing bots.

1. Open Telegram on your phone or desktop.
2. In the search bar, search for **`@BotFather`** (ensure it has the official blue checkmark).
3. Click **Start** or send `/start`.
4. Send the command:
   ```text
   /newbot
   ```
5. **Choose a Display Name**:
   BotFather will ask: *"Alright, a new bot. How are we going to call it? Please choose a name for your bot."*
   - Type any name you like, for example: `Market News Agent` or `Omer Market Intelligence`.
6. **Choose a Username**:
   BotFather will ask: *"Good. Now let's choose a username for your bot. It must end in `bot`."*
   - Type a unique username ending in `bot`, for example:
     `omer_market_news_bot` or `my_stock_catalyst_bot`.
7. **Copy Your Bot Token**:
   BotFather will congratulate you and provide an API token formatted like this:
   ```text
   Use this token to access the HTTP API:
   7829104820:AAHq_z8ExampleToken12345abcdef
   ```
   👉 **Save this token securely** — this is your `TELEGRAM_BOT_TOKEN`.

---

### Step 1.1: Verify Your Bot Token via `getMe`

Right after getting your token, open your web browser and navigate to:
```text
https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getMe
```
*(Replace `<YOUR_BOT_TOKEN>` with the token you just received).*

You will see a JSON response confirming your bot is active:
```json
{
  "ok": true,
  "result": {
    "id": 8801638490,
    "is_bot": true,
    "first_name": "stock-hints",
    "username": "stocks_hints_bot",
    "can_join_groups": true
  }
}
```

> [!NOTE]
> - **What `getMe` tells you**: It confirms your API token is 100% valid and gives you your bot's exact username (e.g. `@stocks_hints_bot`).
> - ⚠️ **Important distinction**: Notice `result.id` (`8801638490`). That is the **Bot's own ID**. Do **NOT** set `TELEGRAM_CHAT_ID` to this ID! A bot cannot send messages to itself (`Forbidden: the bot can't send messages to the bot`).
> - 👤 **To get YOUR Personal Chat ID**: Search for **`@userinfobot`** in Telegram and click **Start** — it immediately replies with your numeric personal ID!
> - 📢 **To get a Channel ID**: Add your bot as an admin to your channel, post a message, and check `getUpdates` (explained in Step 3 Method C below).

---

## 🔑 Step 2: Start Your Bot (or Add to Channel) (**Crucial Step!**)

> [!IMPORTANT]
> **Telegram bots cannot message a user or channel first.** You must either start a chat or add the bot as an administrator.

1. **For Personal Alerts**: In Telegram, search for your bot's username (from `getMe`, e.g. `@stocks_hints_bot`) and click **Start** (or send `/start` or `hello`).
2. **For Channel Alerts**: Create a Telegram Channel, go to **Channel Settings** $\rightarrow$ **Administrators** $\rightarrow$ **Add Administrator**, search for your bot's username, and add it with permission to **Post Messages**. Send any message (e.g. `test`) to the channel.

---

## 🆔 Step 3: Find Your `TELEGRAM_CHAT_ID`

Pick either of the two easy methods below:

### Method A: Fastest (via `@userinfobot` — 10 seconds)
1. In Telegram, search for **`@userinfobot`** (or `@RawDataBot`).
2. Click **Start**.
3. It will reply immediately with your account profile:
   ```text
   Id: 123456789
   First: Omer
   Username: omer_user
   ```
   👉 The number next to **Id** (e.g. `123456789`) is your personal `TELEGRAM_CHAT_ID`.

---

### Method B: Directly from your Web Browser
1. Make sure you sent at least one message (e.g. `hello`) to your new bot in Step 2.
2. Open your web browser and navigate to:
   ```text
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
   *(Replace `<YOUR_BOT_TOKEN>` with the token from Step 1).*
3. You will see a JSON response. Look for the `"chat"` block:
   ```json
   {
     "ok": true,
     "result": [
       {
         "update_id": 987654321,
         "message": {
           "from": { "id": 123456789, "first_name": "Omer" },
           "chat": { "id": 123456789, "type": "private" },
           "text": "hello"
         }
       }
     ]
   }
   ```
   👉 The `"id": 123456789` under `"chat"` is your `TELEGRAM_CHAT_ID`.

---

### Method C: Sending Alerts to a Telegram Channel or Group

If you want alerts broadcast to a private channel or team group instead of a direct message:

1. **Create a Channel or Group** in Telegram.
2. Go to **Channel / Group Settings** $\rightarrow$ **Administrators** $\rightarrow$ **Add Administrator**.
3. Search for your bot's username and add it as an **Admin** with permission to **Post Messages**.
4. Post any test message (e.g. `test`) into the channel.
5. In your browser, open `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`.
6. Look for the channel `"chat"` ID in the JSON response:
   - Channel IDs always start with `-100` (e.g., `-1001928374650`).
   - Group IDs start with `-` (e.g., `-492817263`).
7. Use that full number (including the minus sign) as your `TELEGRAM_CHAT_ID`.

---

## ⚙️ Step 4: Add Tokens to `news-agent/.env`

Open your `news-agent/.env` file (or copy from `.env.example` if not created yet):

```bash
cd news-agent
cp .env.example .env
```

Edit `.env` and fill in your values:

```env
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN="7829104820:AAHq_z8ExampleToken12345abcdef"
TELEGRAM_CHAT_ID="123456789"

# LLM API Keys (Gemini is free and default)
GEMINI_API_KEY="your-gemini-api-key-here"
```

---

## 🧪 Step 5: Test Telegram Alerts

Run our built-in diagnostic test command:

```bash
cd news-agent
python -m src.main test-telegram
```

### What You Should See in Your Terminal:
```text
Sending test notification via Telegram...
✅ Telegram notification sent successfully!
```

### What You Will Receive in Telegram:
An institutional test alert card will arrive instantly:
```text
🚨 [OPPORTUNITY CANDIDATE | Score: 9/10 | CRITICAL]
⚡ [BREAKING CATALYST SURGE]
⭐ [WATCHLIST FOCUS: $NVDA, Semiconductors]
📈 Stance: BULLISH | ⏱ Horizon: SHORT_TERM (1-7 days)
🏭 Sectors: Semiconductors, AI Infrastructure, Cloud

Headline: Nvidia Unveils Next-Generation AI Architecture
Publisher: Reuters Business | 2026-10-18 10:00 UTC

📝 Brief:
Nvidia announced mass production of its next-generation ultra-dense AI clusters...

⚙️ Causal Mechanism:
Accelerating enterprise LLM training demand drives immediate 40% margin uplift...

🟢 Beneficiaries: $NVDA, $TSM, $AVGO
🔴 Negatively Exposed: $INTC

⚖️ Counterarguments: Supply chain packaging constraints at CoWoS foundries
🛑 Invalidation Condition: Customer hyperscaler CAPEX guidance cuts below 15%

⚠️ News-derived hypothesis. Market pricing, valuation, and entry points were not checked.

[🔗 Read Full Article]  [📊 Chart NVDA]  [📊 Chart TSM]
```

---

## ❓ Troubleshooting & FAQs

### 1. `Error: HTTP 401 Unauthorized`
- **Cause**: The `TELEGRAM_BOT_TOKEN` is typed incorrectly or has extra spaces.
- **Fix**: Re-copy the exact token string from `@BotFather` without quotes or leading/trailing whitespace.

### 2. `Error: HTTP 400 Bad Request: chat not found`
- **Cause**: You forgot to press **Start** on your bot in Telegram, or the `TELEGRAM_CHAT_ID` is wrong.
- **Fix**: Open your bot chat in Telegram and send `/start`, then verify your user ID using `@userinfobot`.

### 3. `Error: HTTP 403 Forbidden: bot was blocked by the user`
- **Cause**: The bot was muted or blocked in your Telegram settings.
- **Fix**: Unblock the bot and send it `/start`.

### 4. `Local Mode (Console Output)`
- If `TELEGRAM_BOT_TOKEN` or `TELEGRAM_CHAT_ID` are omitted from `.env`, the agent **does not crash**. It automatically falls back to **Local Mode (`ConsoleNotifier`)**, printing all formatted market catalyst panels directly to your terminal screen via Rich.
