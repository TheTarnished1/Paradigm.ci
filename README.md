

CI Engine: Custom Slack Teammate

A customizable AI agent for Slack that uses RAG (Retrieval-Augmented Generation) to answer questions based on your specific business documents.

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![AI Model](https://img.shields.io/badge/AI-Llama%203.3-purple)

Overview

This project connects a **Llama 3** brain (via Groq) to **Slack**. It features a "Hybrid Memory" system:
1.  **Conversational Mode:** Answers general questions intelligently.
2.  **RAG Mode:** If you add PDF documents to the `docs/` folder, the bot automatically reads them, cites its sources, and provides page numbers for every claim.

Prerequisites

* Python 3.10 or higher
* A Slack Workspace (with Admin rights to create Apps)
* A Groq API Key (Free tier available)

⚙️ Installation

1.  **Clone or Download this repository.**
2.  **Create a Virtual Environment** (Recommended to avoid conflicts):
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install slack_bolt langchain langchain-groq langchain-community langchain-core faiss-cpu sentence-transformers python-dotenv
    ```

 Configuration

### 1. The Secrets (`.env`)
Create a file named `.env` in the root folder and add your keys:

```ini
GROQ_API_KEY=gsk_...
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
2. The DNA (client_config.json)
Edit this file to change the bot's personality without touching code.

JSON

{
    "dna_identity": {
        "ci_name": "LogiBot",
        "business_name": "LogiCorp",
        "base_personality": "Professional, Concise, Safety-Focused"
    },
    "dna_synapse": {
        "model": "llama-3.3-70b-versatile"
    }
 How to Run
Step 1: Build the Memory (Optional)
If you have PDFs in the docs/ folder, run this once to teach the AI:

python setup_tools.py
Step 2: Start the Bot
Run the main engine:

Bash

python slack_bot.py
You should see: CI Teammate is online and listening to Slack...

Step 3: Chat in Slack
Mention: @LogiBot Help me with...

DM: Send a direct message to the bot.

 Project Structure
slack_bot.py - The main engine (Connects Slack + AI).

setup_tools.py - The builder (Reads PDFs and creates the memory).

client_config.json - The personality settings.

docs/ - Place your PDF training files here.

faiss_index/ - The database created by the setup tool (do not edit manually).

