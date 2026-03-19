# 📄 PaperChat

**Chat with research papers using multimodal RAG** — understands text, tables, figures, diagrams, and equations.

---

## What It Does

PaperChat lets you upload any research paper (PDF) and ask questions about it in natural language. It doesn't just search text — it understands the *whole* paper, including charts, diagrams, tables, and equations, by combining three extraction methods before building a searchable vector index.

---

## Architecture

```
PDF uploaded
 │
 ├─ 1. Text extraction       PyMuPDF        (local, free)
 ├─ 2. Table extraction      pdfplumber     (local, free)
 └─ 3. Visual description    Gemini Vision  (only visual pages)
         │
         ▼
 All chunks merged into a single list
         │
 ├─ 4. Embeddings            Gemini Embedding
 └─ 5. Vector store          Chroma (in-memory)
         │
         ▼
 User asks a question
         │
 ├─ 6. Retrieval             Chroma similarity search (top 5 chunks)
 └─ 7. Answer generation     Groq LLM (Llama 3.3 70B)
```

**Key optimization:** pages are checked locally before any API call. Plain text pages skip vision entirely, reducing Gemini Vision calls by 50–70% on typical papers.

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd paperchat
pip install -r requirements.txt
```

### 2. Get API keys

| Service | What it's used for | Free tier |
|---|---|---|
| [Google AI Studio](https://aistudio.google.com/) | Embeddings + Vision | Generous free quota |
| [Groq](https://console.groq.com/) | Chat / answer generation | ~14,400 req/day |

### 3. Configure environment

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY="your_gemini_key"
GROQ_API_KEY="your_groq_key"
```

### 4. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Usage

1. Upload a PDF research paper using the file uploader.
2. Wait for processing — a progress bar tracks each page (text → tables → visuals).
3. Once indexed, ask anything: *"What does Figure 3 show?"*, *"Summarize the results table"*, *"Explain the loss function used"*.
4. Re-uploading the same paper skips reprocessing entirely (MD5-based cache).

---
## How Each Part Works

**Text extraction (PyMuPDF)** — Pulls raw text from each page locally. Fast, free, no API.

**Table extraction (pdfplumber)** — Detects table structure natively and formats each row as `col1 | col2 | col3` text. More accurate than vision for clean digital tables.

**Visual description (Gemini Vision)** — Pages with images or diagrams are rendered as PNG and sent to `gemini-2.0-flash`. The model describes axes, trends, components, and equations in plain English. That description is then embedded like any other chunk, making visuals fully searchable.

**Embeddings (Gemini `gemini-embedding-001`)** — Converts all chunks to vectors once per paper upload. Gemini embeddings handle technical/scientific language well.

**Vector store (Chroma, in-memory)** — Stores vectors for the session. In-memory is ideal here since papers change every session; raw chunks are deleted after indexing to free RAM.

**Answer generation (Groq)** — The top 5 retrieved chunks are passed as context to Llama 3.3 70B. If a rate limit is hit, it silently falls back to `llama-3.1-8b-instant`, then `mixtral-8x7b-32768`.

---

## Project Structure

```
paperchat/
├── app.py            # Full application (extraction, embedding, chat UI)
├── requirements.txt  # Python dependencies
├── .env              # API keys (never commit this)
└── README.md
```

---

## Requirements

- Python 3.9+
- Internet connection (for Gemini and Groq API calls)
- A Google AI Studio API key and a Groq API key
