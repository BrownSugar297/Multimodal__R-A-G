import warnings
warnings.filterwarnings("ignore")


import os
import io
import base64
import hashlib
import time
import streamlit as st
import fitz                      
import pdfplumber                
from dotenv import load_dotenv
from groq import Groq            
from google import genai         
from google.genai import types
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import CharacterTextSplitter
from typing import List

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
groq_client   = Groq(api_key=os.getenv("GROQ_API_KEY"))


class GeminiEmbeddings(Embeddings):
    def __init__(self):
        self.model = "models/gemini-embedding-001"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Called once during indexing — embeds all chunks together
        result = gemini_client.models.embed_content(
            model=self.model,
            contents=texts
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> List[float]:
        # Called on every user question to find matching chunks
        result = gemini_client.models.embed_content(
            model=self.model,
            contents=[text]
        )
        return result.embeddings[0].values


def page_has_visuals(page: fitz.Page) -> bool:
    """
    Local check (no API) — does this page have images or drawn elements?
    get_images()   → embedded images (figures, photos)
    get_drawings() → vector drawings (charts, diagrams, boxes)
    Threshold >5 on drawings ignores simple decorative borders/lines.
    """
    has_images   = len(page.get_images()) > 0
    has_drawings = len(page.get_drawings()) > 5
    return has_images or has_drawings


def page_to_image_bytes(page: fitz.Page, dpi: int = 120) -> bytes:
    """
    Render a PDF page as PNG in memory.
    120 DPI = clear enough for Gemini Vision, small enough to save tokens.
    Higher DPI = better quality but more API tokens consumed.
    """
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return pix.tobytes("png")


def describe_page_visuals(image_bytes: bytes, page_num: int) -> str:
    """
    Send a page image to Gemini Vision.
    Returns a detailed text description of all visual elements.
    This text is then embedded as a normal chunk — making visuals searchable.
    """
    b64 = base64.standard_b64encode(image_bytes).decode()

    vision_prompt = (
        "You are analyzing one page from a research paper. "
        "Describe ONLY the visual elements you see:\n"
        "- Charts/plots: axes labels, trends, key data values\n"
        "- Tables: column headers and important cell values\n"
        "- Equations: explain what they represent in plain English\n"
        "- Architecture/flow diagrams: components and how they connect\n"
        "Be specific — your description will be used for Q&A retrieval.\n"
        "If NO visuals exist on this page, reply exactly: NO_VISUALS"
    )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",   
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/png",
                                data=b64,
                            )
                        ),
                        types.Part(text=vision_prompt),
                    ],
                )
            ],
        )
        text = response.text.strip()
        if text == "NO_VISUALS":
            return ""
        return f"[Page {page_num + 1} — Visual Description]\n{text}"

    except Exception:
        return ""   


def extract_tables_from_pdf(pdf_bytes: bytes) -> dict:
    """
    Extract all tables from PDF without any API call.
    Returns { page_index: "formatted table as text" }
    """
    tables_by_page = {}
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if not tables:
                    continue
                table_texts = []
                for table in tables:
                    rows = []
                    for row in table:
                        clean_row = [cell or "" for cell in row]
                        rows.append(" | ".join(clean_row))
                    table_texts.append("\n".join(rows))
                tables_by_page[page_num] = (
                    f"[Page {page_num + 1} — Table]\n" +
                    "\n\n".join(table_texts)
                )
    except Exception:
        pass
    return tables_by_page


def process_pdf(file, progress_bar) -> List[str]:
    """
    Full multimodal pipeline per page:
      1. Extract plain text   → free, local
      2. Extract tables       → free, local
      3. Describe visuals     → Gemini Vision (only if page has visuals)
    Returns unified chunk list ready for embedding.
    """
    pdf_bytes = file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)

    # Extract all tables in one free local pass before the main loop
    tables_by_page = extract_tables_from_pdf(pdf_bytes)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_chunks: List[str] = []
    vision_calls = 0

    for page_num in range(total_pages):
        page = doc[page_num]

        # 1. Plain text
        page_text = page.get_text("text") or ""
        if page_text.strip():
            all_chunks.extend(text_splitter.split_text(page_text))

        # 2. Tables (already extracted, just append)
        if page_num in tables_by_page:
            all_chunks.append(tables_by_page[page_num])

        # 3. Vision — ONLY if page actually has images or diagrams
        if page_has_visuals(page):
            img_bytes = page_to_image_bytes(page)
            desc = describe_page_visuals(img_bytes, page_num)
            if desc:
                all_chunks.append(desc)
                vision_calls += 1
                # Small delay to stay within Gemini free tier rate limits
                # (15 RPM free — 1 second gap keeps us well within limits)
                time.sleep(1)

        progress_bar.progress(
            (page_num + 1) / total_pages,
            text=f"📄 Page {page_num + 1}/{total_pages}  |  Vision calls: {vision_calls}"
        )

    doc.close()
    return all_chunks


# ════════════════════════════════════════════════════════════════════════════
#  STEP 5 — VECTOR DATABASE  (Chroma, local in-memory)
#  ─────────────────────────────────────────────────────────────────────────
#  Why in-memory (not persisted to disk)?
#  This app is designed for user-uploaded PDFs that change every session.
#  Persisting to disk only makes sense for a fixed knowledge base.
#  In-memory + @st.cache_resource = fast within a session, clean between.
#
#  RAM optimization: raw chunks deleted immediately after Chroma is built.
#  Chroma only needs the vectors — the original strings are redundant.
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def create_vector_db(file_hash: str, chunks: tuple):
    embeddings = GeminiEmbeddings()
    db = Chroma.from_texts(list(chunks), embeddings)
    # chunks tuple is held by cache — but the local list is freed after this
    return db


# ════════════════════════════════════════════════════════════════════════════
#  STEP 6 — CHAT  (Groq)
#  ─────────────────────────────────────────────────────────────────────────
#  Groq runs Llama 3.3 70B on LPU chips — extremely fast responses.
#  Free tier: ~14,400 requests/day — far more than enough.
#  Model rotation: if one hits rate limit, silently tries next.
# ════════════════════════════════════════════════════════════════════════════

GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # best quality, use first
    "llama-3.1-8b-instant",      # fastest, first fallback
    "mixtral-8x7b-32768",        # long context, last fallback
]

SYSTEM_PROMPT = """You are an expert research paper assistant.
You have full access to the paper's text, tables, figures, equations, and diagrams.

RULES:
1. Answer concisely — max 4-5 sentences or bullet points.
2. Reference figures/tables explicitly when relevant (e.g. "As shown in Table 2...").
3. If the answer is not in the paper, say so clearly and answer from general knowledge.
4. Never fabricate data, numbers, or statistics not present in the context."""


def ask_groq(context: str, question: str) -> str:
    """
    Retrieve-then-generate:
    Top 5 relevant chunks (context) + user question → Groq → answer.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context from the paper:\n{context}\n\nQuestion: {question}"
        }
    ]

    for i, model in enumerate(GROQ_MODELS):
        try:
            resp = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=512,    # concise answers
                temperature=0.2,   # low = factual, high = creative
            )
            return resp.choices[0].message.content

        except Exception as e:
            msg = str(e).lower()
            is_rate_limit = "429" in msg or "rate_limit" in msg
            if is_rate_limit and i < len(GROQ_MODELS) - 1:
                continue   # try next model silently
            if i == len(GROQ_MODELS) - 1:
                raise Exception("ALL_QUOTA_EXCEEDED")
            raise e

    raise Exception("ALL_QUOTA_EXCEEDED")


# ════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="PaperChat", page_icon="📄")
st.title("📄 PaperChat")
st.caption("Multimodal RAG  ·  Gemini (embeddings + vision)  ·  Groq (chat)")

uploaded_file = st.file_uploader("Upload your Research Paper (PDF)", type="pdf")

if uploaded_file:

    # MD5 hash of file bytes = unique ID for caching
    # Same paper re-uploaded → instant load, zero API calls
    raw_bytes = uploaded_file.read()
    file_hash = hashlib.md5(raw_bytes).hexdigest()
    uploaded_file.seek(0)   # reset pointer so processors can read the file

    # Only process if this is a NEW paper
    if (
        "docsearch" not in st.session_state
        or st.session_state.get("loaded_file") != file_hash
    ):
        st.session_state.chat_history = []   # clear chat for new paper

        progress_bar = st.progress(0, text="📖 Starting multimodal processing…")
        chunks = process_pdf(uploaded_file, progress_bar)

        progress_bar.progress(1.0, text="🧠 Building vector index…")
        st.session_state.docsearch = create_vector_db(file_hash, tuple(chunks))
        st.session_state.loaded_file = file_hash
        st.session_state.total_chunks = len(chunks)

        # Free raw chunks from local scope — Chroma has everything it needs
        del chunks

        progress_bar.empty()
        st.success(
            f"✅ Ready! Indexed **{st.session_state.total_chunks}** chunks "
            f"(text + tables + visual descriptions)"
        )

    docsearch = st.session_state.docsearch

    # Render previous chat messages
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

    question = st.chat_input("Ask about text, figures, tables, equations…")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Thinking…"):
            try:
                # Retrieve top 5 most relevant chunks from vector DB
                retriever = docsearch.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(question)
                context = "\n\n".join(d.page_content for d in docs)

                # Generate answer using Groq
                answer = ask_groq(context, question)

                with st.chat_message("assistant"):
                    st.write(answer)

                st.session_state.chat_history.append(("user", question))
                st.session_state.chat_history.append(("assistant", answer))

            except Exception as e:
                if "ALL_QUOTA_EXCEEDED" in str(e):
                    st.warning("⏳ Rate limit hit on all Groq models!")
                    st.info("💡 Wait a minute and try again.")
                else:
                    st.error("Something went wrong. Please try again!")
                    st.exception(e)
