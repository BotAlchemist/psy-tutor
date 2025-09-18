import os
from io import BytesIO

import streamlit as st
import pdfplumber

# ---- LLM client (OpenAI SDK v1.x) ----
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None



# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="📘 PDF Tutor (Chapters Folder)", page_icon="📘", layout="centered")
st.title("📘 PDF Tutor")
st.caption("Reads PDFs from ./psychology_book → pick chapter → choose page → ask a guided question.")

#-------- Settings / API Key ----------
with st.sidebar:
    st.header("Settings")
    #api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    api_key= os.getenv("OPENAI_API_KEY")
    model_name = st.text_input("Model", value="gpt-4o-mini")
    show_page_text = st.checkbox("Show extracted page text (current page only)", value=False)
    st.markdown("---")
    st.caption("The model will read the selected page **plus** the previous and next page (if they exist).")

# --------- Locate chapters (PDFs) ----------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BOOK_DIR = os.path.join(ROOT_DIR, "psychology_book")

if not os.path.isdir(BOOK_DIR):
    st.error("Folder `psychology_book` not found next to this app. Create it and add PDF chapters.")
    st.stop()

def _list_pdfs(path: str):
    files = []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isfile(full) and name.lower().endswith(".pdf"):
            files.append(full)
    # Sort by filename; rename like 01_Chapter.pdf, 02_Chapter.pdf for clean order
    return sorted(files, key=lambda p: os.path.basename(p).lower())

pdf_paths = _list_pdfs(BOOK_DIR)
if not pdf_paths:
    st.warning("No PDFs found in `psychology_book`. Add chapter PDFs and refresh.")
    st.stop()

chapter_path = st.selectbox("Choose a chapter (PDF)", pdf_paths, format_func=os.path.basename)

# --------- Read chosen chapter & extract pages ----------
@st.cache_data(show_spinner=False)
def read_pdf_pages(file_path: str, mtime: float):
    """Return list of page texts for a PDF path; cache busts on file mtime."""
    texts = []
    with open(file_path, "rb") as f:
        data = f.read()
    with pdfplumber.open(BytesIO(data)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            texts.append(txt.strip())
    return texts

mtime = os.path.getmtime(chapter_path)
pages = read_pdf_pages(chapter_path, mtime)
num_pages = len(pages)

if num_pages == 0:
    st.error("Couldn't read any pages from this PDF.")
    st.stop()

# --- dropdown for page selection ---
page_options = list(range(1, num_pages + 1))
page_idx = st.selectbox("Choose a page", page_options, index=0, format_func=lambda i: f"Page {i}")

# Build context window: previous, current, next page (for LLM)
context_texts = []
if page_idx > 1:
    context_texts.append(f"[Prev: Page {page_idx-1}]\n{pages[page_idx - 2]}")
context_texts.append(f"[Current: Page {page_idx}]\n{pages[page_idx - 1]}")
if page_idx < num_pages:
    context_texts.append(f"[Next: Page {page_idx+1}]\n{pages[page_idx]}")

# Join with clear markers
context_text = "\n\n-----\n\n".join([t for t in context_texts if t])

# Preview: ONLY the current page (optional)
if show_page_text:
    with st.expander(f"Show extracted text for page {page_idx}"):
        curr_text = pages[page_idx - 1]
        st.write(curr_text if curr_text else "_No text extracted from this page._")

# ------------------ Guided Tutor Templates ------------------
QUESTION_TEMPLATES = {
    "Summarize key points": "Summarize the most important ideas on this page in simple bullet points for a 9th grade student.",
    "Explain like I'm in high school": "Explain the page content in very simple words for a high school student. Avoid jargon. Use short sentences and small examples.",
    "Explain like I'm in elementary school": "Explain the page as if I am in grade 5. Use everyday examples and comparisons a child understands.",
    "Make quiz questions": "Create 3 short multiple-choice questions (with 4 options each) from this page. Mark the correct answers clearly in the end.",
    "Define difficult words": "List any difficult or technical words on this page and define them in very simple terms.",
    "Step-by-step explanation": "Break down the content of this page step-by-step in clear, numbered points.",
    "Real-life example": "Give one simple real-life example that matches the idea on this page. Keep it relevant for a school student.",
    "Quick recap (30 seconds)": "Give a 4–6 bullet ‘quick recap’ of this page as if I only have 30 seconds before a test.",
}

st.subheader("Ask about this page")

mode = st.selectbox(
    "Choose a help option",
    ["Custom question"] + list(QUESTION_TEMPLATES.keys())
)

if mode == "Custom question":
    question = st.text_input("Type your question here")
else:
    question = QUESTION_TEMPLATES[mode]
    st.info(f"Template selected: **{mode}**")

ask = st.button("Ask")

# ------------------ LLM Call ------------------
def call_llm(api_key: str, model: str, context_text: str, question: str) -> str:
    """Send a grounded, student-friendly prompt to the LLM using the ±1 page context."""
    if OpenAI is None:
        return "OpenAI SDK not found. Run: pip install openai>=1.0.0"
    if not api_key:
        return "No API key provided. Add it in the sidebar or set OPENAI_API_KEY."

    client = OpenAI(api_key=api_key)

    system = (
        "You are a patient, friendly tutor for school students (elementary to high school). "
        "Answer as simply as possible using ONLY the provided text (which includes the selected page plus its neighbors). "
        "If the answer is not in the provided text, say: 'I can't find this in the provided pages.' "
        "Prefer bullet points, short sentences, and clear examples. Avoid jargon."
    )

    # Keep context bounded (protect against very long pages)
    bounded = (context_text or "")

    user = (
        f"Context (selected page with ±1 neighbors):\n---\n{bounded}\n---\n\n"
        f"Student request: {question}\n\n"
        "Rules:\n"
        "1) Use ONLY the text above.\n"
        "2) Keep it simple for a school student.\n"
        "3) Use bullets or short steps when helpful.\n"
        "4) If info is missing, say you can't find it in the provided pages."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {e}"

if ask:
    if not question.strip():
        st.warning("Please type or select a question.")
    else:
        with st.spinner("Thinking..."):
            answer = call_llm(api_key, model_name, context_text, question)
        st.markdown("**Answer:**")
        st.write(answer)





