import streamlit as st

from dd_copilot.cli import build_provider
from dd_copilot.pipeline import analyze

st.set_page_config(page_title="DD-Copilot", layout="centered")
st.title("DD-Copilot — Technical Due Diligence")
st.caption(
    "Paste a URL, upload a PDF, or paste text from public material of a "
    "deep-tech startup. DD-Copilot generates a technical due-diligence "
    "report with citations verified against the original source."
)

source_type = st.radio("Source type", ["URL", "Pasted text", "PDF"], horizontal=True)

source_input = None
if source_type == "URL":
    source_input = st.text_input("Website or whitepaper URL")
elif source_type == "Pasted text":
    source_input = st.text_area("Paste the text here", height=200)
else:
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        source_input = temp_path

llm_choice = st.radio("LLM provider", ["Claude", "Ollama (local)"], horizontal=True)

if st.button("Analyze", disabled=not source_input):
    with st.spinner("Analyzing public material..."):
        provider = build_provider("claude" if llm_choice == "Claude" else "ollama")
        markdown = analyze(source_input, provider)
    st.markdown(markdown)
