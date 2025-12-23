# contr.py - Fully Fixed & Deploy-Ready Legal Contract Analyzer

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os

# ========================
# Page Config & Header
# ========================

st.set_page_config(page_title="AI Legal Contract Analyzer", layout="wide")
st.title("📜 Contract-Sense")
st.markdown("Upload one or more contract PDFs → Get instant summary, clause analysis, and entity extraction.")

# ========================
# Privacy Warning
# ========================

if "privacy_warning_shown" not in st.session_state:
    st.warning("""
    🔒 **Privacy Notice**  
    Your contract text will be sent to the LLM provider (Groq) using **your own API key**.  
    Nothing is stored on the server. For maximum privacy, use your personal key.  
    Do NOT upload highly confidential documents unless you trust the provider.
    """)
    st.session_state.privacy_warning_shown = True
# ========================
# Sidebar: API Key + File Upload
# ========================


st.header("🔑 Your Groq API Key")
# Check if user has already entered a valid key this session
if "api_key_entered" not in st.session_state:
    st.session_state.api_key_entered = False
    st.session_state.api_key = ""

# If key not entered yet → show input field
if not st.session_state.api_key_entered:
    api_key = st.text_input(
        "Enter your Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get free at: https://console.groq.com/keys"
    )

    if api_key and api_key.startswith("gsk_"):  # Basic validation
        st.session_state.api_key = api_key
        st.session_state.api_key_entered = True
        st.success("✅ API key saved! (securely in your browser session)")
        st.rerun()  # Refresh to hide the input
    elif api_key:
        st.error("Invalid key format. Should start with 'gsk_'")

else:
    # Key already entered → show confirmation instead of input
    st.success("✅ Connected with your Groq API key")
    st.caption("Your key is securely stored only in this session.")
    
    # Optional: Add a "Change Key" button
    if st.button("Change API Key"):
        st.session_state.api_key_entered = False
        st.session_state.api_key = ""
        st.rerun()

# Use the stored key from now on
api_key = st.session_state.get("api_key", "")

if not api_key:
    st.stop()  # Stop app until key is provided

st.header("📄 Upload Contract PDFs")
uploaded_files = st.file_uploader(
    "Choose PDF files (multiple allowed)",
    type="pdf",
    accept_multiple_files=True,
    help="Files are processed in memory only — nothing saved to disk"
)

if uploaded_files and st.button("🔄 Process New Files (Clear Cache)"):
    # Clear cached vectorstore when user uploads new files
    st.cache_resource.clear()
    st.success("Cache cleared — processing new files...")
    st.rerun()

# ========================
# Initialize Session State
# ========================

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

# Update session state when new files are uploaded
if uploaded_files is not None:
    st.session_state.uploaded_files = uploaded_files

# ========================
# Check if Files Are Ready
# ========================

if st.session_state.uploaded_files is None or len(st.session_state.uploaded_files) == 0:
    st.info("👆 Please upload at least one PDF contract in the sidebar to begin analysis.")
    st.stop()

current_files = st.session_state.uploaded_files
st.success(f"✅ {len(current_files)} contract file(s) loaded and ready!")

# ========================
# Build Vector Store (In-Memory)
# ========================

@st.cache_resource(show_spinner="Building search index from contracts...")
def build_vectorstore(_files):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for uploaded_file in _files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load_and_split(splitter))
        os.unlink(tmp_path)  # Clean up temp file

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

vectorstore = build_vectorstore(current_files)

# ========================
# Load Groq LLM
# ========================

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=1200,
        groq_api_key=api_key
    )

llm = load_llm()

# ========================
# Helper: Get Context
# ========================

def get_context(k=25):
    docs = vectorstore.similarity_search("contract overview parties dates obligations", k=k)
    return "\n\n".join([doc.page_content for doc in docs])

context = get_context()

# ========================
# Tabs
# ========================

tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "📑 Key Clauses", "🔍 Named Entities"])

with tab1:
    st.header("Contract Overview & Risk Summary")
    with st.spinner("Generating professional summary..."):
        prompt = ChatPromptTemplate.from_template("""
You are a senior legal analyst. Provide a clear, structured summary:

1. Contract Type
2. Parties Involved
3. Effective Date & Duration/Termination
4. Overall Risk Level: Low / Medium / High (explain why)
5. Key Obligations of Each Party
6. Any Red Flags or Unusual Terms

Contract Text:
{context}
""")
        chain = prompt | llm | StrOutputParser()
        st.markdown(chain.invoke({"context": context}))

with tab2:
    st.header("Key Clauses Breakdown")
    clauses = [
        "Parties", "Term & Termination", "Payment Terms", "Confidentiality",
        "Intellectual Property", "Liability & Indemnification", "Governing Law", "Dispute Resolution"
    ]

    for clause in clauses:
        with st.expander(f"📌 {clause}"):
            query = f"Extract and summarize the section about {clause.lower()}"
            docs = vectorstore.similarity_search(query, k=4)
            clause_context = "\n\n".join([d.page_content for d in docs])

            if not clause_context.strip():
                st.info("No relevant section found.")
                continue

            prompt = ChatPromptTemplate.from_template("""
Summarize this clause professionally:

Clause: {clause}
Text:
{clause_context}

Provide:
- Summary
- Risk Level (Low/Medium/High)
- Key Quote
""")
            chain = prompt | llm | StrOutputParser()
            st.markdown(chain.invoke({"clause": clause, "clause_context": clause_context}))

with tab3:
    st.header("Extracted Named Entities")
    with st.spinner("Extracting parties, dates, amounts, etc..."):
        prompt = ChatPromptTemplate.from_template("""
Extract ALL key entities from the contract in a clean Markdown table:

| Entity Type         | Value                          | Context/Quote                          |
|---------------------|--------------------------------|----------------------------------------|
| Parties             | ...                            | ...                                    |
| Effective Date      | ...                            | ...                                    |
| Termination Date    | ...                            | ...                                    |
| Payment Amounts     | ...                            | ...                                    |
| Notice Period       | ...                            | ...                                    |
| Governing Law       | ...                            | ...                                    |
| Jurisdiction        | ...                            | ...                                    |
| Signatories         | ...                            | ...                                    |

Contract:
{context}

Only include entities actually present.
""")
        chain = prompt | llm | StrOutputParser()
        st.markdown(chain.invoke({"context": context}))

# ========================
# Footer
# ========================

st.success("✅ Analysis Complete!")
st.caption("This is AI-assisted review • Always have a qualified lawyer verify critical contracts.")