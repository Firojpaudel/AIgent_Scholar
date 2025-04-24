import streamlit as st
import os
import json
import time
import logging
import arxiv
import shutil
import glob
import atexit
import tempfile
from retry import retry
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.langchain import LangChainLLM
from llama_index.core import StorageContext
import chromadb
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
import requests
import fitz  # PyMuPDF
from dotenv import load_dotenv
from pathlib import Path
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(filename="paper_fetcher.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Check for GPU availability
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logging.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("No GPU available, falling back to CPU")
except Exception as e:
    logging.error(f"Error checking GPU availability: {e}")
    device = "cpu"

logging.info(f"Using device: {device}")

# Initialize embedding and LLM models
try:
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-mpnet-base-v2",
        device=device
    )
    logging.info("Successfully initialized HuggingFaceEmbedding model")
except Exception as e:
    logging.error(f"Failed to initialize HuggingFaceEmbedding: {e}")
    st.error("Failed to initialize the embedding model. Please check the logs for details.")
    st.stop()

# Optimize batch size for embedding
try:
    Settings.embed_model._model._target_device = device
    Settings.embed_model._model.config.batch_size = 128 if device == "cuda" else 32
except Exception as e:
    logging.warning(f"Could not set batch size for embedding model: {e}")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
Settings.llm = LangChainLLM(llm=llm)

# Set Google API key
if os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
else:
    st.error("Google API key not found in .env file")
    st.stop()

# Cleanup function for PDFs, cache files, and ChromaDB
def cleanup_files():
    try:
        logging.info("Starting cleanup_files on app shutdown")
        if os.path.exists("./papers"):
            shutil.rmtree("./papers")
            logging.info("Deleted ./papers directory")
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
            logging.info("Deleted ./chroma_db directory")
        cache_files = list(Path.cwd().glob("papers_cache_*.json"))
        logging.info(f"Found cache files to delete: {[str(f) for f in cache_files]}")
        for cache_file in cache_files:
            try:
                cache_file.unlink(missing_ok=True)
                logging.info(f"Deleted cache file: {cache_file}")
            except Exception as e:
                logging.error(f"Failed to delete cache file {cache_file}: {e}")
        logging.info("Completed cleanup_files")
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")

# Register cleanup function to run on app shutdown
atexit.register(cleanup_files)

# Clean up any existing cache files on app start (fallback)
try:
    logging.info("Cleaning up existing cache files on app start")
    cache_files = list(Path.cwd().glob("papers_cache_*.json"))
    logging.info(f"Found cache files on start: {[str(f) for f in cache_files]}")
    for cache_file in cache_files:
        try:
            cache_file.unlink(missing_ok=True)
            logging.info(f"Deleted cache file on start: {cache_file}")
        except Exception as e:
            logging.error(f"Failed to delete cache file on start {cache_file}: {e}")
except Exception as e:
    logging.error(f"Error during startup cleanup: {e}")

# Streamlit app layout
st.title("AIgent Scholar")
st.markdown("An Agentic RAG Chatbot for Research Paper Search and Query")

# Add MathJax for LaTeX rendering and custom CSS for UI improvement
st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script>
    // Force MathJax to reprocess the page after Streamlit renders
    document.addEventListener("DOMContentLoaded", function() {
        if (typeof MathJax !== "undefined") {
            MathJax.typesetPromise().then(() => {
                console.log("MathJax typesetting complete");
            }).catch((err) => console.log("MathJax typesetting error:", err));
        }
    });
    // Re-run MathJax when the sidebar updates
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (typeof MathJax !== "undefined") {
                MathJax.typesetPromise().then(() => {
                    console.log("MathJax reprocessed sidebar");
                }).catch((err) => console.log("MathJax reprocessing error:", err));
            }
        });
    });
    observer.observe(document.body, { childList: true, subtree: true });
</script>
<style>
    .reportview-container {
        background-color: #f9f9f9;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95% !important; /* Increase main content width */
    }
    h1 {
        font-size: 2.5rem;
        text-align: center;
        color: #2c3e50;
    }
    h2 {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: #34495e;
    }
    .stButton > button {
        background-color: #007BFF;
        color: white;
        padding: 8px 16px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        background-color: #0056b3;
        transform: scale(1.05);
    }
    .stTextInput > div > input {
        border-radius: 5px;
        border: 1px solid #ccc;
        padding: 8px;
    }
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        background-color: #ffffff;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .chat-message {
        padding: 10px 15px;
        margin: 8px 0;
        border-radius: 10px;
        max-width: 80%;
        line-height: 1.5;
        color: #333333 !important;
        opacity: 1 !important;
        font-size: 14px;
    }
    .user-message {
        background-color: #e6f3ff;
        margin-left: auto;
        margin-right: 10px;
        text-align: right;
    }
    .assistant-message {
        background-color: #f0f0f0;
        margin-right: auto;
        margin-left: 10px;
        text-align: left;
    }
    .trash-button > button {
        background-color: #FF6B6B;
        color: white;
        padding: 5px 10px;
        border: none;
        border-radius: 5px;
        font-size: 12px;
        min-height: 30px;
        min-width: 30px;
    }
    .trash-button > button:hover {
        background-color: #e55a5a;
        transform: scale(1.05);
    }
    .sidebar .paper-list {
        list-style-type: disc;
        padding-left: 20px;
        margin-bottom: 10px;
    }
    .sidebar .paper-list-item {
        animation: slideIn 0.5s ease-in-out;
        margin-bottom: 8px;
        font-size: 14px;
        color: #2c3e50;
        transition: all 0.3s ease;
    }
    .sidebar .paper-list-item a {
        color: inherit;
        text-decoration: none;
        cursor: pointer;
    }
    .sidebar .paper-list-item a:hover {
        color: #007BFF;
        text-decoration: underline;
        transform: translateX(5px);
        display: inline-block;
    }
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .reportview-container {
            background-color: #1e1e1e;
        }
        h1, h2 {
            color: #ffffff;
        }
        .chat-container {
            background-color: #2a2a2a;
            border: 1px solid #444444;
        }
        .chat-message {
            color: #e0e0e0 !important;
        }
        .user-message {
            background-color: #1e3a5f;
        }
        .assistant-message {
            background-color: #3a3a3a;
        }
        .sidebar .paper-list-item {
            color: #e0e0e0;
        }
        .sidebar .paper-list-item a:hover {
            color: #66b3ff;
        }
    }
    /* Responsive adjustments for smaller screens */
    @media (max-width: 768px) {
        .main .block-container {
            max-width: 100% !important;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "papers" not in st.session_state:
    st.session_state.papers = []
if "index" not in st.session_state:
    st.session_state.index = None
if "chroma_collection" not in st.session_state:
    st.session_state.chroma_collection = None
if "graph" not in st.session_state:
    st.session_state.graph = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_titles" not in st.session_state:
    st.session_state.indexed_titles = []

# Function to escape HTML characters and format LaTeX for rendering
def format_title_for_display(title):
    title = title.replace("&", "&").replace("<", "<").replace(">", ">")
    # Keep single $...$ for inline LaTeX rendering
    return title

# Fetch papers from arXiv
@retry(tries=1, delay=1, backoff=2)
def fetch_arxiv_papers(query, max_results=5):
    try:
        start_time = time.time()
        logging.info(f"Starting arXiv fetch for query: {query}")
        client = arxiv.Client()
        search_query = query
        if query.lower() == "model context protocol":
            search_query = 'ti:"Model Context Protocol" OR abs:"Model Context Protocol"'
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        papers = []
        for result in client.results(search):
            paper = {
                "title": result.title,
                "abstract": result.summary,
                "url": result.pdf_url,
                "published": result.published.isoformat(),
                "source": "arXiv"
            }
            papers.append(paper)
            logging.info(f"Fetched paper: {paper['title']}")
        logging.info(f"Fetched {len(papers)} papers from arXiv in {time.time() - start_time:.2f} seconds")
        return papers
    except Exception as e:
        logging.error(f"Error fetching arXiv papers: {e}")
        return []

def fetch_all_papers(query, max_results=5):
    try:
        start_time = time.time()
        logging.info(f"Starting fetch_all_papers for query: {query}")
        cache_file = f"papers_cache_{query.replace(' ', '_')}.json"
        if query.lower() == "model context protocol":
            if os.path.exists(cache_file):
                os.remove(cache_file)
                logging.info(f"Cleared cache file: {cache_file}")
        elif os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cached_papers = json.load(f)
                if cached_papers.get("query") == query and cached_papers.get("max_results") == max_results:
                    logging.info(f"Returning cached papers in {time.time() - start_time:.2f} seconds")
                    return cached_papers["papers"]
        papers = fetch_arxiv_papers(query, max_results)
        with open(cache_file, "w") as f:
            json.dump({"query": query, "max_results": max_results, "papers": papers}, f)
        logging.info(f"Cached {len(papers)} papers in {time.time() - start_time:.2f} seconds")
        return papers
    except Exception as e:
        logging.error(f"Error combining papers: {e}")
        return []

@retry(tries=3, delay=2, backoff=2)
def download_pdf(url, filename):
    start_time = time.time()
    logging.info(f"Starting PDF download: {url}")
    response = requests.get(url, stream=True, timeout=5)
    response.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    logging.info(f"Downloaded PDF {filename} in {time.time() - start_time:.2f} seconds")
    return filename

def process_pdf(i, paper):
    start_time = time.time()
    filename = f"./papers/paper_{i}.pdf"
    pdf_text = None
    try:
        download_pdf(paper["url"], filename)
        pdf_doc = fitz.open(filename)
        pdf_text = ""
        for page in pdf_doc:
            pdf_text += page.get_text("text") + "\n"
        pdf_doc.close()
        if not pdf_text.strip():
            logging.warning(f"No text extracted from PDF {filename}")
            pdf_text = None
    except Exception as e:
        logging.error(f"Failed to process PDF {filename}: {e}")
    logging.info(f"Processed PDF for paper {paper['title']} in {time.time() - start_time:.2f} seconds")
    return i, paper, pdf_text

def index_papers(papers, collection_name="research_papers"):
    try:
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
            logging.info("Deleted ./chroma_db directory")
        start_time = time.time()
        logging.info(f"Starting indexing for collection: {collection_name}")
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        embed_model = Settings.embed_model
        test_embedding = embed_model.get_text_embedding("test")
        expected_dim = len(test_embedding)
        logging.info(f"Expected embedding dimension: {expected_dim}")
        chroma_collection = chroma_client.create_collection(name=collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        documents = []
        os.makedirs("./papers", exist_ok=True)

        # Process abstracts (CPU task, no parallelization needed)
        abstract_time = time.time()
        for i, paper in enumerate(papers):
            if not paper.get("title") or not paper.get("abstract"):
                logging.warning(f"Skipping paper {i} due to missing title or abstract")
                continue
            doc = Document(
                text=paper["abstract"],
                metadata={
                    "title": paper["title"],
                    "url": paper["url"],
                    "source": paper["source"],
                    "type": "abstract"
                }
            )
            documents.append(doc)
            logging.info(f"Added abstract for paper: {paper['title']}")
        logging.info(f"Processed abstracts in {time.time() - abstract_time:.2f} seconds")

        # Parallelize PDF downloads and text extraction
        pdf_time = time.time()
        pdf_tasks = [(i, paper) for i, paper in enumerate(papers) if paper["source"] == "arXiv" and paper["url"]]
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_paper = {executor.submit(process_pdf, i, paper): (i, paper) for i, paper in pdf_tasks}
            for future in as_completed(future_to_paper):
                i, paper, pdf_text = future.result()
                if pdf_text:
                    pdf_document = Document(
                        text=pdf_text,
                        metadata={
                            "title": paper["title"],
                            "url": paper["url"],
                            "source": paper["source"],
                            "type": "pdf"
                        }
                    )
                    documents.append(pdf_document)
                    logging.info(f"Added PDF for paper: {paper['title']} (length: {len(pdf_text)} chars)")
        logging.info(f"Processed PDFs in {time.time() - pdf_time:.2f} seconds")

        # Deduplication
        dedup_time = time.time()
        seen = set()
        unique_documents = []
        for doc in documents:
            identifier = f"{doc.metadata['title']}:{doc.metadata['type']}"
            if identifier not in seen:
                unique_documents.append(doc)
                seen.add(identifier)
            else:
                logging.warning(f"Duplicate document skipped: {identifier}")
        logging.info(f"Deduplication completed in {time.time() - dedup_time:.2f} seconds")

        # Indexing with embeddings (GPU-accelerated if available)
        embed_time = time.time()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents=unique_documents,
            storage_context=storage_context
        )
        total_docs = len(chroma_collection.get()["documents"])
        logging.info(f"Embedding and indexing completed in {time.time() - embed_time:.2f} seconds")
        logging.info(f"Total documents in chroma_collection after indexing: {total_docs}")
        logging.info(f"Indexed {len(unique_documents)} documents in collection {collection_name} in {time.time() - start_time:.2f} seconds")
        
        # Update indexed titles
        titles = list(set(paper["title"] for paper in papers))
        st.session_state.indexed_titles = titles
        logging.info(f"Set indexed_titles: {titles}")
        
        return index, chroma_collection
    except Exception as e:
        logging.error(f"Error indexing papers: {e}")
        return None, None

# LangGraph setup
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    documents: list

def retrieve(state):
    query = state["messages"][0].content
    retriever = st.session_state.index.as_retriever(similarity_top_k=10)
    docs = retriever.retrieve(query)
    logging.info(f"Retrieved {len(docs)} documents for query: {query}")
    for i, doc in enumerate(docs):
        logging.info(f"Document {i}: {doc.text[:200]}... Metadata: {doc.metadata}")
    return {"documents": [doc.text for doc in docs], "messages": state["messages"]}

def grade_documents(state):
    docs = state["documents"]
    query = state["messages"][0].content
    docs_input = "\n".join([f"Document {i}: {doc[:200]}..." for i, doc in enumerate(docs)])
    logging.info(f"Grading input - Query: {query}\nDocuments:\n{docs_input}")
    prompt = PromptTemplate(
        input_variables=["query", "docs"],
        template="Given the query '{query}', determine if the following documents contain information that could be relevant to answering it. Respond with 'yes' or 'no' only.\nDocuments:\n{docs}"
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": query, "docs": docs_input})
    logging.info(f"Document grading result: {response}")
    return {"documents": docs if response.lower() == "yes" else [], "messages": state["messages"]}

def generate(state):
    docs = state["documents"]
    query = state["messages"][0].content
    if not docs:
        response = "No relevant documents found in the index. Please try fetching papers with a more specific query."
    else:
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="Answer the question based on the context:\nContext: {context}\nQuestion: {question}"
        )
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"context": "\n".join(docs), "question": query})
    logging.info(f"Generated response for query: {query}")
    return {"messages": [HumanMessage(content=response)]}

def web_search(state):
    query = state["messages"][0].content
    logging.warning("Web search not implemented")
    return {"documents": [], "messages": state["messages"]}

def route(state):
    docs = state["documents"]
    return "generate" if docs else "web_search"

def setup_langgraph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("web_search", web_search)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", route, {"generate": "generate", "web_search": "web_search"})
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()

# Sidebar for indexed papers
with st.sidebar:
    st.header("Indexed Papers")
    if st.session_state.indexed_titles and st.session_state.papers:
        # Create a mapping of titles to URLs
        title_to_url = {paper["title"]: paper["url"] for paper in st.session_state.papers if paper["url"]}
        with st.expander("View Indexed Papers", expanded=True):
            # Create the list of clickable titles linking directly to PDFs
            list_items = ""
            for title in st.session_state.indexed_titles:
                formatted_title = format_title_for_display(title)
                pdf_url = title_to_url.get(title, "#")
                list_items += f'<li class="paper-list-item"><a href="{pdf_url}" target="_blank">{formatted_title}</a></li>'
            st.markdown(
                f'<ul class="paper-list">{list_items}</ul>',
                unsafe_allow_html=True
            )
    else:
        st.write("No papers indexed yet.")

# Main app content
# Fetch papers section
st.header("Fetch Research Papers")
st.markdown("---")
topic = st.selectbox("Select a topic:", [
    "Artificial Intelligence",
    "AI Agents",
    "Machine Learning",
    "Quantum Computing",
    "Robotics",
    "Natural Language Processing",
    "Computer Vision",
    "Other"
])
custom_topic = st.text_input("Enter custom topic:", "", disabled=topic != "Other") if topic == "Other" else ""
query = custom_topic if topic == "Other" and custom_topic else topic
max_results = st.number_input("Maximum number of papers:", min_value=1, max_value=10, value=5)

col1, col2 = st.columns([3, 1])
with col1:
    if st.button("Fetch Papers"):
        with st.spinner("Fetching papers..."):
            for cache_file in glob.glob(f"papers_cache_{query.replace(' ', '_')}.json"):
                os.remove(cache_file)
                logging.info(f"Cleared cache file: {cache_file}")
            st.session_state.papers = fetch_all_papers(query, max_results)
            if st.session_state.papers:
                st.success(f"Fetched {len(st.session_state.papers)} papers!")
                with st.expander("Fetched Papers"):
                    for paper in st.session_state.papers:
                        st.markdown(f"**Title**: {paper['title']}")
                        st.markdown(f"**Abstract**: {paper['abstract']}")
                        st.markdown(f"**URL**: [{paper['url']}]({paper['url']})")
                        st.markdown(f"**Source**: {paper['source']}")
                        st.markdown("---")
            else:
                st.error("No papers fetched. Please try again.")
with col2:
    if st.button("🗑️ Clear Database", key="clear_db", help="Clear ChromaDB", type="primary"):
        with st.spinner("Clearing Chroma database..."):
            try:
                if os.path.exists("./chroma_db"):
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        shutil.move("./chroma_db", os.path.join(tmpdirname, "chroma_db"))
                    st.session_state.index = None
                    st.session_state.chroma_collection = None
                    st.session_state.graph = None
                    st.session_state.indexed_titles = []
                    logging.info("Deleted Chroma database: ./chroma_db")
                    st.success("Chroma database cleared successfully!")
                else:
                    st.info("No Chroma database found.")
            except Exception as e:
                logging.error(f"Error clearing ChromaDB: {e}")
                st.error(f"Failed to clear Chroma database: {e}")

# Index papers section
st.header("Index Papers")
st.markdown("---")
if st.button("Index Fetched Papers"):
    if st.session_state.papers:
        with st.spinner("Indexing papers..."):
            st.session_state.index, st.session_state.chroma_collection = index_papers(
                st.session_state.papers, collection_name="research_papers"
            )
            if st.session_state.index:
                st.success(f"Indexed {len(st.session_state.papers)} papers in collection {st.session_state.chroma_collection.name}")
                st.session_state.graph = setup_langgraph()
                st.rerun()
            else:
                st.error("Indexing failed.")
    else:
        st.error("No papers to index. Please fetch papers first.")

# Chatbot UI for querying
st.header("Query Indexed Papers")
st.markdown("---")
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        with st.container():
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

question = st.text_input("Ask a question about the papers:", key="query_input")
if st.button("Send"):
    if question:
        if st.session_state.index and st.session_state.graph:
            with st.spinner("Retrieving answer..."):
                st.session_state.chat_history.append({"role": "user", "content": question})
                inputs = {"messages": [HumanMessage(content=question)]}
                result = st.session_state.graph.invoke(inputs)
                answer = result["messages"][-1].content
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()
        else:
            st.error("Please index papers before querying.")
    else:
        st.warning("Please enter a question.")