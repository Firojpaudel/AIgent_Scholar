# AIgent_Scholar: Agentic RAG for Fetching AI Agent Papers 📚

**AIgent_Scholar** is a Streamlit-powered web app that fetches, indexes, and queries the latest research papers from arXiv, focusing on AI agents and beyond. It started as a Jupyter Notebook for fetching AI agent papers but evolved into a full-fledged app that lets you explore papers on any topic, index them efficiently, and query them using a smart chatbot interface. Built with LlamaIndex, LangGraph, and ChromaDB, this app leverages agentic Retrieval-Augmented Generation (RAG) to streamline your research experience.

---

## Features

- **Paper Fetching**: Retrieve the latest papers from arXiv on any topic, with a focus on AI agents.
- **Flexible Topic Selection**: Choose from predefined topics (e.g., AI Agents, Machine Learning, Quantum Computing) or enter a custom topic.
- **Indexing with LlamaIndex**: Efficiently index fetched papers using LlamaIndex and ChromaDB for fast retrieval.
- **Agentic RAG with LangGraph**: A LangGraph-powered workflow routes queries through retrieval, grading, and generation steps for accurate answers.
- **Chatbot Interface**: Query indexed papers via a user-friendly chat interface built with Streamlit.
- **Direct PDF Access**: Click paper titles in the sidebar to open PDFs in a new tab for easy reading.
- **Backend Optimization**: Parallel PDF downloads, GPU-accelerated indexing (if available), and cache management for improved performance.

---

## Demo

![AIgent_Scholar in Action](./Images/Projectdemo.gif)

---

## Tech Stack

- **Streamlit**: Frontend framework for the web app.
- **arXiv API**: Fetches research papers from arXiv.
- **LlamaIndex**: Indexing and retrieval of papers for RAG.
- **LangGraph**: Powers the agentic RAG workflow for query routing.
- **LangChain**: Used for the Google Gemini LLM (`gemini-1.5-flash`) to generate answers.
- **ChromaDB**: Backend vector store for efficient paper indexing.
- **PyMuPDF (fitz)**: Extracts text from PDFs for indexing.
- **HuggingFace Embeddings**: Generates embeddings using `sentence-transformers/all-mpnet-base-v2`.
- **Google Gemini LLM**: `gemini-1.5-flash` for answering user queries.
- **Python Libraries**: `requests`, `torch`, `retry`, and more for robust functionality.

---

## Background

This project began as a Jupyter Notebook focused on fetching AI agent papers from arXiv to explore advancements in the field. I later transformed it into a Streamlit app to make it more accessible and interactive, expanding its scope to fetch papers on any topic. It uses LlamaIndex and ChromaDB for indexing, and a LangGraph-powered agentic RAG pipeline for querying. I initially tried fetching papers from other sources like Scholarly, but rate-limiting issues put a pause on that—web-based alternatives are on my to-do list for future development.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- A Google API key for the Gemini LLM (set in a `.env` file)
- Optional: NVIDIA GPU for accelerated indexing (CUDA support required)

### Steps
1. **Clone the Repository**:
```bash
   git clone https://github.com/your-username/AIgent_Scholar.git
   cd AIgent_Scholar
```

2. **Set Up a Virtual Environment** *(optional but recommended)*:
```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
```
>[!note]
> You could use conda as well 🤷🏻

3. **Install Dependencies**:

```bash
    pip install -r requirements.txt
```

4. **Set Up Environment Variables**:
- Create a `.env` file in the project root.
- Add your Google API key

    ```text
    GOOGLE_API_KEY="your-google-api-key"
    ```
5. **Run the App:**

```bash
streamlit run app.py
```

> Open your browser to http://localhost:8501 to start exploring.
---
## Usage

1. **Fetch Papers**:
   - Select a topic from the dropdown (e.g., "AI Agents") or enter a custom topic.
   - Specify the maximum number of papers to fetch (1-10).
   - Click **Fetch Papers** to retrieve papers from arXiv.

2. **Index Papers**:
   - After fetching, click **Index Fetched Papers** to process and index the papers using LlamaIndex and ChromaDB.
   - Indexed papers will appear in the sidebar under "Indexed Papers."

3. **View Papers**:
   - Expand the "Indexed Papers" section in the sidebar.
   - Click on a paper title to open its PDF in a new tab.

4. **Query Papers**:
   - Use the chat interface at the bottom to ask questions about the indexed papers (e.g., "What is a MCP?").
   - The agentic RAG system will retrieve relevant info and provide an answer.

>[!tip]
> The agent might not recognize the topic straight on. So first mention the paper then go into its topics.

---

## Future Plans

- **Expand Paper Sources**: Explore web-based alternatives (e.g., browser/websearch) to fetch papers beyond arXiv, addressing rate-limiting issues with Scholarly.
- **Enhanced LaTeX Rendering**: Improve rendering of LaTeX in paper titles. 

>[!note]
> Well I simply didnot research much here..

- **Performance Optimization**: Further optimize indexing and retrieval for larger datasets.
- **UI Improvements**: Add interactive features like paper summaries and keyword highlighting.

---

## Contributing

I’d love for you to contribute! Here’s how:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit them (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a pull request with a detailed description of your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## Contact

Got questions, feedback, or ideas? Reach out via GitHub Issues or email me at [firojpaudel.professional@gmail.com](mailto:firojpaudel.professional@gmail.com). Let’s chat! 🤝
