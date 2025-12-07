                          ┌───────────────────────────────────┐
                          │        RAG PROJECT ROOT           │
                          └───────────────────────────────────┘
                                        │
      ┌─────────────────────────────────┼──────────────────────────────────┐
      │                                 │                                  │
      ▼                                 ▼                                  ▼
┌───────────────┐             ┌──────────────────┐              ┌────────────────────┐
│  PDF PIPELINE │             │   WEB RAG PIPE   │              │ MULTIMODAL PIPELINE│
└───────────────┘             └──────────────────┘              └────────────────────┘
      │                                 │                                  │
      ▼                                 ▼                                  ▼

(1) Fetch PDFs + Metadata      (1) Search Google via SerpAPI      (1) Extract text from PDFs
    ↓                          (2) Scrape websites                (2) Extract images from PDFs
(2) Extract PDF text              ↓                               (3) Describe images (LLaVA)
(3) Build vector stores         (3) Store scraped text            (4) Build FAISS index
    Chroma + FAISS                 into ChromaDB                 (5) Retrieve + send to DeepSeek
(4) Expose via Flask API        (4) Query with RAG
(5) Query via Streamlit

