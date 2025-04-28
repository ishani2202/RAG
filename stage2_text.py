import os
import json
import fitz  # PyMuPDF for PDF text extraction
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

# Directories
pdf_dir = "PDF_data"

# Load Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Index for vector storage
embedding_dim = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_dim)

# Load DeepSeek LLM
deepseek_llm = OllamaLLM(model="deepseek-r1:7b")  # DeepSeek for response generation

# ✅ Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text if text.strip() else None  # Return None if text is empty

# ✅ Function to recursively chunk text for embedding
def chunk_text(text, chunk_size=525, overlap=250):
    if not isinstance(text, str):  # Ensure text is valid
        return []
    
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# ✅ Function to store text in FAISS
def store_in_faiss(text_data):
    vectors = []
    text_chunks = []

    for doc_id, text in text_data.items():
        if isinstance(text, str) and text.strip():  # ✅ Only process valid text
            chunks = chunk_text(text)
            text_chunks.extend(chunks)
            vectors.extend(embedding_model.encode(chunks))

    if vectors:  # ✅ Avoid errors if no valid vectors exist
        vectors = np.array(vectors).astype("float32")
        index.add(vectors)  # Add to FAISS index

    return text_chunks

# ✅ Function to generate workflow using DeepSeek
def generate_workflow_with_deepseek(user_query, retrieved_text):
    prompt = (
        f"User Query: {user_query}\n\n"
        f"Retrieved Information from Research Papers:\n\n"
        f"{retrieved_text}\n\n"
        f"Use the user querry, understand the problem statement and then if needed use the retrived information to enhnace your answer. I need the output to be a structured flow to solve the problem"
    )

    return deepseek_llm.invoke(prompt)

# ✅ Take user input (query)
user_query = input("Enter your query: ")

# ✅ Extract text from PDFs
pdf_results = {}
aggregated_text = ""

for pdf_file in os.listdir(pdf_dir):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"Processing {pdf_file}...")

        # Extract text
        extracted_text = extract_text_from_pdf(pdf_path)
        if extracted_text:
            aggregated_text += extracted_text + "\n\n"
            pdf_results[pdf_file] = extracted_text

# ✅ Store extracted text in FAISS
print("PDF_done")
text_chunks = store_in_faiss(pdf_results)
print("text_chunked")

# ✅ Retrieve relevant text for user query using FAISS
query_vector = embedding_model.encode([user_query]).astype("float32")
D, I = index.search(query_vector, k=5)  # Retrieve top 5 relevant chunks
print("retrived simmilarity")

retrieved_text = "\n".join([text_chunks[idx] for idx in I[0] if idx < len(text_chunks)])

# ✅ Generate final workflow using DeepSeek
print("deepseek_started")
final_response = generate_workflow_with_deepseek(user_query, retrieved_text)

# ✅ Save results
with open("extracted_pdf_data.json", "w", encoding="utf-8") as f:
    json.dump(pdf_results, f, indent=4)

with open("generated_workflow.txt", "w", encoding="utf-8") as f:
    f.write(final_response)

print("\nGenerated Workflow:\n")
print(final_response)
print("\nExtraction complete! Data saved to extracted_pdf_data.json and generated_workflow.txt.")
