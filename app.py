import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from flask import Flask, request, jsonify

# Load metadata
file_path = "metadata.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

dict_data = {i["pdf_url"]: i["summary"] for i in data}

# Save as a JSON file (optional, for debugging purposes)
with open("dict_data.json", "w", encoding="utf-8") as f:
    json.dump(dict_data, f, indent=4)

# Prepare embeddings
pdf_urls = list(dict_data.keys())
summaries = list(dict_data.values())
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = np.array(model.encode(summaries)).astype('float32')

# Build FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product for similarity search
index.add(embeddings)

# Initialize Ollama LLM
ollama_llm = OllamaLLM(model="deepseek-r1:7b")

# Flask app initialization
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Flask API is running! Use POST /search to query."

@app.route("/search", methods=["POST"])
def search():
    try:
        query = request.json.get("query")
        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Generate synthetic summary
        instructions = (
            "You are a highly knowledgeable assistant in natural language processing. "
            "Your task is to generate a synthetic summary that reformulates the following query with additional context on retrieving documents using Retrieval Augmented Generation (RAG). "
            "Ensure that the summary is concise, informative, and clear."
        )
        complete_prompt = instructions + "\nQuery: " + query
        synthetic_summary = ollama_llm(complete_prompt)

        # Encode query
        query_embedding = model.encode([synthetic_summary]).astype('float32')

        # Search FAISS index
        _, top_indices = index.search(query_embedding, 10)  # Get top 10 results
        results = [{"pdf_url": pdf_urls[idx], "similarity": float(np.dot(embeddings[idx], query_embedding.T))} for idx in top_indices[0]]

        return jsonify({"query": query, "synthetic_summary": synthetic_summary, "results": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
