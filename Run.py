import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
import json
import json
from langchain_ollama import OllamaLLM

# Define the file path
file_path = "metadata.json"

# Open and read the JSON file
with open(file_path, "r") as file:
    data = json.load(file)  # Load JSON content into a Python dictionary


dict_data={}
for i in data:
    pdf_url=i["pdf_url"]
    summary=i["summary"]
    dict_data[pdf_url]=summary
    

with open("dict_data.json", "w", encoding="utf-8") as f:
    json.dump(dict_data, f, indent=4)


    


pdf_urls = list(dict_data.keys())
summaries = list(dict_data.values())

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(summaries)
embeddings = np.array(embeddings).astype('float32')

query = "Improve the quality of code for a problem statement by iteratively going through research papers"

instructions = (
    "You are a highly knowledgeable assistant in natural language processing. "
    "Your task is to generate a synthetic summary that reformulates the following query with additional context on retrieving documents using Retrieval Augmented Generation (RAG). "
    "Ensure that the summary is concise, informative, and clear."
)
complete_prompt = instructions + "\nQuery: " + query

ollama_llm = OllamaLLM(model="deepseek-r1:7b")

synthetic_summary = ollama_llm(complete_prompt)
print("Synthetic Summary from deepseekr1 via Ollama with instructions:")
print(synthetic_summary)

query_embedding = model.encode(synthetic_summary).astype('float32')

similarities = np.dot(embeddings, query_embedding)


top_k = 10
top_indices = np.argsort(similarities)[::-1][:top_k]

print("\nTop similar summaries based on dot product similarity:")
for idx in top_indices:
    print(f"PDF URL: {pdf_urls[idx]}")
    print(f"Dot Product Similarity: {similarities[idx]:.4f}")
    # print(f"summary of extracted paper:{summaries[idx]}")
    print("-" * 40)
