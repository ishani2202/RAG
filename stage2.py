import os
import json
import fitz  # PyMuPDF for PDF text extraction
import numpy as np
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

# Directories
pdf_dir = "PDF_data"
image_dir = "PDF_IMAGES"  # Folder where user uploads images

# Load Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Index for vector storage
embedding_dim = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_dim)

# Load Ollama LLMs
llava_llm = OllamaLLM(model="llava:7b")  # LLaVA for image descriptions
deepseek_llm = OllamaLLM(model="deepseek-r1:7b")  # DeepSeek for response generation

# ✅ Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text if text.strip() else None  # Return None if text is empty

# ✅ Function to extract images from PDFs
def extract_images_from_pdf(pdf_path, output_dir="PDF_IMAGES"):
    os.makedirs(output_dir, exist_ok=True)
    images = []

    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_path = os.path.join(output_dir, f"{os.path.basename(pdf_path)}_page{i}_img{img_index}.png")

            with open(image_path, "wb") as f:
                f.write(image_bytes)
            images.append(image_path)

    return images

import base64
from io import BytesIO

def process_images_with_llava(image_paths):
    responses = {}

    for img_path in image_paths:
        try:
            # ✅ Open Image and Convert to Base64
            with open(img_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            # ✅ Use OllamaLLM with Correct Image Format
            prompt = (
                "Analyze this image carefully. Provide a detailed description explaining its structure, "
                "key insights, and any relevant observations that can be used for NLP research or automation."
            )

            # ✅ Send Base64 Image String to LLaVA
            description = llava_llm.invoke(prompt, images=[base64_image])
            responses[img_path] = description

        except Exception as e:
            responses[img_path] = f"Error processing image: {str(e)}"

    return responses

# ✅ Function to recursively chunk text for embedding
def chunk_text(text, chunk_size=256, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# ✅ Function to store text & image descriptions in FAISS
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
        f"Generate a structured response with step-by-step details."
    )

    return deepseek_llm.invoke(prompt)

# ✅ Take user input (query & optional image)
user_query = input("Enter your query: ")

image_paths = []
if os.path.exists(image_dir):
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith((".png", ".jpg", ".jpeg"))]

# ✅ Extract text & images from PDFs
pdf_results = {}
aggregated_text = ""
aggregated_images = {}

for pdf_file in os.listdir(pdf_dir):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"Processing {pdf_file}...")

        # Extract text
        extracted_text = extract_text_from_pdf(pdf_path)
        if extracted_text:
            aggregated_text += extracted_text + "\n\n"
            pdf_results[pdf_file] = extracted_text

        # Extract images
        extracted_images = extract_images_from_pdf(pdf_path)
        pdf_results[pdf_file + "_images"] = extracted_images

# ✅ Process images (from PDFs + user input) with LLaVA
# Ensure only image lists are added, ignore text
all_images = image_paths + [
    img for img_list in pdf_results.values() if isinstance(img_list, list) for img in img_list
]

image_descriptions = process_images_with_llava(all_images)

aggregated_images.update(image_descriptions)

# ✅ Store extracted text & images in FAISS
text_chunks = store_in_faiss(pdf_results)

# ✅ Retrieve relevant text for user query using FAISS
query_vector = embedding_model.encode([user_query]).astype("float32")
D, I = index.search(query_vector, k=5)  # Retrieve top 5 relevant chunks

retrieved_text = "\n".join([text_chunks[idx] for idx in I[0] if idx < len(text_chunks)])

# ✅ Generate final workflow using DeepSeek
final_response = generate_workflow_with_deepseek(user_query, retrieved_text)

# ✅ Save results
with open("extracted_pdf_data.json", "w", encoding="utf-8") as f:
    json.dump(pdf_results, f, indent=4)

with open("generated_workflow.txt", "w", encoding="utf-8") as f:
    f.write(final_response)

print("\nGenerated Workflow:\n")
print(final_response)
print("\nExtraction complete! Data saved to extracted_pdf_data.json and generated_workflow.txt.")
