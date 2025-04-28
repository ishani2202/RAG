import argparse
import os
import sys
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function


CHROMA_PATH = "chroma"
OUTPUT_FILE = "output.txt"
SERP_API_KEY = "0db5b9a8e9236c1f36cdda43aa5953ce80b1e6af2129d1e1a7da8993c6a6b420"  # Replace with your actual SerpAPI Key

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # CLI to accept query text
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    user_query = args.query_text

    # ✅ Step 1: Use an LLM to refine the query into a better search phrase
    search_query = refine_query_with_llm(user_query)
    print(f"🔍 Improved Search Query: {search_query}")

    # ✅ Step 2: Fetch top Google articles
    top_articles = get_google_results(search_query)

    # ✅ Step 3: Scrape article content
    scraped_articles = {url: scrape_article(url) for url in top_articles[:1]}

    # ✅ Step 4: Store articles in ChromaDB
    store_in_chroma(scraped_articles)

    # ✅ Step 5: Generate response using RAG
    query_rag(user_query)


def refine_query_with_llm(user_query):
    """Use an LLM to refine the user's query into a more effective Google search."""
    llm = OllamaLLM(model="mistral")
    prompt = f"Rewrite the following question to make it a better search query for Google:\n\n{user_query}"
    refined_query = llm.invoke(prompt)
    return refined_query.strip()


def get_google_results(query):
    """Fetch top Google search results using SerpAPI."""
    search = GoogleSearch({
        "q": query,
        "num": 1,  # Fetch top 10 results
        "api_key": SERP_API_KEY
    })
    
    results = search.get_dict()
    top_links = [res['link'] for res in results.get("organic_results", [])]
    print(top_links)
    
    return top_links


def scrape_article(url):
    """Scrape and extract text from the given article URL."""
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")  
        article_text = "\n".join([p.get_text() for p in paragraphs])
        return article_text.strip()
    
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return None


def store_in_chroma(scraped_articles):
    """Store scraped web articles into ChromaDB for retrieval."""
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    for url, text in scraped_articles.items():
        if text:
            print(text)
            db.add_texts([text])
    print(text)
    print("\n✅ Articles stored in ChromaDB!")


def query_rag(query_text: str):
    """Retrieve relevant articles and generate AI response."""
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    # Format the retrieved context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Generate response
    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)

    # Write output to a text file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write(response_text.strip())

    print(f"✅ Output saved to {OUTPUT_FILE}")
    return response_text


if __name__ == "__main__":
    main()
