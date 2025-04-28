import os
from serpapi import GoogleSearch

SERP_API_KEY = "0db5b9a8e9236c1f36cdda43aa5953ce80b1e6af2129d1e1a7da8993c6a6b420"  # Replace with your SerpAPI Key

def get_google_results(query):
    search = GoogleSearch({
        "q": query,
        "num": 1,  # Fetch top 10 results
        "api_key": SERP_API_KEY
    })
    
    results = search.get_dict()
    top_links = [res['link'] for res in results.get("organic_results", [])]
    
    return top_links

# Example Usage
query = "latest AI research 2024"
top_articles = get_google_results(query)
print(top_articles)  # List of top 10 article URLs

import requests
from bs4 import BeautifulSoup

def scrape_article(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract the article text (modify selectors based on site structure)
        paragraphs = soup.find_all("p")  
        article_text = "\n".join([p.get_text() for p in paragraphs])
        
        return article_text.strip()
    
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return None

# Example: Scrape first 3 articles
scraped_articles = {url: scrape_article(url) for url in top_articles[:3]}
for url, text in scraped_articles.items():
    print(f"\n🔹 {url}\n{text[:500]}...")  # Show first 500 chars
    
    
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_PATH = "chroma"

# ✅ Initialize ChromaDB
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OllamaEmbeddings())

# ✅ Store scraped articles
for url, text in scraped_articles.items():
    if text:
        db.add_texts([text])

print("✅ Articles added to ChromaDB!")
