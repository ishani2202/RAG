import os
import json
import requests
import feedparser
from datetime import datetime

def download_pdf(pdf_url, filename):
    """Download a PDF from the given URL and save it locally."""
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Error downloading {pdf_url}: {e}")

def extract_pdf_url(entry):
    """
    Extract the PDF URL from an arXiv entry.
    It looks for a link with type 'application/pdf' or constructs one by replacing '/abs/' with '/pdf/'.
    """
    pdf_url = None
    if hasattr(entry, 'links'):
        for link in entry.links:
            if getattr(link, "rel", None) == "related" and getattr(link, "type", None) == "application/pdf":
                pdf_url = link.href
                break
    if not pdf_url and hasattr(entry, "id"):
        pdf_url = entry.id.replace('/abs/', '/pdf/')
    return pdf_url

def fetch_all_arxiv_papers(query, batch_size=10000):
    """
    Fetch all arXiv papers matching the query by paginating over results.
    
    Parameters:
    - query: The search query (e.g., "NAACL")
    - batch_size: Number of results to retrieve per API call.
    
    Returns a list of feed entries.
    """
    base_url = "http://export.arxiv.org/api/query"
    all_entries = []
    start = 0
    
    while True:
        params = {
            "search_query": query,
            "start": start,
            "max_results": batch_size
        }
        print(f"Fetching entries {start} to {start+batch_size} ...")
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            feed = feedparser.parse(response.text)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

        # Get the total number of results available (if provided)
        total_results = int(feed.feed.get("opensearch_totalresults", 0))
        entries = feed.entries
        if not entries:
            break

        all_entries.extend(entries)
        print(f"Retrieved {len(entries)} entries (Total so far: {len(all_entries)}/{total_results}).")
        start += batch_size

        # Break if we've retrieved all available results
        if start >= total_results:
            break

    return all_entries

def main():
    # Define your search query (e.g., for NAACL-related papers)
    query = "NAACL"
    print(f"Fetching papers for query: {query}")
    
    # Fetch all matching papers using pagination
    entries = fetch_all_arxiv_papers(query, batch_size=10000)
    
    if not entries:
        print("No data returned from the API. Exiting.")
        return

    # Create a directory for PDFs if it doesn't already exist
    os.makedirs("pdfs", exist_ok=True)
    
    metadata_list = []

    # Loop through each paper entry
    for entry in entries:
        print(f"Processing entry: {entry.get('id', 'No ID available')}")
        
        # Check if published date exists and is after 2020
        if hasattr(entry, "published"):
            try:
                pub_date = datetime.strptime(entry.published, '%Y-%m-%dT%H:%M:%SZ')
                if pub_date.year <= 2018:
                    print(f"Skipping paper published in {pub_date.year}.")
                    continue
            except Exception as e:
                print(f"Error parsing published date for {entry.get('id', 'Unknown ID')}: {e}")
                continue
        else:
            print("No published date available; skipping entry.")
            continue

        pdf_url = extract_pdf_url(entry)
        if not pdf_url:
            print("No PDF URL found for this entry.")
            continue

        # Generate a filename based on the paper's ID
        paper_id = entry.id.split("/")[-1] if hasattr(entry, "id") else "unknown"
        filename = os.path.join("pdfs", f"{paper_id}.pdf")
        print("Downloading PDF from:", pdf_url)
        download_pdf(pdf_url, filename)
        
        # Extract additional metadata
        title = entry.title if hasattr(entry, "title") else "No title available"
        summary = entry.summary if hasattr(entry, "summary") else "No summary available"
        published = entry.published if hasattr(entry, "published") else "No published date available"
        authors = [author.name for author in entry.authors] if hasattr(entry, "authors") else []

        # Build a metadata dictionary for this paper
        metadata = {
            "id": entry.id,
            "paper_id": paper_id,
            "title": title,
            "summary": summary,
            "published": published,
            "authors": authors,
            "pdf_url": pdf_url
        }
        metadata_list.append(metadata)
    
    # Save the metadata to a JSON file
    with open("metadata.json", "a", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=4)
    print("Metadata saved to metadata.json")

if __name__ == "__main__":
    main()
