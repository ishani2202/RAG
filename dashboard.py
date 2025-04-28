import streamlit as st
import requests

# Set up the dashboard
st.set_page_config(page_title="Research Paper Search", layout="wide")

st.title("📚 Research Paper Search with RAG")
st.write("Enter a query to find the most relevant research papers.")

# Input field
query = st.text_input("Enter your search query")

if st.button("🔍 Search"):
    if query:
        API_URL = "https://3336-2601-14d-5202-9930-84d7-2cb4-d7c-59fc.ngrok-free.app/search"  # Update this with your Ngrok/Render URL
        response = requests.post(API_URL, json={"query": query})

        if response.status_code == 200:
            results = response.json().get("results", [])
            
            if results:
                st.success("Top Research Papers Found:")
                for res in results:
                    st.markdown(f"📄 [{res['pdf_url']}]({res['pdf_url']}) - **Similarity:** {res['similarity']:.4f}")
            else:
                st.warning("No relevant papers found.")
        else:
            st.error(f"Error: {response.status_code}")
    else:
        st.warning("Please enter a query.")
