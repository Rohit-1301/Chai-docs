"""
Script to create and populate Qdrant collections for all documentation types.
Run this script once to set up all collections before using the Streamlit app.
"""

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

# Text splitter configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

def create_collection(collection_name, urls):
    """Create a Qdrant collection from web URLs"""
    print(f"Creating collection: {collection_name}")
    
    try:
        # Load documents from URLs
        loader = WebBaseLoader(urls)
        docs = loader.load()
        print(f"Loaded {len(docs)} documents")
        
        # Split documents
        splits = text_splitter.split_documents(docs)
        print(f"Created {len(splits)} text chunks")
        
        # Create vector store
        vectorstore = QdrantVectorStore.from_documents(
            splits,
            embeddings,
            url="http://localhost:6333",
            collection_name=collection_name
        )
        
        print(f"✅ Successfully created collection: {collection_name}")
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error creating collection {collection_name}: {str(e)}")
        return None

# Define collections and their URLs
collections_config = {
    "html-docs": [
        "https://developer.mozilla.org/en-US/docs/Web/HTML",
        "https://www.w3schools.com/html/",
        "https://html.spec.whatwg.org/",
    ],
    "git-docs": [
        "https://git-scm.com/doc",
        "https://docs.github.com/en/get-started",
        "https://www.atlassian.com/git/tutorials",
    ],
    "sql-docs": [
        "https://dev.mysql.com/doc/",
        "https://www.postgresql.org/docs/",
        "https://www.w3schools.com/sql/",
    ],
    "cpp-docs": [
        "https://en.cppreference.com/",
        "https://www.learncpp.com/",
        "https://isocpp.org/",
    ],
    "django-docs": [
        "https://docs.djangoproject.com/en/stable/",
        "https://www.djangoproject.com/start/",
        "https://tutorial.djangogirls.org/en/",
    ],
    "devops-docs": [
        "https://docs.docker.com/",
        "https://kubernetes.io/docs/",
        "https://docs.aws.amazon.com/",
    ]
}

def main():
    print("🚀 Starting Qdrant collections setup...")
    print("Make sure Docker and Qdrant are running on localhost:6333")
    
    # Test connection to Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333")
        collections = client.get_collections()
        print(f"✅ Connected to Qdrant. Existing collections: {len(collections.collections)}")
    except Exception as e:
        print(f"❌ Cannot connect to Qdrant: {str(e)}")
        print("Please make sure Qdrant is running with: docker-compose -f docker-compose.db.yml up -d")
        return
    
    # Create each collection
    for collection_name, urls in collections_config.items():
        print(f"\n📚 Processing {collection_name}...")
        create_collection(collection_name, urls)
    
    print("\n🎉 Setup complete! You can now run the Streamlit app.")

if __name__ == "__main__":
    main()