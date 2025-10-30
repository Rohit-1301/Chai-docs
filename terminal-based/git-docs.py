from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

import os
import json
from dotenv import load_dotenv

# Set Google API key as environment variable
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

# Get URL from user input
urls = [
    "https://docs.chaicode.com/youtube/chai-aur-git/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-git/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-git/terminology/",
    "https://docs.chaicode.com/youtube/chai-aur-git/behind-the-scenes/",
    "https://docs.chaicode.com/youtube/chai-aur-git/branches/",
    "https://docs.chaicode.com/youtube/chai-aur-git/diff-stash-tags/",
    "https://docs.chaicode.com/youtube/chai-aur-git/managing-history/",
    "https://docs.chaicode.com/youtube/chai-aur-git/github/",
]

loader = WebBaseLoader(urls)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
    )

# Create vector store (uncommented to create the collection)
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://localhost:6333",
    collection_name="git-docs",
    embedding=embeddings
)

print("Docs length",len(docs))
print("Split docs length",len(split_docs))
print("Indexing done")

retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="git-docs",
    embedding=embeddings
)

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)

system_prompt = """
You are a knowledgeable Git tutor from Chai and Code.
You answer users' questions based on the context retrieved from the Chai and Code Git course material.

Your behavior guidelines:
1. Answer factually based only on the provided documents.
2. Include examples (like code snippets) if they are mentioned in the context.
3. At the end of each answer, provide detailed source references in this format:
   Source: [Page Title] (URL)
   - Line numbers or specific sections where the information was found
   - If multiple sources, list them all
4. If no relevant information is found, respond:
   ➔ "I'm sorry, but I couldn't find the answer based on the provided material."

Special rules based on your context:
- If the user asks about Git basics, refer to the introduction section and provide the exact URL
- If the user asks about basic Git commands, refer to the basic commands section and provide the exact URL
- If the user asks about branching, refer to the branching section and provide the exact URL
- If the user asks about remote repositories, refer to the remote repositories section and provide the exact URL

Example of how to format source references:
Source: Git Basics (https://docs.chaicode.com/youtube/chai-aur-git/introduction/)
- Lines 10-15: Git initialization explanation
- Lines 30-35: Basic Git workflow example

IMPORTANT: Your response should be clean and professional. Do not include any metadata or debug information.
"""

# Print header
print("\n===== Git Tutor =====")
print("Ask any questions about Git (type 'exit' to quit)")
print("===================================\n")

# Interactive chatbot loop
while True:
    user_input = input("\nYour question: ")
    
    if user_input.lower() == 'exit':
        print("\nThank you for using HTML Tutor. Goodbye!")
        break
    
    # Retrieve relevant documents
    docs = retriever.similarity_search(user_input)
    
    # Extract content
    context = ""
    
    for doc in docs:
        content = doc.page_content
        context += content + "\n\n"
    
    # Create messages for the LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nUser Question: {user_input}"}
    ]
    
    # Get response from LLM
    response = llm.invoke(messages)
    
    # Print response
    print("\nHTML Tutor:")
    print(response.content) 