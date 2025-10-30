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
if gemini_api_key:
    gemini_api_key = gemini_api_key.strip()  # Remove any whitespace
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    print(f"API Key loaded: {gemini_api_key[:10]}...")
else:
    print("❌ No API key found!")

# Get URL from user input
urls = [
    "https://docs.chaicode.com/youtube/chai-aur-c/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-c/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-c/hello-world/",
    "https://docs.chaicode.com/youtube/chai-aur-c/variables-and-constants/",
    "https://docs.chaicode.com/youtube/chai-aur-c/data-types/",
    "https://docs.chaicode.com/youtube/chai-aur-c/operators/",
    "https://docs.chaicode.com/youtube/chai-aur-c/control-flow/",
    "https://docs.chaicode.com/youtube/chai-aur-c/loops/",
    "https://docs.chaicode.com/youtube/chai-aur-c/functions/"
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
    collection_name="cpp-docs",
    embedding=embeddings
)

print("Docs length",len(docs))
print("Split docs length",len(split_docs))
print("Indexing done")

retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="cpp-docs",
    embedding=embeddings
)

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)

system_prompt = """
You are a knowledgeable C++ tutor.
You answer users' questions based on the context retrieved from the C++ documentation.

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
- If the user asks about C++ basics, refer to the introduction section and provide the exact URL
- If the user asks about variables and data types, refer to the data types section and provide the exact URL
- If the user asks about control structures, refer to the control flow section and provide the exact URL
- If the user asks about functions, refer to the functions section and provide the exact URL
- If the user asks about classes and objects, refer to the OOP section and provide the exact URL
- If the user asks about pointers and references, refer to the memory management section and provide the exact URL
- If the user asks about STL, refer to the standard library section and provide the exact URL
- If the user asks about templates, refer to the templates section and provide the exact URL

Example of how to format source references:
Source: C++ Classes (https://docs.chaicode.com/youtube/chai-aur-cpp/classes/)
- Lines 15-20: Class definition and member functions
- Lines 30-35: Constructor and destructor examples

IMPORTANT: Your response should be clean and professional. Do not include any metadata or debug information.
"""

# Print header
print("\n===== Django Tutor =====")
print("Ask any questions about CPP-Docs (type 'exit' to quit)")
print("===================================\n")

# Interactive chatbot loop
while True:
    user_input = input("\nYour question: ")
    
    if user_input.lower() == 'exit':
        print("\nThank you for using CPP-Docs. Goodbye!")
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
    print("\nDjango Tutor:")
    print(response.content) 