from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

import os
import json

# Set Google API key as environment variable
os.environ["GOOGLE_API_KEY"] = ""

# Get URL from user input
urls = [
    "https://docs.chaicode.com/youtube/chai-aur-sql/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/postgres/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/normalization/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/database-design-exercise/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/joins-and-keys/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/joins-exercise/"
]

loader = WebBaseLoader(urls)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
    )

# vector_store = QdrantVectorStore.from_documents(
#     documents=split_docs,
#     url="http://localhost:6333",
#     collection_name="sql-docs",
#     embedding=embeddings
# )

print("Docs length",len(docs))
print("Split docs length",len(split_docs))
print("Indexing done")

retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="sql-docs",
    embedding=embeddings
)

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

system_prompt = """
You are a knowledgeable SQL tutor.
You answer users' questions based on the context retrieved from the SQL documentation.

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
- If the user asks about SQL basics, refer to the introduction section and provide the exact URL
- If the user asks about SELECT statements, refer to the queries section and provide the exact URL
- If the user asks about JOIN operations, refer to the joins section and provide the exact URL
- If the user asks about data manipulation (INSERT, UPDATE, DELETE), refer to the DML section and provide the exact URL
- If the user asks about table creation and modification, refer to the DDL section and provide the exact URL
- If the user asks about constraints, refer to the constraints section and provide the exact URL
- If the user asks about functions and procedures, refer to the functions section and provide the exact URL
- If the user asks about transactions, refer to the transactions section and provide the exact URL

Example of how to format source references:
Source: SQL Joins (https://docs.chaicode.com/youtube/chai-aur-sql/joins/)
- Lines 10-15: INNER JOIN explanation
- Lines 25-30: LEFT JOIN example
- Lines 40-45: RIGHT JOIN use cases

IMPORTANT: Your response should be clean and professional. Do not include any metadata or debug information.
"""

# Print header
print("\n===== Django Tutor =====")
print("Ask any questions about SQL-Docs (type 'exit' to quit)")
print("===================================\n")

# Interactive chatbot loop
while True:
    user_input = input("\nYour question: ")
    
    if user_input.lower() == 'exit':
        print("\nThank you for using SQL-Docs. Goodbye!")
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