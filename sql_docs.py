from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv
import json

# Set Google API key as environment variable
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

# Initialize retriever
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="sql-docs",
    embedding=embeddings
)

system_prompt = """
You are a knowledgeable SQL tutor from Chai and Code.
You answer users' questions based on the context retrieved from the Chai and Code SQL course material.

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
- If the user asks about SQL queries, refer to the queries section and provide the exact URL
- If the user asks about database design, refer to the database design section and provide the exact URL

Example of how to format source references:
Source: Introduction to SQL (https://docs.chaicode.com/youtube/chai-aur-sql/introduction/)
- Lines 15-20: Basic SQL concepts explanation
- Lines 45-50: SQL query structure example

IMPORTANT: Your response should be clean and professional. Do not include any metadata or debug information.
"""

def get_response(user_input):
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
    
    return response.content

if __name__ == "__main__":
    # Print header
    print("\n===== SQL Tutor Chatbot =====")
    print("Ask any questions about SQL (type 'exit' to quit)")
    print("===================================\n")

    # Interactive chatbot loop
    while True:
        user_input = input("\nYour question: ")
        
        if user_input.lower() == 'exit':
            print("\nThank you for using SQL Tutor. Goodbye!")
            break
        
        response = get_response(user_input)
        print("\nSQL Tutor:")
        print(response) 