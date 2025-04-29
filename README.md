🚀 Chai-Docs RAG App
This project is part of the Gen-AI Cohort, led by Hitesh Choudhary and Piyush Garg. It shows how you can build a smart Q&A system using AI, vector databases, and real documentation.

📌 What’s This About?
The Chai-Docs RAG App is a question-answering tool that uses AI to give answers based only on the Chai-Docs documentation. It uses a Retrieval-Augmented Generation (RAG) setup behind the scenes and wraps it all in a simple, clean Streamlit interface.

🎯 What It Can Do
🔍 Scrape Docs – Pulls in the content from Chai-Docs

🧠 Chunk Smartly – Breaks down docs using LangChain so they're easy to search

🧭 Semantic Search – Finds the most relevant info using vector similarity

💾 Qdrant Integration – Uses Qdrant (via Docker) to store and retrieve doc chunks

🤖 Gemini 1.5 Pro – Generates answers from the matched docs

💡 Streamlit UI – Gives users a clean interface to ask questions and get results

🛠️ Tech Behind the Scenes
Python

Streamlit

LangChain

Qdrant (Docker)

Gemini 1.5 Pro API

🔄 How It Works
Scrape & Chunk – The app scrapes Chai-Docs and splits the text into smart chunks.

Embed & Store – It converts those chunks into embeddings and stores them in Qdrant.

Ask a Question – You type your question into the Streamlit app.

Search & Match – The app finds the most relevant chunks based on your question.

Generate an Answer – Gemini 1.0 Pro uses the matched chunks to create a response.

See the Result – The final answer appears right in the UI.

🧪 See It in Action
🎥 Check out the demo and code here:
👉 GitHub Repository

🧰 Getting Started
What You’ll Need
Python 3.8+

Docker

Gemini 1.5 Pro API key (OpenAI-compatible)

git clone https://github.com/Rohit-1301/Chai-docs.git
cd Chai-docs
pip install -r requirements.txt

Run Qdrant via Docker
docker-compose -f docker-compose.db.yml up -d

Start the App
streamlit run Chai_docs.py

