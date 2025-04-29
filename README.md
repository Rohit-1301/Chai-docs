---

```markdown
# 🚀 Chai-Docs RAG App

A Retrieval-Augmented Generation (RAG) system built as part of the **Gen-AI Cohort** led by [Hitesh Choudhary](https://github.com/hiteshchoudhary) and [Piyush Garg](https://github.com/piyushgarg-dev). This project demonstrates how modern Gen-AI systems can leverage documentation and vector databases to produce accurate, context-aware answers.

---

## 📌 What is this?

This app answers user questions **based entirely on the Chai-Docs documentation** using the RAG pipeline. Ask a question and receive accurate, context-driven responses, all presented through a clean Streamlit interface.

---

## 🎯 Features

- 🔍 **Documentation Scraping** — Extracts and preprocesses Chai-Docs content
- 🧠 **Smart Chunking** — Uses LangChain to break down content meaningfully
- 🧭 **Semantic Search** — Embeds content and queries using vector similarity
- 💾 **Qdrant Integration** — Stores and retrieves document chunks via Dockerized vector database
- 🤖 **Gemini 1.0 Pro** — Generates responses based on matched content
- 💡 **Streamlit UI** — Clean, interactive frontend for user interaction

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **Qdrant (via Docker)**
- **Gemini 1.0 Pro API**

---

## 🔄 How It Works

1. **Scrape & Chunk Docs**: Chai-Docs are scraped and chunked using LangChain.
2. **Embed & Store**: Content chunks are embedded and stored in Qdrant.
3. **User Query**: A user inputs a question via the Streamlit UI.
4. **Search & Retrieve**: The app embeds the question, retrieves the most relevant chunks from Qdrant.
5. **Generate Answer**: Chosen chunks and the question are passed to Gemini 1.0 Pro to generate a response.
6. **Display**: The answer is displayed in the Streamlit interface.

---

## 🧪 Demo

🎥 **Check out the demo video** and project repository:  
👉 [GitHub Repository](https://github.com/Rohit-1301/Chai-docs)

---

## 🧰 Getting Started

### Prerequisites

- Python 3.8+
- Docker
- OpenAI-compatible Gemini 1.0 Pro API Key

### Installation

```bash
git clone https://github.com/Rohit-1301/Chai-docs.git
cd Chai-docs
pip install -r requirements.txt
```

### Run Qdrant via Docker

```bash
docker-compose -f docker-compose.db.yml up -d
```

### Start the App

```bash
streamlit run Chai_docs.py
```

---

## 🤝 Acknowledgments

Huge thanks to:

- [Hitesh Choudhary](https://github.com/hiteshchoudhary)
- [Piyush Garg](https://github.com/piyushgarg-dev)

For leading the **Gen-AI Cohort** and providing this practical, hands-on learning journey.

---

## 📬 Connect with Me

Made with 💙 by [Rohit](https://github.com/Rohit-1301)  
Feel free to ⭐ this repo and drop your feedback or issues.

---

## 🏷️ Tags

`#GenAI` `#LangChain` `#Streamlit` `#Qdrant` `#Gemini` `#RAG` `#AIProjects` `#ChaiDocs`
```

---

Let me know if you'd like this to include setup for `.env` files, or deploy instructions for platforms like Streamlit Cloud or Render.
