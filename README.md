# 🧠 DocMind – AI-Powered Document Q&A

> **RAG pipeline built using Endee Vector Database + OpenAI**  
Upload documents, ask questions, and get accurate AI-powered answers.

---

## 🎯 Project Overview

DocMind is an AI application that allows users to upload PDF or TXT documents and ask natural language questions.  
The system uses **vector search and Retrieval-Augmented Generation (RAG)** to provide accurate, context-based answers.

---

## 🚀 Features

- 📄 Upload PDF/TXT documents  
- 🔍 Semantic search using vector embeddings  
- 🤖 AI-generated answers using retrieved context  
- ⚡ Fast retrieval with Endee vector database  
- 📊 Displays relevant source chunks  

---

## 🏗️ System Workflow


Upload Document → Chunking → Embeddings → Store in Endee → User Query → Embedding → Similarity Search → Retrieve Data → Generate Answer


---

## 🧠 How It Works

1. **Document Upload** – User uploads PDF or TXT file  
2. **Chunking** – Text is split into smaller parts  
3. **Embedding** – Each chunk is converted into vector form  
4. **Storage** – Stored in Endee vector database  
5. **Query Processing** – User question is converted into embedding  
6. **Similarity Search** – Endee retrieves relevant chunks  
7. **Answer Generation** – LLM generates final answer  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Endee (Vector Database)  
- OpenAI API  
- PyPDF2  

---

## ⚙️ Setup Instructions

```bash
pip install -r requirements.txt
streamlit run app.py
📸 Demo

Example:

Input: What is Artificial Intelligence?
Output: AI is the simulation of human intelligence in machines...
📡 Use of Vector Database (Endee)
Stores embeddings efficiently
Performs similarity search using cosine distance
Retrieves top relevant results for queries
Enables fast and scalable AI applications
📁 Project Structure
docmind/
 ├── app.py
 ├── requirements.txt
 ├── README.md
🚀 Conclusion

This project demonstrates how vector databases and AI can be combined to build real-world intelligent applications using semantic search and RAG pipelines.

📄 Submission

GitHub Repository:
https://github.com/kaviha2006/endee

You can paste it directly and submit 🚀
