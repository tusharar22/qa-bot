# 📄 RAG Chatbot (LangChain + IBM Watsonx)

This is an AI-powered Retrieval-Augmented Generation (RAG) chatbot built using **LangChain**, **IBM Watsonx**, and **Gradio**. It allows users to upload a PDF or TXT file and ask questions related to the document content. The chatbot uses a Large Language Model (LLM) and document embeddings to return context-aware answers.

---

## 🚀 Features

- 📚 Load and parse `.pdf` or `.txt` documents  
- 🔍 Split documents into retrievable chunks  
- 🧠 Embed documents using IBM Watsonx Embeddings  
- 🤖 Answer questions using the Mistral LLM hosted on IBM Watsonx  
- 🧩 Built with LangChain's RetrievalQA chain  
- 💬 User-friendly Gradio interface  

---

## 📦 Tech Stack

- **Python**
- **LangChain**
- **IBM Watsonx AI** (`mistralai/mixtral-8x7b-instruct-v01`)
- **Watsonx Embeddings**
- **Chroma Vector Store**
- **Gradio**

---

## 📁 Project Structure

```
qabot.py              # Main application code
README.md             # Project documentation (this file)
requirements.txt      # (Optional) Dependencies list
```

---

## 🛠️ How It Works

1. **Upload** a `.pdf` or `.txt` file  
2. **Enter a query** in the textbox  
3. The app:
   - Loads the document
   - Splits it into chunks
   - Embeds chunks using Watsonx
   - Stores them in a vector DB (Chroma)
   - Uses LangChain’s RetrievalQA with a Watsonx-hosted LLM to answer the query

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt  # If you have a requirements file
python qabot.py
```

App will launch on: [http://localhost:7860](http://localhost:7860)

---

## 🧪 Sample Use Cases

- Ask questions about a research paper
- Extract key information from legal documents
- Analyze business reports or product manuals

---

## 📬 Contact

**Tushar Arora**  
Email: tusharar22@gmail.com  
LinkedIn: [linkedin.com/in/tushar-arora-53b99b257](https://www.linkedin.com/in/tushar-arora-53b99b257)
