# 📄 Chat with PDF (RAG Application)

An AI-powered application that allows users to upload a PDF and ask questions in natural language. The system retrieves relevant information from the document and generates accurate, context-aware answers using a Retrieval-Augmented Generation (RAG) pipeline.

---

## 🚀 Features

* 📄 Upload any PDF document
* 💬 Ask questions in natural language
* 🔍 Context-aware answers using semantic search
* 📊 Displays source chunks for transparency
* ⚡ Fast retrieval using vector database (FAISS)
* 🧠 Local LLM support (no API dependency)

---

## 🧠 How It Works

1. **PDF Upload**
   User uploads a document through the Streamlit interface.

2. **Text Splitting**
   The document is divided into smaller chunks using a text splitter.

3. **Embeddings Generation**
   Each chunk is converted into vector embeddings using a sentence transformer model.

4. **Vector Storage**
   Embeddings are stored in a FAISS vector database for fast similarity search.

5. **Query Processing**

   * User submits a question
   * System retrieves the most relevant chunks

6. **Answer Generation**
   A language model generates an answer based only on retrieved context.

---

## 🛠️ Tech Stack

* **Frontend/UI**: Streamlit
* **Framework**: LangChain
* **Vector Database**: FAISS
* **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
* **LLM**: FLAN-T5 (via HuggingFace Transformers)
* **Language**: Python

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/chat-with-pdf.git
cd chat-with-pdf
```

### 2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install streamlit langchain langchain-community langchain-core \
faiss-cpu sentence-transformers transformers pypdf
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 📸 Demo

Upload a PDF and ask questions like:

* "What is this document about?"
* "Summarize the key points"
* "Explain section 2"

---

## 💡 Use Cases

* 📚 Study and research assistance
* 🏢 Business document analysis
* ⚖️ Legal/financial document querying
* 📑 Knowledge base systems
* 🧑‍💻 Developer documentation assistants

---

## 📌 Future Improvements

* 💬 Chat history (multi-turn conversation)
* 📚 Multiple PDF support
* 🔎 Highlight answers in PDF
* ☁️ Cloud deployment
* 🧠 More powerful LLM integration

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
