# 🌐 ChatGroq Website Q&A with LangChain & Streamlit

A simple Streamlit app that lets you **ask questions about any website** using the power of **LangChain**, **Groq's LLMs**, and **FAISS** vector search!

---

## 🚀 Features

- 🔗 Input any webpage URL
- 🤖 Uses `meta-llama/llama-4-scout-17b` via Groq for natural language responses
- 🔍 Uses FAISS for document similarity search
- 🧠 Embedding via HuggingFace's `bge-large-en-v1.5`
- 🧱 Chunking with LangChain's RecursiveCharacterTextSplitter
- 🔒 `.env` support for secure Groq API key usage

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Groq LLMs](https://console.groq.com/)
- [HuggingFace Embeddings](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [FAISS Vector Store](https://github.com/facebookresearch/faiss)

---

## 📦 Installation

1. **Clone the repo**

```bash
git clone https://github.com/your-username/chatgroq-webqa.git
cd chatgroq-webqa
```

2. **Create a virtual environment (optional)**

``` bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create .env file**
```bash
GROQ_API_KEY=your_groq_api_key_here
```

--- 

## 🧪 Run the App
streamlit run app.py

---

## 📜 License
MIT License – See LICENSE for full details.