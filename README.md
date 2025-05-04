 **Retrieval-Augmented Generation (RAG) System**

**Description**:
This project is a RAG (Retrieval-Augmented Generation) system focused on providing information about the Diebold Nixdorf company. It uses web scraping, vector embeddings, and a language model to answer user queries based on real company data.

---![WhatsApp Image 2025-05-02 at 17 26 27_ce1ce2a9](https://github.com/user-attachments/assets/09a8b4c7-9871-40da-ad95-2fa256cf980b)


**Main Components**:

1. **basic\_rag\_demo.ipynb**

   * A Jupyter notebook that crawls the Diebold Nixdorf website.
   * It collects and processes text content from multiple web pages.

2. **final.py**

   * A Streamlit web app that provides an interactive interface.
   * Allows users to enter questions and get answers based on the collected data.

---

**How It Works**:

* **Document Loading**:
  Content is scraped from predefined URLs using `UnstructuredURLLoader`.

* **Text Splitting**:
  Uses `RecursiveCharacterTextSplitter` to divide content into chunks for better processing.

* **Embeddings**:
  Embeddings are generated using Google’s Generative AI model.

* **Vector Storage**:
  Embeddings are stored in ChromaDB (located in the `chroma_data` folder).

* **Query Handling**:
  When a user asks a question, the system retrieves the most relevant chunks and uses Gemini Pro to generate a response.

---

**Technologies Used**:

* Python
* Jupyter Notebook
* Streamlit
* ChromaDB
* Google Generative AI (Embeddings + Gemini Pro)
* Langchain
* Transformers
* Unstructured.io

---

**Setup Instructions**:

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install required dependencies using `pip install -r requirements.txt`.
4. Run `basic_rag_demo.ipynb` to prepare the data.
5. Launch the app using `streamlit run final.py`.

---

**Directory Structure**:

```
TraeAI/
├── basic_rag_demo.ipynb       # Notebook for web scraping and data processing
├── final.py                   # Streamlit app for user queries
├── chroma_data/               # Folder for ChromaDB persistent storage
├── requirements.txt           # Dependencies
└── README.txt                 # Project documentation
```

