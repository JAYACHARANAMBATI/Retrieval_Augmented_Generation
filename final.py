import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os


load_dotenv()


urls = [
    'https://www.dieboldnixdorf.com/en-us/',
    'https://www.dieboldnixdorf.com/en-us/banking/',
    'https://www.dieboldnixdorf.com/en-us/retail/',
    'https://www.dieboldnixdorf.com/en-us/support/',
    'https://www.dieboldnixdorf.com/en-us/about-us/',
    'https://www.dieboldnixdorf.com/en-us/careers/',
    'https://www.dieboldnixdorf.com/en-us/contact-us/',
    'https://www.dieboldnixdorf.com/en-us/about-us/global-locations/'
]


loader = UnstructuredURLLoader(urls=urls)
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


persist_directory = "./chroma_data"  


os.makedirs(persist_directory, exist_ok=True)


vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})


llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


st.title("Company Information Retrieval")
st.write("This app allows you to ask questions about the Diebold Nixdorf company.")


user_question = st.text_input("Ask a question:")


if user_question:
    with st.spinner("Retrieving answer..."):
        
        response = rag_chain.invoke({"input": user_question})
        st.write("Answer:", response["answer"])


st.sidebar.title("Information")
st.sidebar.write("This app uses information from the Diebold Nixdorf website.")
