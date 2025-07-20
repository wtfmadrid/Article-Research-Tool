import os
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

os.environ["OPENAI_API_KEY"] = "enter your api keys"

st.title("Articles Research Tool 🔍")
st.sidebar.title("Articles URLs \U0001F4C4")

if "url_count" not in st.session_state:
    st.session_state.url_count = 3  # start with 3 inputs
if "urls" not in st.session_state:
    st.session_state.urls = [""] * st.session_state.url_count

# Function to remove URL at index
def remove_url(index):
    st.session_state.urls.pop(index)
    st.session_state.url_count -= 1

# Display URL inputs with remove buttons
for i in range(st.session_state.url_count):
    cols = st.sidebar.columns([4, 1])
    st.session_state.urls[i] = cols[0].text_input(f"URL {i + 1}", value=st.session_state.urls[i], key=f"url_{i}")
    if st.session_state.url_count > 1:
        if cols[1].button("❌", key=f"remove_{i}"):
            remove_url(i)
            st.rerun()


# Add new URL field
if st.session_state.url_count < 10:
    if st.sidebar.button("➕ Add another URL"):
        st.session_state.url_count += 1
        st.session_state.urls.append("")

urls = st.session_state.urls

process_url_clicked =  st.sidebar.button("Process URLs")

main_placefolder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    #load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading...\u2705 \u2705")
    data = loader.load()
    #split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size = 1000
    )
    main_placefolder.text("Splitting Data...\u2705")
    docs = text_splitter.split_documents(data)
    #create embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Embedding Vector Started....\u2705 \u2705")
    time.sleep(2)
    #Create a PKL file
    if not os.path.exists("vector_index_store"):
        vectorstore_openai.save_local("vector_index_store")
    if os.path.exists("vector_index_store"):
        st.session_state.vectorIndex = FAISS.load_local(
            "vector_index_store", embeddings, allow_dangerous_deserialization=True
        )

query = main_placefolder.text_input("Question: ")

if query:
    if "vectorIndex" in st.session_state:
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorIndex.as_retriever()
        )
        result = chain({"question": query})
        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)

    




