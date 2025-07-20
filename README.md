# 🧠 Article Research Tool 🔍

## Overview

The Article Research Tool is a powerful Streamlit-based application that allows users to input multiple article URLs, convert the content into embeddings using OpenAI's models, and query the information using natural language. It's particularly useful for research, summarization, and quick information retrieval across multiple sources.

💡 Note: ChatGPT and most LLMs currently cannot browse or process URLs directly. This tool bridges that gap by scraping, embedding, and enabling natural language queries over web content — all in one streamlined workflow.

## 🔧 Technologies Used

Python -	Core programming language

Streamlit -	For creating an interactive web-based UI

LangChain -	Used to chain LLMs with tools like vector stores

OpenAI API -	For generating embeddings and running the language model

FAISS -	Efficient vector store for storing and querying document embeddings

Unstructured -	For loading and parsing web content from URLs

## 📂 Project Structure

urlretrieval_project.py: Main application file built with Streamlit. Contains the frontend and backend logic.

ai_project.ipynb & langchain_ai_project.ipynb: Educational notebooks that demonstrate:

- How content from URLs is scraped.

- How it's split into chunks using RecursiveCharacterTextSplitter.

- How embeddings are generated using OpenAIEmbeddings.

- How those embeddings are stored and retrieved using FAISS.

These notebooks are perfect if you want to understand the underlying mechanisms before diving into the main application.

## ✨ Key Features

- Add up to 10 URLs dynamically through the sidebar.

- Store and reuse document embeddings via FAISS.

- Ask natural language questions about the combined content of the URLs.

- Clean, responsive UI powered by Streamlit.

- Works around ChatGPT’s limitation of not being able to access or summarize live web pages.

## 🚀 Getting Started

Load the urlretrieval_project.py on any IDE.

To run the tool locally:

streamlit run urlretrieval_project.py
