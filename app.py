import streamlit as st
#from langchain_groq import ChatGroq
from langchain.chat_models import ChatOpenAI

#from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
#from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from dotenv import load_dotenv
import os

# ------------------ Load Env ------------------ #
load_dotenv()
os.environ["LANGSMITH_ENDPOINT"]  = os.getenv("LANGSMITH_ENDPOINT")
os.environ["OPENAI_API_KEY"]      = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_PROJECT"]   = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_API_KEY"]   = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
groq_api_key = os.getenv("groq_api_key")

# ------------------ Load Documents ------------------ #
def load_documents_from_folder(folder_path):
    docs = []
    for pdf_file in Path(folder_path).rglob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs.extend(loader.load())
    return docs

def create_vectorstore(docs):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# ------------------ Prompts ------------------ #
prompt_retriever = ChatPromptTemplate.from_messages([
    ("system", "Rephrase user query based on chat history which could be better understood by LLM."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

prompt_answer = ChatPromptTemplate.from_template("""
You are a medical assistant AI trained to help diagnose potential health conditions.

Use the information provided below to generate a possible diagnosis or medical advice.
Do NOT make assumptions. Use only the information in the context and input.

<context>
{context}
</context>                  
Question: {input}
Based on the above, what are the likely diagnoses or conditions?
Also suggest if the patient should consult a doctor urgently. If available in the context, suggest medicines else respond: I can't suggest medicines in this case.
Answer:
""")

# ------------------ Initialize Chatbot ------------------ #
@st.cache_resource
def init_chatbot():
    folder_path = "D:/PROJECTS/Medical_Report_Summerizer/Dataset"
    docs = load_documents_from_folder(folder_path)
    vectorstore = create_vectorstore(docs)
    retriever = vectorstore.as_retriever()

#    llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
    llm = ChatOpenAI( model="gpt-4o",temperature=0, max_tokens=256)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt_retriever)
    document_chain = create_stuff_documents_chain(llm, prompt_answer)
    retriever_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    return retriever_chain

retriever_chain = init_chatbot()

# ------------------ Setup Memory ------------------ #
session_id = "user-session-id"  # Can be dynamic (e.g., user IP or UUID)

chatbot_with_history = RunnableWithMessageHistory(
    retriever_chain,
    lambda session_id: InMemoryChatMessageHistory(),
    input_key="input",
    history_messages_key="chat_history"
)

# ------------------ Streamlit UI ------------------ #
st.set_page_config(page_title="Medical RAG Chatbot", layout="centered")
st.title("🩺 Medical Assistant Chatbot")
st.markdown("This bot is built on an open-source model and trained on medical reports. Accuracy may vary.")
st.markdown("**Give your age, gender, and symptoms.**")

query = st.text_input("Enter your question")

if query:
    response = chatbot_with_history.invoke(
        {"input": query},
        config={"session_id": session_id}
    )

    st.markdown(f"**🧑 You:** {query}")
    st.markdown(f"**🤖 Bot:** {response['answer']}")

# ------------------ Show Chat History (optional) ------------------ #
if st.toggle("Show chat history"):
    history = chatbot_with_history.get_message_history(session_id).messages
    for msg in history:
        role = "You" if msg.type == "human" else "Bot"
        st.markdown(f"**{role}:** {msg.content}")
