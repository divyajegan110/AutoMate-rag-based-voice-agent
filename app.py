
import os
import subprocess
import pkg_resources
import streamlit as st
import speech_recognition as sr
import pyttsx3
import base64
import threading

from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)


google_api_key = os.getenv("google_api_key")  # Replace with your actual API key or set it in .env
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
   google_api_key=google_api_key
)
st.set_page_config(page_title="AutoMate", layout="wide")
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}
.answer-box {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    font-size: 18px;
    color: white;
    margin-top: 20px;
}
.status-box {
    font-size: 16px;
    font-weight: 500;
    color: #2563eb;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚗 AutoMate</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Drive with Confidence, Powered by Intelligence</div>', unsafe_allow_html=True)

recognizer = sr.Recognizer()

def text_to_speech(text):
    def speak():
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)
        tts_engine.setProperty('volume', 1)
        tts_engine.say(text)
        tts_engine.runAndWait()

    threading.Thread(target=speak).start()

@st.cache_resource
def load_vectorstore():
    persist_directory = "chroma_db"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )

    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    loader = UnstructuredPDFLoader("hyundai-warning-lights-indicators.pdf")
    car_docs = loader.load()

    splits = text_splitter.split_documents(car_docs)

    if not splits:
        raise ValueError("No text extracted from PDF. Check OCR setup.")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vectorstore.persist()
    return vectorstore


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()
prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are an assistant for question-answering tasks.
Use the following retrieved context to answer.
If unsure, say you don't know.
Don't mention the car brand anywhere.
Keep it within 3 sentences.

Question: {question}
Context: {context}
Answer:
"""
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=google_api_key
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown("###  Choose How You Want to Ask")
    input_option = st.radio(
        "",
        ("🎤 Speak", "⌨️ Type"),
        horizontal=True
    )

if "rag_answer" not in st.session_state:
    st.session_state["rag_answer"] = None

if "query" not in st.session_state:
    st.session_state["query"] = ""

def capture_audio():
    with sr.Microphone() as source:
        status = st.empty()
        status.markdown('<div class="status-box">🎤 Listening...</div>', unsafe_allow_html=True)

        audio = recognizer.listen(source)

        status.markdown('<div class="status-box">🧠 Processing speech...</div>', unsafe_allow_html=True)

        try:
            query = recognizer.recognize_google(audio)
            return query
        except:
            status.markdown('<div class="status-box">❌ Could not understand audio</div>', unsafe_allow_html=True)
            return ""


if input_option == "⌨️ Type":
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        user_input = st.text_input("💬 Ask your question")

    if user_input:
        st.session_state["query"] = user_input
        st.session_state["rag_answer"] = None

elif input_option == "🎤 Speak":
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        record_clicked = st.button("🎙️ Start Recording", use_container_width=True)

    if record_clicked:
        st.session_state["query"] = capture_audio()
        st.session_state["rag_answer"] = None
if st.session_state["query"]:
    st.markdown(
        f"""
        <div style="
            text-align:center;
            font-size:18px;
            margin-top:20px;
            color:#10b981;
        ">
        🗣 You said: <b>{st.session_state["query"]}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

if "my car is burning" in st.session_state["query"].lower():
    st.session_state["rag_answer"] = (
        "Exit the vehicle immediately and move to a safe distance. "
        "Call emergency services right away."
    )
    text_to_speech(st.session_state["rag_answer"])

elif st.session_state["query"] and not st.session_state["rag_answer"]:
    with st.spinner("🤖 Generating intelligent response..."):
        response = rag_chain.invoke(st.session_state["query"])
        st.session_state["rag_answer"] = response.content
        text_to_speech(st.session_state["rag_answer"])

if st.session_state["rag_answer"]:
    st.markdown('<div class="answer-box">' + st.session_state["rag_answer"] + '</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state["query"] = ""
        st.session_state["rag_answer"] = None
