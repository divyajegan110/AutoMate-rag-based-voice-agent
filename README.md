🔥 One-Line Problem Statement

Modern vehicles display complex warning indicators that most drivers don’t fully understand, leading to confusion, delayed action, and potential safety risks.

💡 Proposed Solution

AutoMate is an AI-powered co-pilot that lets drivers speak or type dashboard warnings and instantly receive clear, concise, and safety-focused explanations using Retrieval-Augmented Generation (RAG).

📖 Project Description

AutoMate is an intelligent automotive assistant designed to simplify vehicle diagnostics for everyday drivers.

It uses Retrieval-Augmented Generation (RAG) to extract relevant information from official vehicle manuals and combines it with a large language model to deliver accurate, context-aware explanations.

Users can interact through voice or text, making the experience hands-free and intuitive — just like a real co-pilot.

The system persists embeddings using ChromaDB for optimized performance and avoids redundant document processing with smart caching.

Built for clarity. Designed for safety.

🧠 Key Features

🎤 Voice & Text Interaction

📄 PDF-based Knowledge Retrieval

⚡ Persistent Vector Database (ChromaDB)

🧠 Gemini LLM Integration

🚨 Emergency Response Handling

🔊 Text-to-Speech Output

💾 Cached Vector Store (Performance Optimized)

🛠 Tech Stack
Frontend

Streamlit

AI / LLM

Google Gemini (gemini-2.5-flash)

Gemini Embeddings (models/gemini-embedding-001)

RAG Pipeline

LangChain

ChromaDB (persistent vector storage)

UnstructuredPDFLoader

RecursiveCharacterTextSplitter

Voice Processing

SpeechRecognition

pyttsx3 (Text-to-Speech)

Storage

Local Chroma persistent directory

🏗 Architecture Overview

User Query (Voice/Text)
→ Speech Recognition (if voice)
→ Retriever (Chroma Vector DB)
→ Context Injection via Prompt Template
→ Gemini LLM
→ AI Response
→ Text-to-Speech Output