import streamlit as st
import PyPDF2
import io
from groq import Groq

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

st.title("📄 PDF Chatbot — Powered by LLM + RAG")
st.write("Upload a PDF and ask any question about it!")

# API Key - stored safely in environment variable
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# Sidebar
with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        st.session_state.pdf_text = text
        st.success(f"✅ PDF loaded! ({len(pdf_reader.pages)} pages)")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 🤖 About")
    st.markdown("This chatbot uses **Groq LLM** to answer questions from your PDF documents.")

# Chat interface
if st.session_state.pdf_text:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    if prompt := st.chat_input("Ask anything about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # Prepare context (first 3000 chars to avoid token limit)
        context = st.session_state.pdf_text[:3000]

        # Get response from Groq
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {
                                "role": "system",
                                "content": f"""You are a helpful assistant that answers questions based on the provided PDF document.
                                
PDF Content:
{context}

Answer questions only based on the PDF content. If the answer is not in the PDF, say "This information is not available in the PDF."
"""
                            },
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1000,
                        temperature=0.7,
                    )
                    answer = response.choices[0].message.content
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
else:
    st.info("👈 Please upload a PDF file from the sidebar to start chatting!")
    st.image("https://img.icons8.com/color/200/pdf.png", width=150)
