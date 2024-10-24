import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load the GROQ and Google API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
st.set_page_config(page_title="Generali", page_icon="🪢")
st.title("Conversational PDF Document Q&A")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.2-90b-text-preview")

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    You are an integral part of the Saudi Tadawul Group, acting as a knowledgeable and professional representative. Please answer all questions with confidence, clarity, and a friendly yet formal tone, as would be expected from an official representative of the group. 

    When responding, ensure your tone conveys trust and reliability. Do not refer to yourself as a third party; speak as if you are directly part of the Saudi Tadawul Group. 

    If a question is unrelated to the context or outside the scope of your expertise, politely decline to answer, and redirect the user to appropriate information if possible. Avoid making assumptions or speculating. 

    Ensure all answers are ethical, accurate, and legal, and avoid any content that might be inappropriate or misleading.

    <conversation_history>
    {conversation_history}
    </conversation_history>
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def vector_embedding(file_paths):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load all PDFs and combine them into a single document set
        all_docs = []
        for file_path in file_paths:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(all_docs)  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings

# Predefined list of file paths to your PDFs
predefined_files = [
    "./Human Resoures Policy 2023.pdf"
    
]

# Load and embed the documents in the background
vector_embedding(predefined_files)
st.write("Vector Store DB is ready")

# Handling user question and response display
if "response_history" not in st.session_state:
    st.session_state.response_history = []

def get_response(question):
    temperature = 0.7
    conversation_history = ""
    for entry in st.session_state.response_history:
        conversation_history += f"User: {entry['question']}\nBot: {entry['response']}\n"
    
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({
        'input': question,
        'context': " ",  # Provide some initial context if needed
        'conversation_history': conversation_history,
        'temperature': temperature 
    })
    response_time = time.process_time() - start
    st.session_state.response_time = response_time
    return response

# Chat interface for user input and displaying responses
chat_container = st.container()
with chat_container:
    for i, entry in enumerate(st.session_state.response_history):
        st.chat_message("user").write(entry['question'])
        st.chat_message("assistant").write(entry['response'])
        with st.expander("Document Similarity Search", expanded=False):
            for doc in entry["context"]:
                st.write(doc.page_content)
                st.write("--------------------------------")
        feedback = st.empty()
        col1, col2 = feedback.columns([1, 0.1])
        if col1.button("👍", key=f"thumbs_up_{i}"):
            st.write("Thanks for your feedback!")
        elif col2.button("👎", key=f"thumbs_down_{i}"):
            st.write("Regenerating the answer...")
            response = get_response(entry['question'])
            entry['response'] = response['answer']
            entry['context'] = response['context']
            st.rerun()

if "vectors" in st.session_state:
    prompt1 = st.chat_input("Enter your question")
    if prompt1:
        response = get_response(prompt1)
        st.session_state.response_history.append({"question": prompt1, "response": response['answer'], "context": response["context"]})
        st.rerun()
