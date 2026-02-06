import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from chromadb.config import Settings

# Page configuration
st.set_page_config(page_title="AI Research Assistant", page_icon="🤖")

st.title("🤖 AI Research Assistant")
st.write("Ask questions about the Llama 2 research paper")


@st.cache_resource(show_spinner="Loading RAG system...")
def initialize_rag_system():
    """Initialize the RAG system with caching."""
    loader = PyPDFLoader("./source_data/llama2.pdf")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Use in-memory ChromaDB to avoid persistence issues
    vector_store = Chroma.from_documents(
        docs, 
        embeddings,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=False)
    )
    
    llm = Ollama(model="aiResearcher:latest")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(vector_store.as_retriever(), combine_docs_chain)
    
    return rag_chain


# Initialize RAG system
rag_chain = initialize_rag_system()

# Use a form to handle input and clear after submission
with st.form(key="question_form", clear_on_submit=True):
    user_question = st.text_input("Enter your question:")
    submit_button = st.form_submit_button("Ask")

# Process question
if submit_button and user_question:
    with st.spinner("Generating response..."):
        response = rag_chain.invoke({"input": user_question})
        answer = response.get('answer', 'No answer generated.')
    
    st.markdown("### Answer")
    st.write(answer)
    
    # Show sources
    with st.expander("View Source Documents"):
        for i, doc in enumerate(response.get('context', [])[:3]):
            st.write(f"**Source {i+1}:**")
            st.write(doc.page_content[:500] + "...")
            st.divider()
