# app.py
"""
Streamlit UI for Psychology Research Assistant
All RAG logic is in walter_v2.py
"""

import streamlit as st
from walter_v2 import PsychologyRAG

# Page config
st.set_page_config(
    page_title="Psychology Research Assistant",
    page_icon="🧠",
    layout="centered"
)

# Title
st.title("🧠 Psychology Research Assistant")
st.markdown("Ask questions about your research papers!")

# Initialize session state
if 'rag_model' not in st.session_state:
    st.session_state.rag_model = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # Model selection
    model_name = st.selectbox(
        "Select Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
        index=0
    )
    
    # Number of retrieved chunks
    top_k = st.slider(
        "Number of source chunks to retrieve",
        min_value=1,
        max_value=10,
        value=3
    )
    
    # Temperature
    temperature = st.slider(
        "Temperature (0 = focused, 1 = creative)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1
    )
    
    # Index management
    st.header("Index Management")
    
    if st.button("🔄 Rebuild Index", help="Re-process all documents"):
        if st.session_state.rag_model:
            with st.spinner("Rebuilding index..."):
                st.session_state.rag_model.rebuild_index()
                st.success("Index rebuilt!")
                st.rerun()
    
    # Show index status
    if st.session_state.rag_model:
        stats = st.session_state.rag_model.get_index_stats()
        if stats["loaded"]:
            st.success("✅ Index loaded")
            with st.expander("Index Details"):
                st.json(stats)
        else:
            st.info("📚 Index not loaded yet")

# Load RAG model
@st.cache_resource(show_spinner=False)
def initialize_rag_model():
    """Initialize the RAG model (cached across sessions)"""
    try:
        with st.spinner("Initializing RAG model..."):
            rag = PsychologyRAG()
            rag.load_or_create_index()
            return rag
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

# Initialize model if not already done
if st.session_state.rag_model is None:
    st.session_state.rag_model = initialize_rag_model()

# Chat interface
if st.session_state.rag_model and st.session_state.rag_model.index:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}** (Score: {source['score']:.3f})")
                        st.text(source['text'][:500] + "...")
                        if source.get('metadata'):
                            st.json(source['metadata'])
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Query the model
                    result = st.session_state.rag_model.query(
                        question=prompt,
                        model_name=model_name,
                        temperature=temperature,
                        top_k=top_k,
                        streaming=True
                    )
                    
                    response = result["response"]
                    
                    # Handle streaming
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    if result["streaming"] and hasattr(response, 'response_gen'):
                        # Streaming response
                        for text in response.response_gen:
                            full_response += text
                            response_placeholder.markdown(full_response + "▌")
                        response_placeholder.markdown(full_response)
                    else:
                        # Non-streaming response
                        full_response = str(response)
                        response_placeholder.markdown(full_response)
                    
                    # Prepare sources for storage
                    sources = []
                    if hasattr(response, 'source_nodes') and response.source_nodes:
                        for node in response.source_nodes:
                            sources.append({
                                'score': node.score,
                                'text': node.text,
                                'metadata': node.metadata if hasattr(node, 'metadata') else None
                            })
                    
                    # Add assistant message to chat with sources
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"Error during query: {str(e)}")

else:
    # Instructions if no model/index
    st.info("""
    👋 Welcome! To get started:
    
    1. Make sure you have a `psych_pdfs` folder with PDF files
    2. Click the button below to initialize the system
    """)
    
    if st.button("🚀 Initialize System"):
        st.session_state.rag_model = None
        st.rerun()