import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from dotenv import load_dotenv
import os

# Page config
st.set_page_config(
    page_title="Psychology Research Assistant",
    page_icon="🧠",
    layout="centered"
)

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Title
st.title("🧠 Psychology Research Assistant")
st.markdown("Ask questions about your research papers!")

# Initialize session state
if 'index' not in st.session_state:
    st.session_state.index = None
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
        st.session_state.index = None
    
    # Show index status
    if st.session_state.index:
        st.success("✅ Index loaded")
    else:
        st.info("📚 Index not loaded yet")

# Function to load or create index
@st.cache_resource(show_spinner=False)
def load_index():
    """Load index from storage or create new one"""
    PERSIST_DIR = "./storage"
    DATA_DIR = "./psych_pdfs"
    
    # Check if we have a persisted index
    if os.path.exists(PERSIST_DIR):
        with st.spinner("Loading existing index..."):
            # Load from disk
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            st.success("Loaded existing index!")
    else:
        with st.spinner("Creating new index from documents..."):
            # Check if data directory exists
            if not os.path.exists(DATA_DIR):
                st.error(f"Please create a '{DATA_DIR}' folder and add your PDFs!")
                return None
            
            # Check if there are files
            files = os.listdir(DATA_DIR)
            if not files:
                st.error(f"No files found in '{DATA_DIR}' folder!")
                return None
            
            # Create new index (your famous 5 lines!)
            documents = SimpleDirectoryReader(DATA_DIR).load_data()
            index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True
            )
            
            # Persist for next time
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            st.success(f"Created new index from {len(documents)} documents!")
    
    return index

# Load index if not already loaded
if st.session_state.index is None:
    st.session_state.index = load_index()

# Chat interface
if st.session_state.index:
    # Configure query engine with current settings
    from llama_index.llms.openai import OpenAI
    
    llm = OpenAI(model=model_name, temperature=temperature)
    query_engine = st.session_state.index.as_query_engine(
        llm=llm,
        similarity_top_k=top_k,
        streaming=True
    )
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_engine.query(prompt)
                
                # Stream the response
                response_placeholder = st.empty()
                full_response = ""
                
                if hasattr(response, 'response_gen'):
                    # Streaming response
                    for text in response.response_gen:
                        full_response += text
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                else:
                    # Non-streaming response
                    full_response = str(response)
                    response_placeholder.markdown(full_response)
                
                # Show source nodes
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    with st.expander("📚 Sources"):
                        for i, node in enumerate(response.source_nodes):
                            st.markdown(f"**Source {i+1}** (Score: {node.score:.3f})")
                            st.text(node.text[:500] + "...")
                            if node.metadata:
                                st.json(node.metadata)
                            st.divider()
        
        # Add assistant message to chat
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

else:
    # Instructions if no index
    st.info("""
    👋 Welcome! To get started:
    
    1. Create a `data` folder in your project directory
    2. Add your PDF files to the `data` folder
    3. Click the button below to reload
    """)
    
    if st.button("🔄 Reload and Check for Documents"):
        st.rerun()