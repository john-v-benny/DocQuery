import streamlit as st
import os
import sys
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your existing query_data functions
# Make sure query_data.py is in the same directory or in Python path
try:
    from query_data import LocalEmbeddingFunction, CHROMA_PATH, PROMPT_TEMPLATE
except ImportError:
    st.error("Could not import from query_data.py. Make sure the file is in the same directory.")
    st.stop()

@st.cache_resource
def initialize_components():
    """Initialize the embedding model and database connection"""
    try:
        # Load the embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding_function = LocalEmbeddingFunction(model)
        
        # Connect to Chroma DB
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        llm_model = genai.GenerativeModel('gemini-2.0-flash')
        
        return db, llm_model, True
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None, False

def query_database(query_text, db, llm_model, k=3, relevance_threshold=0.3):
    """Query the database and generate response using your existing logic"""
    try:
        # Retrieve relevant chunks (same logic as your query_data.py)
        results = db.similarity_search_with_relevance_scores(query_text, k=k)
        
        if len(results) == 0:
            return "No results found in the database.", [], [], []
        
        # Show all results but warn about low relevance
        if results[0][1] < relevance_threshold:
            warning_msg = f"⚠️ Best match has low relevance score ({results[0][1]:.3f}). Results may not be accurate."
        else:
            warning_msg = None
        
        # Filter results by threshold but keep at least one result if any exist
        filtered_results = [(doc, score) for doc, score in results if score >= relevance_threshold]
        if not filtered_results and results:
            # If no results meet threshold, take the best one but add warning
            filtered_results = [results[0]]
        
        if not filtered_results:
            return "Unable to find matching results.", [], [], []
        
        # Prepare context (same as your original code)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        # Generate response using Gemini (same as your original code)
        response = llm_model.generate_content(prompt)
        response_text = response.text
        
        # Add warning to response if relevance is low
        if warning_msg:
            response_text = f"{warning_msg}\n\n{response_text}"
        
        # Extract sources and scores
        sources = [doc.metadata.get("source", "Unknown") for doc, _score in filtered_results]
        scores = [score for doc, score in filtered_results]
        contexts = [doc.page_content for doc, _score in filtered_results]
        
        return response_text, sources, scores, contexts
        
    except Exception as e:
        return f"Error processing query: {str(e)}", [], [], []

def main():
    st.set_page_config(
        page_title="RAG Document Query System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 RAG Document Query System")
    st.markdown("Query your document database using natural language")
    
    # Initialize components
    with st.spinner("Initializing system components..."):
        db, llm_model, success = initialize_components()
    
    if not success:
        st.error("Failed to initialize system. Please check your configuration.")
        st.stop()
    
    # Sidebar for settings
    st.sidebar.header("Query Settings")
    k = st.sidebar.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
    relevance_threshold = st.sidebar.slider("Relevance threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    
    # Add helpful info about thresholds
    st.sidebar.info("""
    **Relevance Threshold Guide:**
    - 0.0-0.3: Very loose matching
    - 0.3-0.5: Moderate matching (recommended)
    - 0.5-0.7: Strict matching
    - 0.7+: Very strict (may return no results)
    """)
    
    # Display current settings info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Configuration")
    st.sidebar.text(f"Database: {CHROMA_PATH}")
    st.sidebar.text(f"Model: all-MiniLM-L6-v2")
    st.sidebar.text(f"LLM: Gemini 2.0 Flash")
    
    # Add chunk size warning
    st.sidebar.markdown("---")
    st.sidebar.warning("""
    **Important:** Your chunks are very small (100 chars). 
    Consider increasing chunk_size in create_db.py to 500-1000 for better results.
    """)
    
    # Main query interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter your query")
        query_text = st.text_area(
            "What would you like to know?",
            placeholder="Enter your question here...",
            height=100
        )
        
        query_button = st.button("🔍 Query Database", type="primary")
    
    with col2:
        st.subheader("System Status")
        st.success("✅ Database connected")
        st.success("✅ Embedding model loaded")
        st.success("✅ LLM configured")
        
        if os.path.exists(CHROMA_PATH):
            st.info(f"📁 Database found")
        else:
            st.error("❌ Database not found")
    
    # Process query
    if query_button and query_text.strip():
        with st.spinner("Processing your query..."):
            try:
                response_text, sources, scores, contexts = query_database(
                    query_text, db, llm_model, k, relevance_threshold
                )
            except ValueError as e:
                st.error(f"Error unpacking query results: {str(e)}")
                st.stop()
        
        # Display results
        st.subheader("🤖 AI Response")
        if "Unable to find matching results" in response_text or "Error processing query" in response_text:
            st.warning(response_text)
        else:
            # Display the response in a nice container
            with st.container():
                st.markdown(f"**Answer:** {response_text}")
            
            # Display sources and relevance scores
            if sources and scores and len(sources) > 0:
                st.subheader("📋 Sources & Context")
                
                # Create tabs for better organization
                if len(sources) > 1:
                    tabs = st.tabs([f"Source {i+1}" for i in range(len(sources))])
                    for i, tab in enumerate(tabs):
                        with tab:
                            st.markdown(f"**Source:** {sources[i]}")
                            st.markdown(f"**Relevance Score:** {scores[i]:.3f}")
                            st.markdown("**Context:**")
                            if i < len(contexts):
                                st.text_area("", value=contexts[i], height=150, key=f"context_{i}")
                else:
                    st.markdown(f"**Source:** {sources[0]}")
                    st.markdown(f"**Relevance Score:** {scores[0]:.3f}")
                    st.markdown("**Context:**")
                    if len(contexts) > 0:
                        st.text_area("", value=contexts[0], height=150)
    
    elif query_button:
        st.warning("Please enter a query before searching.")
    
    # Example queries section
    st.markdown("---")
    st.subheader("💡 Try These Example Queries")
    example_queries = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
        "What are the important findings mentioned?",
        "Tell me about the methodology used"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(example_queries):
        with cols[i % 2]:
            st.button(f"📝 {example}", key=f"example_{i}")

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit | Powered by your existing RAG system*")

if __name__ == "__main__":
    main()