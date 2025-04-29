import streamlit as st
from html_docs import get_response as get_html_response
from git_docs import get_response as get_git_response
from sql_docs import get_response as get_sql_response
from cpp_docs import get_response as get_cpp_response
from django_docs import get_response as get_django_response
from devops_docs import get_response as get_devops_response

# Set page config
st.set_page_config(
    page_title="ChaiDocs - Documentation Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-size: 1.2em;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e6f3ff;
    }
    .chat-message.assistant {
        background-color: #f0f0f0;
    }
    .source-reference {
        background-color: #e8e8e8;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'current_topic' not in st.session_state:
    st.session_state.current_topic = None

# Sidebar for topic selection
with st.sidebar:
    st.title("📚 ChaiDocs")
    st.markdown("---")
    st.write("Select a documentation to chat with:")
    
    # Topic buttons
    if st.button("HTML Documentation", key="html"):
        st.session_state.current_topic = "HTML"
        st.session_state.chat_history["HTML"] = []
    
    if st.button("Git Documentation", key="git"):
        st.session_state.current_topic = "Git"
        st.session_state.chat_history["Git"] = []
    
    if st.button("SQL Documentation", key="sql"):
        st.session_state.current_topic = "SQL"
        st.session_state.chat_history["SQL"] = []
    
    if st.button("C++ Documentation", key="cpp"):
        st.session_state.current_topic = "C++"
        st.session_state.chat_history["C++"] = []
    
    if st.button("Django Documentation", key="django"):
        st.session_state.current_topic = "Django"
        st.session_state.chat_history["Django"] = []
    
    if st.button("DevOps Documentation", key="devops"):
        st.session_state.current_topic = "DevOps"
        st.session_state.chat_history["DevOps"] = []

# Main content area
st.title("ChaiDocs - Documentation Assistant")

# Show welcome message if no topic is selected
if not st.session_state.current_topic:
    st.markdown("""
    ### 👋 Welcome to ChaiDocs!
    
    Select a documentation from the sidebar to start chatting with our AI assistant.
    
    Each documentation contains:
    - Detailed explanations
    - Code examples
    - Source references
    - Practical insights
    
    Choose a topic to begin your learning journey!
    """)
else:
    # Display current topic
    st.header(f"Chatting with {st.session_state.current_topic} Documentation")
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        if st.session_state.current_topic in st.session_state.chat_history:
            for message in st.session_state.chat_history[st.session_state.current_topic]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if "sources" in message:
                        with st.expander("Source References"):
                            st.write(message["sources"])
    
    # User input
    user_input = st.chat_input("Ask your question...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history[st.session_state.current_topic].append({
            "role": "user",
            "content": user_input
        })
        
        # Get response based on current topic
        with st.spinner("Searching for answers..."):
            try:
                if st.session_state.current_topic == "HTML":
                    response = get_html_response(user_input)
                elif st.session_state.current_topic == "Git":
                    response = get_git_response(user_input)
                elif st.session_state.current_topic == "SQL":
                    response = get_sql_response(user_input)
                elif st.session_state.current_topic == "C++":
                    response = get_cpp_response(user_input)
                elif st.session_state.current_topic == "Django":
                    response = get_django_response(user_input)
                elif st.session_state.current_topic == "DevOps":
                    response = get_devops_response(user_input)
                
                # Add assistant response to chat history
                st.session_state.chat_history[st.session_state.current_topic].append({
                    "role": "assistant",
                    "content": response
                })
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.session_state.chat_history[st.session_state.current_topic].append({
                    "role": "assistant",
                    "content": "I'm sorry, I encountered an error while processing your request. Please try again."
                })
        
        # Rerun to update the chat display
        st.rerun()

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Chai and Code") 