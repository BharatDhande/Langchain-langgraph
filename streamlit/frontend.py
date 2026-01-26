import streamlit as st
from backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# **************************************** utility functions *************************

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []
    # Force initialize chat_threads if it doesn't exist
    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = []
    # Add new thread
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    st.rerun()

def add_thread(thread_id):
    """Safely add a thread to the chat_threads list"""
    # Initialize if not exists
    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = []
    
    # Ensure it's a list
    if not isinstance(st.session_state['chat_threads'], list):
        st.session_state['chat_threads'] = []
    
    # Add thread if not already present
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        return state.values.get('messages', [])
    except Exception as e:
        st.error(f"Error loading conversation: {e}")
        return []

# **************************************** INITIALIZE SESSION STATE ******************

# This function runs only once to initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        # Generate thread ID first
        thread_id = generate_thread_id()
        
        # Initialize message history
        st.session_state['message_history'] = []
        
        # Initialize thread ID
        st.session_state['thread_id'] = thread_id
        
        # Initialize chat_threads from backend
        try:
            threads = retrieve_all_threads()
            # DEBUG
            print(f"INIT: Retrieved threads: {threads}, type: {type(threads)}")
            
            # Ensure we have a valid list
            if threads is None:
                st.session_state['chat_threads'] = []
            elif isinstance(threads, list):
                st.session_state['chat_threads'] = threads
            else:
                # Try to convert
                try:
                    st.session_state['chat_threads'] = list(threads)
                except:
                    st.session_state['chat_threads'] = []
        except Exception as e:
            print(f"INIT ERROR: {e}")
            st.session_state['chat_threads'] = []
        
        # Add current thread
        current_threads = st.session_state['chat_threads']
        if not isinstance(current_threads, list):
            st.session_state['chat_threads'] = []
            current_threads = []
        
        if thread_id not in current_threads:
            st.session_state['chat_threads'].append(thread_id)
        
        # Mark as initialized
        st.session_state['initialized'] = True
        
        print(f"INIT COMPLETE: thread_id={thread_id}, chat_threads={st.session_state['chat_threads']}")

# Initialize session state
initialize_session_state()

# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

# Display conversations if we have any
if st.session_state['chat_threads']:
    st.sidebar.header('My Conversations')
    
    for thread_id in st.session_state['chat_threads'][::-1]:
        # Create display name
        display_name = str(thread_id)
        if len(display_name) > 15:
            display_name = display_name[:12] + "..."
        
        # Use a unique key for each button
        if st.sidebar.button(f"📝 {display_name}", key=f"thread_{thread_id}"):
            st.session_state['thread_id'] = thread_id
            messages = load_conversation(thread_id)
            
            # Convert messages to chat format
            temp_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = 'user'
                else:
                    role = 'assistant'
                temp_messages.append({'role': role, 'content': msg.content})
            
            st.session_state['message_history'] = temp_messages
            st.rerun()
else:
    st.sidebar.info("No previous conversations")

# **************************************** Main UI ************************************

st.title("Chat with LangGraph")

# Show current thread ID
current_thread = st.session_state.get('thread_id', 'Unknown')
st.caption(f"Thread ID: {str(current_thread)[:20]}...")

# Display chat history
for message in st.session_state.get('message_history', []):
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Chat input
user_input = st.chat_input('Type your message here...')

if user_input:
    # Ensure message_history exists
    if 'message_history' not in st.session_state:
        st.session_state['message_history'] = []
    
    # Add user message to history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    
    # Display user message
    with st.chat_message('user'):
        st.markdown(user_input)
    
    # Prepare config for chatbot
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    
    # Display assistant response with streaming
    with st.chat_message("assistant"):
        def ai_only_stream():
            try:
                for message_chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages"
                ):
                    if isinstance(message_chunk, AIMessage):
                        yield message_chunk.content
            except Exception as e:
                yield f"Error getting response: {str(e)}"
        
        ai_message = st.write_stream(ai_only_stream())
    
    # Add assistant message to history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})