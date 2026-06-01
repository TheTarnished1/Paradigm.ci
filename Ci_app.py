import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="Paradigm",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# --- 2. GEMINI-STYLE THEME ---
st.markdown("""
    <style>
        /* Root Colors */
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --bg-tertiary: #e8eaed;
            --text-primary: #202124;
            --text-secondary: #5f6368;
            --accent: #1f1f1f;
            --border: #dadce0;
        }
        
        /* Dark Mode Override */
        [data-theme="dark"] {
            --bg-primary: #121212;
            --bg-secondary: #1e1e1e;
            --bg-tertiary: #2d2d2d;
            --text-primary: #ffffff;
            --text-secondary: #b0b0b0;
            --accent: #aecbfa;
            --border: #3c4043;
        }

        /* Main Container */
        [data-testid="stAppViewContainer"] { 
            background-color: var(--bg-primary) !important; 
        }
        [data-testid="stHeader"] { 
            background-color: rgba(0,0,0,0) !important; 
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] { 
            background-color: var(--bg-secondary) !important; 
            border-right: 1px solid var(--border);
        }
        
        /* Typography */
        h1, h2, h3, p, div, span, label, li { 
            color: var(--text-primary) !important; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        /* Chat Messages */
        .stChatMessage { 
            background-color: transparent !important; 
            padding: 1rem 0 !important;
        }
        
        .stChatMessage[data-testid="stChatMessage"] { 
            border: none !important;
        }
        
        /* Input Field - Gemini Style */
        .stChatInputContainer {
            background-color: var(--bg-primary) !important;
            border-top: 1px solid var(--border);
            padding: 1rem !important;
        }
        
        .stChatInput input {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 24px !important;
            padding: 12px 20px !important;
            font-size: 16px !important;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        }
        
        .stChatInput input:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.1) !important;
        }
        
        /* Sidebar Buttons */
        div.stButton > button {
            background-color: transparent;
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 24px;
            width: 100%;
            padding: 10px 16px;
            font-size: 14px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            transition: all 0.2s ease;
        }
        
        div.stButton > button:hover {
            background-color: var(--bg-tertiary);
            border-color: var(--accent);
        }
        
        /* Hide Default Elements */
        #MainMenu, footer, .viewerBadge_container__1QSob { 
            visibility: hidden; 
        }
        
        /* Animations */
        @keyframes slideInBottom {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .message-animate {
            animation: slideInBottom 0.4s ease-out;
        }
        
        .fade-in {
            animation: fadeIn 0.6s ease-out;
        }
        
        /* Code Blocks */
        pre {
            background-color: var(--bg-tertiary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            padding: 16px !important;
        }
        
        code {
            color: var(--accent) !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD SYSTEM ---
@st.cache_resource
def load_system():
    try:
        with open("client_config.json", 'r') as f:
            dna = json.load(f)
    except:
        dna = {
            "dna_identity": {"ci_name": "Paradigm", "base_personality": "helpful and direct"},
            "dna_synapse": {"model": "llama-3.3-70b-versatile", "creativity_index": 0.5}
        }

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        memory = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except:
        memory = None

    try:
        synapse = ChatGroq(
            model_name=dna['dna_synapse']['model'],
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=dna['dna_synapse'].get('creativity_index', 0.5),
            streaming=True
        )
    except Exception as e:
        st.error(f"Failed to initialize Paradigm: {e}")
        st.stop()
    
    return synapse, memory, dna

synapse, memory, dna = load_system()

# --- 4. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversations" not in st.session_state:
    st.session_state.conversations = []

if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚡ Paradigm")
    st.markdown("---")
    
    # New Conversation Button
    if st.button("+ New Conversation", use_container_width=True, key="new_chat"):
        conv_id = datetime.now().isoformat()
        st.session_state.conversations.append({
            "id": conv_id,
            "title": "New Conversation",
            "messages": [],
            "created_at": conv_id
        })
        st.session_state.current_conversation_id = conv_id
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Conversation History
    st.markdown("**Recent**")
    if st.session_state.conversations:
        for conv in reversed(st.session_state.conversations[-5:]):
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                if st.button(conv["title"][:30], use_container_width=True, key=f"conv_{conv['id']}"):
                    st.session_state.current_conversation_id = conv['id']
                    st.session_state.messages = conv['messages']
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{conv['id']}", help="Delete"):
                    st.session_state.conversations.remove(conv)
                    st.session_state.messages = []
                    st.rerun()
    else:
        st.caption("No conversations yet")
    
    st.markdown("---")
    
    # System Info
    col1, col2 = st.columns(2)
    with col1:
        st.caption("**Model**")
        st.caption(dna['dna_synapse']['model'].split('/')[-1])
    with col2:
        st.caption("**Status**")
        st.caption("🟢 Online")

# --- 6. MAIN CONTENT AREA ---
st.markdown("<div class='fade-in'>", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([0.9, 0.1])
with col1:
    st.markdown("# ⚡ Paradigm")
    st.markdown(f"*{dna['dna_identity'].get('base_personality', 'here to serve. not replace.')}*")
with col2:
    if st.button("🔧", help="Settings", key="settings"):
        st.session_state.show_settings = True

st.markdown("---")

# Display Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="⚡" if message["role"] == "assistant" else "👤"):
        st.markdown(f"<div class='message-animate'>{message['content']}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# --- 7. CHAT INPUT & GENERATION ---
if prompt := st.chat_input("Ask Paradigm anything...", key="chat_input"):
    # Add user message
    st.chat_message("user", avatar="👤").markdown(f"<div class='message-animate'>{prompt}</div>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # RAG Search
    context_text = "No documents available."
    source_refs = []
    if memory:
        try:
            docs = memory.similarity_search(prompt, k=2)
            if docs:
                context_text = "\n\n".join([d.page_content for d in docs])
                for doc in docs:
                    src = doc.metadata.get('source', 'Unknown').split('/')[-1]
                    page = doc.metadata.get('page', 0) + 1
                    source_refs.append(f"{src} (Pg {page})")
        except:
            pass
    
    # Memory/History
    conversation_history = ""
    for msg in st.session_state.messages[-5:]:
        conversation_history += f"{msg['role'].upper()}: {msg['content']}\n"
    
    # System Prompt
    system_prompt = f"""
You are Paradigm, a proprietary AI assistant.

IDENTITY:
- Name: {dna['dna_identity']['ci_name']}
- Personality: {dna['dna_identity'].get('base_personality', 'helpful and direct')}
- Directive: {dna['dna_identity'].get('core_directive', 'Provide useful, accurate information')}

STRICT RULES:
1. Be concise, professional, and direct.
2. Use emojis sparingly but naturally.
3. Provide sources when available.
4. Give broad answers when asked, concise by default.
5. Be morally correct and helpful.

CONTEXT FROM DOCUMENTS:
{context_text}

CONVERSATION HISTORY:
{conversation_history}

USER QUERY: {prompt}

Respond naturally and helpfully:
"""
    
    # Stream Response
    with st.chat_message("assistant", avatar="⚡"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            stream = synapse.stream(system_prompt)
            for chunk in stream:
                full_response += chunk
                message_placeholder.markdown(f"<div class='message-animate'>{full_response}</div>", unsafe_allow_html=True)
            
            # Add source references if available
            if source_refs:
                st.markdown("---")
                st.caption(f"📚 Sources: {', '.join(source_refs)}")
                full_response += f"\n\n---\n📚 Sources: {', '.join(source_refs)}"
        
        except Exception as e:
            full_response = f"❌ Error: {str(e)}"
            message_placeholder.error(full_response)
    
    # Save to session
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Save to conversation history
    if st.session_state.current_conversation_id:
        for conv in st.session_state.conversations:
            if conv['id'] == st.session_state.current_conversation_id:
                conv['messages'] = st.session_state.messages
                # Update title based on first user message
                if len(conv['messages']) == 2:
                    conv['title'] = prompt[:40] + "..." if len(prompt) > 40 else prompt
                break
    
    st.rerun()
