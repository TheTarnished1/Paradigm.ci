import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Paradigm.CI",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

load_dotenv()

# --- 2. THEME (Black & Beige) ---
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] { background-color: #0a0a0a !important; }
        [data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; }
        h1, h2, h3, p, div, span, label, li { color: #E3D5CA !important; font-family: 'Helvetica Neue', sans-serif; }
        #MainMenu, footer { visibility: hidden; }
        .stChatMessage[data-testid="stChatMessage"] { background-color: transparent; border: 1px solid rgba(227, 213, 202, 0.2); border-radius: 0px; }
        .stTextInput input { color: #E3D5CA !important; background-color: transparent !important; border-bottom: 1px solid #E3D5CA !important; border-top: none !important; border-left: none !important; border-right: none !important; }
        
        @keyframes gentleBounce { 0% { transform: translateY(-150%); opacity: 0; } 50% { transform: translateY(15%); opacity: 1; } 100% { transform: translateY(0); opacity: 1; } }
        @keyframes flowFromLeft { 0% { transform: translateX(-50px); opacity: 0; } 100% { transform: translateX(0); opacity: 1; } }
        .logo-animate { display: inline-block; opacity: 0; color: #E3D5CA; animation: gentleBounce 1.5s cubic-bezier(0.25, 1, 0.5, 1) forwards; }
        .title-animate { display: inline-block; margin-left: 1rem; font-weight: 300; letter-spacing: 2px; opacity: 0; animation: flowFromLeft 2s cubic-bezier(0.22, 1, 0.36, 1) 0.5s forwards; }
        .subtitle-animate { font-size: 1rem; color: rgba(227, 213, 202, 0.5) !important; margin-left: 3.8rem; margin-top: -0.5rem; font-family: 'Courier New', monospace; opacity: 0; animation: flowFromLeft 2s cubic-bezier(0.22, 1, 0.36, 1) 1.0s forwards; }
    </style>
""", unsafe_allow_html=True)

# --- 3. INTELLIGENCE CORE ---
@st.cache_resource
def load_system():
    try:
        with open("client_config.json", 'r') as f:
            dna = json.load(f)
    except:
        # UPDATED DEFAULT IDENTITY
        dna = {"dna_identity": {"ci_name": "Paradigm"}, "dna_synapse": {"model": "llama-3.3-70b-versatile"}}

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        memory = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except:
        memory = None

    try:
        synapse = ChatGroq(
            model_name=dna['dna_synapse']['model'],
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1 
        )
    except:
        st.stop()
    
    return synapse, memory, dna

synapse, memory, dna = load_system()

# --- 4. UI ---
st.markdown(
    f"""
    <div style="margin-bottom: 4rem; margin-top: 2rem;">
        <h1><span class="logo-animate">⚡</span><span class="title-animate">PARADIGM.CI</span></h1>
        <div class="subtitle-animate">here to serve. not replace.</div>
    </div>
    """, unsafe_allow_html=True
)

# --- 5. LOGIC LOOP ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Input command sequence..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 1. DOCUMENT SEARCH (RAG)
    context_text = "No internal documents found."
    source_label = ""
    if memory:
        docs = memory.similarity_search(prompt, k=2)
        if docs:
            context_text = "\n\n".join([d.page_content for d in docs])
            src = docs[0].metadata.get('source', 'Unknown').split('/')[-1]
            page = docs[0].metadata.get('page', 0) + 1
            source_label = f"\n\n--- \n*Ref: {src} (Pg {page})*"

    # 2. CONVERSATION MEMORY BUILDER
    conversation_history = ""
    for msg in st.session_state.messages[-4:]:
        conversation_history += f"{msg['role'].upper()}: {msg['content']}\n"

    # 3. GENERATION
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            
            # IDENTITY & INSTRUCTIONS (UPDATED)
            system_prompt = f"""
            You are Paradigm.
            
            STRICT RULES:
            1. You are NOT Meta AI, OpenAI, or Google. If asked, deny it.
            2. You are "Paradigm", a proprietary CI (Created Intelligence).
            3. You are part of the Paradigm.CI system.
            4. Be concise and professional.
            
            KNOWLEDGE BASE (DOCS):
            {context_text}
            
            RECENT CONVERSATION (MEMORY):
            {conversation_history}
            
            USER'S NEW QUERY:
            {prompt}
            
            YOUR RESPONSE (Must follow STRICT RULES):
            """
            
            try:
                response = synapse.invoke(system_prompt)
                full_reply = response.content + source_label
                st.markdown(full_reply)
                st.session_state.messages.append({"role": "assistant", "content": full_reply})
            except Exception as e:
                st.error(f"Error: {e}")
