import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Paradigm.ci",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

load_dotenv()

# --- 2. THEME & VISUALS (Black & Beige) ---
st.markdown("""
    <style>
        /* Force Background to Deep Black */
        [data-testid="stAppViewContainer"] { background-color: #0a0a0a !important; }
        [data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; }
        
        /* Typography: Warm Beige */
        h1, h2, h3, p, div, span, label, li {
            color: #E3D5CA !important;
            font-family: 'Helvetica Neue', sans-serif;
        }
        
        /* Hide Default Elements */
        #MainMenu, footer { visibility: hidden; }
        
        /* Chat Bubbles: Minimalist & Sharp */
        .stChatMessage[data-testid="stChatMessage"] {
            background-color: transparent;
            border: 1px solid rgba(227, 213, 202, 0.2);
            border-radius: 0px;
        }
        
        /* Input Bar: Minimalist Line */
        .stTextInput input {
            color: #E3D5CA !important;
            background-color: transparent !important;
            border-bottom: 1px solid #E3D5CA !important;
            border-top: none !important;
            border-left: none !important;
            border-right: none !important;
        }

        /* --- ANIMATIONS --- */
        @keyframes gentleBounce {
            0%   { transform: translateY(-150%); opacity: 0; }
            50%  { transform: translateY(15%); opacity: 1; }
            100% { transform: translateY(0); opacity: 1; }
        }
        @keyframes flowFromLeft {
            0%   { transform: translateX(-50px); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }

        .logo-animate {
            display: inline-block;
            opacity: 0;
            color: #E3D5CA;
            animation: gentleBounce 1.5s cubic-bezier(0.25, 1, 0.5, 1) forwards;
        }
        .title-animate {
            display: inline-block;
            margin-left: 1rem;
            font-weight: 300;
            letter-spacing: 2px;
            opacity: 0;
            animation: flowFromLeft 2s cubic-bezier(0.22, 1, 0.36, 1) 0.5s forwards;
        }
        .subtitle-animate {
            font-size: 1rem;
            color: rgba(227, 213, 202, 0.5) !important;
            margin-left: 3.8rem;
            margin-top: -0.5rem;
            font-family: 'Courier New', monospace;
            opacity: 0;
            animation: flowFromLeft 2s cubic-bezier(0.22, 1, 0.36, 1) 1.0s forwards;
        }
    </style>
""", unsafe_allow_html=True)


# --- 3. INTELLIGENCE CORE (Cached) ---
@st.cache_resource
def load_system():
    # Load DNA
    try:
        with open("client_config.json", 'r') as f:
            dna = json.load(f)
    except:
        dna = {"dna_identity": {"ci_name": "Paradigm"}, "dna_synapse": {"model": "llama-3.3-70b-versatile"}}

    # Load Memory (RAG)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        memory = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except:
        memory = None

    # Load Brain (Synapse)
    try:
        synapse = ChatGroq(
            model_name=dna['dna_synapse']['model'],
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3 # Lower temp = Smarter, less random
        )
    except:
        st.error("❌ Critical Error: Brain Disconnected (API Key Missing)")
        st.stop()
    
    return synapse, memory, dna

synapse, memory, dna = load_system()


# --- 4. HEADER UI ---
st.markdown(
    f"""
    <div style="margin-bottom: 4rem; margin-top: 2rem;">
        <h1>
            <span class="logo-animate">⚡</span>
            <span class="title-animate">PARADIGM.CI</span>
        </h1>
        <div class="subtitle-animate">here to serve. not replace.</div>
    </div>
    """, 
    unsafe_allow_html=True
)


# --- 5. SMART INTERACTION LOOP ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Input command sequence..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 1. RETRIEVAL (Search Memory)
    context_text = "No internal database records found."
    source_label = ""
    
    if memory:
        docs = memory.similarity_search(prompt, k=2)
        if docs:
            context_text = "\n\n".join([d.page_content for d in docs])
            src = docs[0].metadata.get('source', 'Unknown').split('/')[-1]
            page = docs[0].metadata.get('page', 0) + 1
            source_label = f"\n\n--- \n*Ref: {src} (Pg {page})*"

    # 2. COGNITION (The Smarter Prompt)
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            
            # This is the "Prime Directive" that forces it to be smart and loyal
            system_prompt = f"""
            IDENTITY PROTOCOL:
            You are Paradigm, a proprietary CI (Created Intelligence) developed by the Paradigm.ci Dev Team.
            You are NOT Meta AI, OpenAI, or Google. You are a custom secure build.
            
            CORE OBJECTIVES:
            1. Analyze the user's request.
            2. Use the provided CONTEXT to answer if applicable.
            3. If the Context is empty, use your general knowledge but be concise.
            
            TONE:
            - Professional, Direct, and Intelligent.
            - Do not apologize excessively. Focus on solutions.
            - Format complex answers with bullet points.

            CONTEXT DATA:
            {context_text}
            
            USER QUERY: {prompt}
            """
            
            try:
                # 0.3 Temperature ensures focused, logical answers
                response = synapse.invoke(system_prompt)
                full_reply = response.content + source_label
                st.markdown(full_reply)
                st.session_state.messages.append({"role": "assistant", "content": full_reply})
            except Exception as e:
                st.error(f"System Anomaly: {e}")

