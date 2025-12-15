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

# --- 2. CINEMATIC ANIMATIONS (CSS) ---
st.markdown("""
    <style>
        /* Hide Default Streamlit Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* --- KEYFRAMES --- */
        
        @keyframes gentleBounce {
            0%   { transform: translateY(-150%); opacity: 0; }
            50%  { transform: translateY(15%); opacity: 1; }
            70%  { transform: translateY(-5%); }
            100% { transform: translateY(0); opacity: 1; }
        }

        @keyframes flowFromLeft {
            0%   { transform: translateX(-50px); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }

        @keyframes softRise {
            0%   { transform: translateY(100px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }

        /* --- CLASS STYLES --- */

        .logo-animate {
            display: inline-block;
            opacity: 0;
            animation: gentleBounce 1.5s cubic-bezier(0.25, 1, 0.5, 1) forwards;
        }

        .title-animate {
            display: inline-block;
            margin-left: 1rem;
            font-weight: 700;
            opacity: 0;
            animation: flowFromLeft 2s cubic-bezier(0.22, 1, 0.36, 1) 0.5s forwards;
        }

        /* NEW: Subtitle Animation (Delays until 1s) */
        .subtitle-animate {
            font-size: 1.2rem;
            color: rgba(255, 255, 255, 0.6); /* Slightly transparent white */
            margin-left: 3.8rem; /* Align under the text, not the logo */
            margin-top: -0.5rem;
            font-style: italic;
            opacity: 0;
            animation: flowFromLeft 2s cubic-bezier(0.22, 1, 0.36, 1) 1.0s forwards;
        }
        
        /* Input Bar Animation */
        section[data-testid="stBottom"] {
            opacity: 0;
            animation: softRise 1.8s cubic-bezier(0.22, 1, 0.36, 1) 1.5s forwards;
        }
        
        .stChatMessage {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
""", unsafe_allow_html=True)


# --- 3. CACHED RESOURCES ---
@st.cache_resource
def load_system():
    # Load DNA
    try:
        with open("client_config.json", 'r') as f:
            dna = json.load(f)
    except:
        dna = {"dna_identity": {"ci_name": "Paradigm"}, "dna_synapse": {"model": "llama-3.3-70b-versatile", "creativity_index": 0.5}}

    # Load Memory
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        memory = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except:
        memory = None

    # Load Brain
    try:
        synapse = ChatGroq(
            model_name=dna['dna_synapse']['model'],
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=dna['dna_synapse']['creativity_index']
        )
    except:
        synapse = None
        st.error("❌ Groq API Key missing.")
        st.stop()
    
    return synapse, memory, dna

synapse, memory, dna = load_system()


# --- 4. HEADER UI (With Subtitle) ---
st.markdown(
    f"""
    <div style="margin-bottom: 3rem;">
        <h1>
            <span class="logo-animate">⚡</span>
            <span class="title-animate">Paradigm.ci</span>
        </h1>
        <div class="subtitle-animate">Here to serve, not replace</div>
    </div>
    """, 
    unsafe_allow_html=True
)


# --- 5. CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Input command sequence..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    context_text = ""
    source_label = ""
    
    if memory:
        docs = memory.similarity_search(prompt, k=2)
        if docs:
            context_text = "\n\n".join([d.page_content for d in docs])
            src = docs[0].metadata.get('source', 'Unknown').split('/')[-1]
            page = docs[0].metadata.get('page', 0) + 1
            source_label = f"\n\n--- \n*Ref: {src} (Pg {page})*"

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            system_prompt = with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            # We add a "Identity Override" here
            system_prompt = f"""
            You are Paradigm, a private AI agent developed by the Paradigm Dev Team.
            You are NOT Meta AI, OpenAI, or Google. You are a custom proprietary build.
            
            If asked "Who created you?", answer: "I am Paradigm.proto, a custom CI engineered by the Paradigm Team."
            
            Use the context below to answer accurately.
            CONTEXT: {context_text}
            QUESTION: {prompt}
            """
            try:
                response = synapse.invoke(system_prompt)
                full_reply = response.content + source_label
                st.markdown(full_reply)
                st.session_state.messages.append({"role": "assistant", "content": full_reply})
            except Exception as e:
                st.error(f"Error: {e}")
            """
            try:
                response = synapse.invoke(system_prompt)
                full_reply = response.content + source_label
                st.markdown(full_reply)
                st.session_state.messages.append({"role": "assistant", "content": full_reply})
            except Exception as e:

                st.error(f"Error: {e}")



