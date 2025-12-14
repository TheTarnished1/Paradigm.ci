import os
import json
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- SYSTEM INIT ---
load_dotenv()
if not os.environ.get("SLACK_BOT_TOKEN"):
    print("❌ ERROR: SLACK_BOT_TOKEN not found in .env file.")
    exit()

app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

class CIEngine:
    def __init__(self):
        self.dna = None
        self.memory = None
        self.synapse = None

    def boot(self):
        # 1. Load DNA
        if not os.path.exists("client_config.json"):
            print("❌ ERROR: client_config.json missing.")
            return

        with open("client_config.json", 'r') as f:
            self.dna = json.load(f)
        
        # 2. Ignite Synapse (The Brain)
        try:
            self.synapse = ChatGroq(
                temperature=self.dna['dna_synapse']['creativity_index'],
                model_name=self.dna['dna_synapse']['model'],
                api_key=os.getenv("GROQ_API_KEY")
            )
            print(f"🤖 CI '{self.dna['dna_identity']['ci_name']}' Brain Online.")
        except Exception as e:
            print(f"❌ Error connecting to Groq: {e}")
            return

        # 3. Try to Load Memory
        print("🧠 Checking for Memory Bank...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        try:
            self.memory = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            print("   ✅ Memory Loaded (RAG Mode Active).")
        except:
            self.memory = None
            print("   ⚠️ No Memory Found. Switching to Conversational Mode.")

    def process(self, query, channel_id):
        identity = self.dna['dna_identity'].copy()
        if channel_id in self.dna.get('channel_mutations', {}):
            identity.update(self.dna['channel_mutations'][channel_id])

        # MANUAL RETRIEVAL
        context_text = "No documents available."
        source_docs = []
        
        if self.memory:
            source_docs = self.memory.similarity_search(query, k=2)
            context_text = "\n\n".join([doc.page_content for doc in source_docs])
        
        system_prompt = f"""
        You are {identity['ci_name']} supporting {identity['business_name']}.
        ROLE: {identity.get('role', 'Assistant')}
        TONE: {identity['base_personality']}
        DIRECTIVE: {identity['core_directive']}
        
        CONTEXT FROM FILES:
        {context_text}
        
        User Question: {{question}}
        
        Answer:
        """
        
        prompt_template = PromptTemplate(template=system_prompt, input_variables=["question"])
        chain = prompt_template | self.synapse | StrOutputParser()
        answer = chain.invoke({"question": query})
        
        return {"result": answer, "source_documents": source_docs}

# Boot the Engine
engine = CIEngine()
engine.boot()

# --- SHARED RESPONSE FUNCTION ---
def generate_response(text, channel, ts, say):
    say(f"Thinking...", thread_ts=ts)
    try:
        result = engine.process(text, channel)
        answer = result['result']
        sources = result['source_documents']

        response = f"*{answer}*"
        if sources:
            response += "\n\n> 📚 *Source Verification:*"
            unique_refs = set()
            for doc in sources:
                src = doc.metadata.get('source', 'Unknown').split('/')[-1]
                page = doc.metadata.get('page', 0) + 1
                ref_id = f"{src}:{page}"
                if ref_id not in unique_refs:
                    response += f"\n> • `{src}` (Pg {page})"
                    unique_refs.add(ref_id)

        say(response, thread_ts=ts)
    except Exception as e:
        say(f"❌ System Error: {str(e)}", thread_ts=ts)
        print(f"Error: {e}")

# --- HANDLER 1: MENTIONS (Public Channels) ---
@app.event("app_mention")
def handle_mention(event, say):
    generate_response(event['text'], event['channel'], event.get('ts'), say)

# --- HANDLER 2: DIRECT MESSAGES (The Missing Piece) ---
@app.event("message")
def handle_message_events(event, say):
    # 1. Ignore messages from bots (prevents infinite loops)
    if event.get('bot_id'): 
        return
    
    # 2. If it's a Direct Message (IM), answer it!
    if event['channel_type'] == 'im':
        generate_response(event['text'], event['channel'], event.get('ts'), say)
    
    # 3. If it's a public channel message but NOT a mention, ignore it.
    # (This silences the "Unhandled request" warning)

if __name__ == "__main__":
    if os.environ.get("SLACK_APP_TOKEN"):
        SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
    else:
        print("❌ ERROR: SLACK_APP_TOKEN missing in .env")