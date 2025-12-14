import os
import json
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class CISetup:
    def __init__(self, dna_path="client_config.json"):
        self.dna_path = dna_path
        with open(dna_path, 'r') as f:
            self.dna = json.load(f)

    def sequence_dna(self):
        """Step 1: Scans folder and updates JSON."""
        print("🧬 SEQUENCING DNA...")
        target_dir = self.dna['dna_memory']['source_directory']
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print(f"   Created {target_dir}")
        
        pdf_files = [os.path.basename(f) for f in glob.glob(os.path.join(target_dir, "*.pdf"))]
        self.dna['dna_memory']['active_documents'] = pdf_files
        
        with open(self.dna_path, 'w') as f:
            json.dump(self.dna, f, indent=4)
        print(f"   Updated DNA with {len(pdf_files)} documents.")

    def encode_memory(self):
        """Step 2: Reads PDFs and saves FAISS index."""
        print("🧠 ENCODING MEMORY...")
        doc_names = self.dna['dna_memory']['active_documents']
        source_dir = self.dna['dna_memory']['source_directory']
        raw_docs = []

        for doc in doc_names:
            path = os.path.join(source_dir, doc)
            if os.path.exists(path):
                print(f"   Reading: {doc}")
                loader = PyPDFLoader(path)
                raw_docs.extend(loader.load())

        if not raw_docs:
            print("   ❌ No documents to process.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(raw_docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local("faiss_index")
        print("   ✅ Memory Bank (faiss_index) saved successfully.")

if __name__ == "__main__":
    setup = CISetup()
    setup.sequence_dna()
    setup.encode_memory()