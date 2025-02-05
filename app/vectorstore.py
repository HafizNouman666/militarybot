from langchain_chroma import Chroma
from tqdm import tqdm
import os

def get_vectorstore(embeddings, docs=None):
    persist_dir = "./app/vector_db_store"
    
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    
    # Create new vector store only if doesn't exist
    if docs:
        chunk_size = 100
        for i in tqdm(range(0, len(docs), chunk_size)):
            chunk = docs[i:i + chunk_size]
            Chroma.from_documents(
                documents=chunk,
                embedding=embeddings,
                persist_directory=persist_dir
            )
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    
    raise ValueError("Vector store not found and no documents provided to create one")