from flask import jsonify, request
from app import app
from .document_loader import load_documents
from .embeddings import get_embeddings
from .vectorstore import get_vectorstore
from .chain import create_chain
import os 
import re
import json


# Initialize components
#pdf_files = [f"{app.config['PDF_DIR']}/{f}" for f in os.listdir(app.config['PDF_DIR']) if f.endswith(".pdf")]
#docs = load_documents(pdf_files)
embeddings = get_embeddings()
vectorstore = get_vectorstore(embeddings)
#vectorstore = init_vectorstore(docs, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
rag_chain = create_chain(retriever)


def clean_response(text):
    # First pass: Remove all backslashes and code blocks
    text = re.sub(r'\\"', '"', text)  
    text = re.sub(r'```json|```', '', text)  
    text = re.sub(r'\\n', ' ', text)  
    
    
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and 'answer' in parsed:
            return parsed['answer'].strip()
    except json.JSONDecodeError:
        pass
    
    
    text = re.sub(r'^.*?"answer"\s*:\s*"', '', text)  
    text = re.sub(r'",\s*"references".*$', '', text)  
    text = re.sub(r'{|}|\[|\]|"', '', text)  
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text).strip()  
    return text


@app.route('/query', methods=['POST'])
def process_query():
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        result = rag_chain.invoke({"input": query})
        
        
        cleaned_answer = clean_response(result["answer"])
        
        
        cleaned_refs = []
        for doc in result.get("context", []):
            cleaned_text = clean_response(doc.page_content)
            cleaned_refs.append({
                "exact_text": cleaned_text,
                "source": os.path.basename(doc.metadata.get("source")),  
                "page": doc.metadata.get("page"),
                "figure": None
            })
            
        return jsonify({
            "answer": cleaned_answer,
            "references": cleaned_refs
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


