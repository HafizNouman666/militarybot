from flask import jsonify, request
from app import app
from .document_loader import load_documents
from .embeddings import get_embeddings
from .vectorstore import get_vectorstore
from .chain import create_chain
from langchain.schema import HumanMessage, AIMessage
import os 
import re
import json


embeddings = get_embeddings()
vectorstore = get_vectorstore(embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
rag_chain = create_chain(retriever)


def load_chat_history(chat_id):
    """Load the conversation history for a given chat_id from a JSON file."""
    chat_dir = "chat_history"
    if not os.path.exists(chat_dir):
        os.makedirs(chat_dir)
    file_path = os.path.join(chat_dir, f"{chat_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
           
            return json.load(f)
    else:
        return []


def save_chat_history(chat_id, history):
    """Save the conversation history (a list of message dicts) to a JSON file."""
    chat_dir = "chat_history"
    if not os.path.exists(chat_dir):
        os.makedirs(chat_dir)
    file_path = os.path.join(chat_dir, f"{chat_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def clean_response(text):
   
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
    chat_id = data.get("id", None)
    query = data.get('query', '')

    if not chat_id:
        return jsonify({"error": "No chat id provided"}), 400
    if not query:
        return jsonify({"error": "No query provided"}), 400

    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        stored_history = load_chat_history(chat_id)  
        messages = []
        for msg in stored_history:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        rag_chain.memory.chat_memory.messages = messages

        # (Optionally) Add the new human query to the chain's memory.
        human_msg = HumanMessage(content=query)
        rag_chain.memory.chat_memory.add_message(human_msg)

        result = rag_chain.invoke({"question": query})
        print("Chain result:", result)
        cleaned_answer = clean_response(result["answer"])

        
        cleaned_refs = []
        for doc in result.get("source_documents", []):
            cleaned_text = clean_response(doc.page_content)
            cleaned_refs.append({
                "exact_text": cleaned_text,
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "page": doc.metadata.get("page"),
                "figure": None
            })

        
        assistant_msg = AIMessage(content=result["answer"])
        rag_chain.memory.chat_memory.add_message(assistant_msg)

        
        updated_history = []
        for m in rag_chain.memory.chat_memory.messages:
            
            if isinstance(m, HumanMessage):
                role = "human"
            elif isinstance(m, AIMessage):
                role = "assistant"
            else:
                role = "unknown"
            updated_history.append({"role": role, "content": m.content})
        
        save_chat_history(chat_id, updated_history)

        return jsonify({
            "answer": cleaned_answer,
            "references": cleaned_refs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


