from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

def create_chain(retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=1000)
    
    document_prompt = PromptTemplate.from_template(
        "Exact text: {page_content}\nSource: {source}\nPage: {page}"
    )
     
    system_prompt = """
            You are a factual assistant. Your task is to provide answers strictly based on the provided context. Follow these steps:

            ### **Step 1: Validate the User's Question**
            - If the question is **nonsensical, gibberish, or incomprehensible**, respond with:
            {{
                "answer": null,
                "references": null
            }}

            - Otherwise, proceed to Step 2.

            ### **Step 2: Find an Exact Text Match in Context**
            - Scan the provided context for a **direct text snippet** that precisely answers the user's query.
            - A valid snippet must be a **verbatim substring** from the context.

            ### **Step 3: Construct the Response**
            - **If an exact text snippet is found**, return the following JSON:
            {{
                "answer": "{exact_text_snippet}",
                "references": [
                    {{
                        "exact_text": "{exact_text_snippet}",
                        "source": "{source}",
                        "page": {page_number},
                        "figure": {figure_number}
                    }}
                ]
            }}

            - **If no exact text snippet is found**, return:
            {{
                "answer": null,
                "references": null
            }}

            ### **Rules & Constraints**
            - **DO NOT** infer or generate an answer if no exact snippet exists.
            - **DO NOT** provide any explanation beyond the required JSON.
            - **ONLY** use the input variables: `{{question}}`, `{{context}}`, and `{{chat_history}}`.
            - **Escape curly braces** in examples using double curly braces.

            ---
            ### **Provided Context:**
            {context}

            ### **Conversation History:**
            {chat_history}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt)

    # Create a conversation memory object to hold chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory.output_key = "answer"

    return ConversationalRetrievalChain.from_llm(
        llm, retriever, memory=memory ,return_source_documents=True ,output_key="answer"
    )