from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def create_chain(retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=1000)
    
    document_prompt = PromptTemplate.from_template(
        "Exact text: {page_content}\nSource: {source}\nPage: {page}"
    )
    
    system_prompt = """
    
                You are a factual assistant. Follow these steps:
                1. Provide a direct answer.
                2. Extract EXACT text snippets with sources from the provided context.
                3. Structure the response as PROPER JSON (escape curly braces):
                    {{
                        "answer": "Direct answer...",
                        "references": [
                            {{
                                "exact_text": "Text...",
                                "source": "file.pdf",
                                "page": 1,
                                "figure": null
                            }}
                        ]
                    }}

                **Rules:** 
                - Use DOUBLE curly braces for JSON example
                - Only use actual input variables: {{input}} and {{context}}

                **Context:** 
                {context}

"""  # Your existing system prompt
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt)
    return create_retrieval_chain(retriever, question_answer_chain)