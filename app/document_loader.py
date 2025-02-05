from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor

def load_documents(pdf_files):
    def load_pdf(file_path):
        loader = PyPDFLoader(file_path)
        return loader.load()

    data = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(load_pdf, pdf_files)
        for result in results:
            data.extend(result)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    return text_splitter.split_documents(data)