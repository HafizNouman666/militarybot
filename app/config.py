import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    PDF_DIR = '/media/rapidsai/Data2/Final-Data-for-AI-Proj/militarybot/app/pdfs'
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')