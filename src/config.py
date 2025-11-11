import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
    UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "./model") 

    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

    ALLOWED_EXTENSIONS = {".txt", ".pdf", ".md"}
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))

config = Config()