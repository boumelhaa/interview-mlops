from typing import List, Dict, Any
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def read_data(self, input_path: str) -> pd.DataFrame:
        pass
    
    def generate_embeddings(self, descriptions: List[str]) -> np.ndarray:
        pass
    
    def process_data(self, input_path: str, output_path: str) -> None:
        pass 