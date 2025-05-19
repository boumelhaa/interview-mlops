from typing import List, Dict, Any, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        """
        Initialize the EmbeddingProcessor with a specific model and batch size.
        
        Args:
            model_name (str): Name of the sentence-transformer model to use
            batch_size (int): Number of items to process in each batch
        """
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def read_data(self, input_path: str) -> pd.DataFrame:
        """
        Read data from a Parquet file.
        
        Args:
            input_path (str): Path to the input Parquet file
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If the required columns are missing
        """
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        df = pd.read_parquet(input_path)
        required_columns = ['account_id', 'item_description', 'vendor_name', 
                          'vendor_cuisines', 'feature_timestamp']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    
    def prepare_account_texts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare concatenated text for each account by combining item descriptions,
        vendor names, and cuisines.
        
        Args:
            df (pd.DataFrame): Input dataframe with account data
            
        Returns:
            pd.DataFrame: Dataframe with concatenated text for each account
        """
        # Group by account_id and aggregate the text fields
        account_texts = df.groupby('account_id').agg({
            'item_description': lambda x: ' | '.join(x.unique()),
            'vendor_name': lambda x: ' | '.join(x.unique()),
            'vendor_cuisines': lambda x: ' | '.join(x.unique())
        }).reset_index()
        
        # Create the final text for embedding
        account_texts['text_for_embedding'] = (
            "Items: " + account_texts['item_description'] + 
            " | Vendors: " + account_texts['vendor_name'] + 
            " | Cuisines: " + account_texts['vendor_cuisines']
        )
        
        return account_texts
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            np.ndarray: Array of embeddings with shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def process_data(self, input_path: str, output_path: str) -> None:
        """
        Process the input data by generating account-level embeddings and saving to a new Parquet file.
        
        Args:
            input_path (str): Path to the input Parquet file
            output_path (str): Path where the output Parquet file will be saved
        """
        start_time = time.time()
        
        try:
            # Read data
            logger.info(f"Reading data from {input_path}")
            df = self.read_data(input_path)
            
            # Prepare account-level texts
            logger.info("Preparing account-level texts")
            account_texts = self.prepare_account_texts(df)
            
            # Generate embeddings
            logger.info("Generating account embeddings")
            texts = account_texts['text_for_embedding'].tolist()
            embeddings = self.generate_embeddings(texts)
            
            # Create final output dataframe
            output_df = pd.DataFrame({
                'account_id': account_texts['account_id'],
                'text_for_embedding': texts
            })
            
            # Add embeddings to dataframe
            embedding_columns = [f'embedding_{i}' for i in range(self.embedding_dim)]
            output_df[embedding_columns] = pd.DataFrame(embeddings)
            
            # Save results
            logger.info(f"Saving results to {output_path}")
            output_df.to_parquet(output_path, index=False)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
            logger.info(f"Generated embeddings for {len(output_df)} accounts")
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise

def main():
    processor = EmbeddingProcessor()
    input_path = "data/sample_data.parquet"
    output_path = "data/account_embeddings.parquet"
    processor.process_data(input_path, output_path)

if __name__ == "__main__":
    main() 