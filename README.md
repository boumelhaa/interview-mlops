# MLOps Interview Assignment

## Overview
This assignment tests your ability to work with data processing, embeddings, and MLOps best practices. You will create a Python class that processes data from a Parquet file, generates account-level embeddings by combining item descriptions, vendor names, and cuisines, and outputs the results to a new Parquet file.

## Project Structure
```
.
├── data/
│   └── generate_data.py
├── src/
│   └── embedding_processor.py
├── requirements.txt
└── README.md
```

## Assignment Steps

1. **Data Generation**
   - Run the data generation script to create sample data
   - The script will create a Parquet file with 1000 records
   - Fields: account_id, item_description, vendor_name, vendor_cuisines, feature_timestamp

2. **Implementation Task**
   Create a class called `EmbeddingProcessor` in `src/embedding_processor.py` that:
   - Reads data from the generated Parquet file
   - Groups data by account_id
   - Concatenates item descriptions, vendor names, and cuisines for each account
   - Generates embeddings for the concatenated text using Sentence Transformers
   - Outputs the processed data with account-level embeddings to a new Parquet file

3. **Requirements**
   - Use pandas for data manipulation
   - Use sentence-transformers for generating embeddings
   - Implement proper error handling
   - Add type hints
   - Include docstrings
   - Write clean, maintainable code
   - Handle text concatenation efficiently
   - Ensure proper grouping of account data

## Getting Started

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate sample data:
```bash
python data/generate_data.py
```

4. Implement your solution in `src/embedding_processor.py`

## Expected Output
The output Parquet file should contain:
- account_id: The unique identifier for each account
- text_for_embedding: The concatenated text used for embedding
- embedding_0 to embedding_N: The embedding vectors for each account

## Evaluation Criteria
- Code quality and organization
- Error handling
- Documentation
- Performance considerations
- MLOps best practices
- Efficient text concatenation and grouping
- Memory optimization for large datasets

## Time Limit
- 2 hours

Good luck!