# RAG System with Fine-tuned Phi-2 Model

This project implements a Retrieval-Augmented Generation (RAG) system using a fine-tuned Phi-2 model with ChromaDB for document storage and retrieval.

## Project Structure

- `main.py` - Main application file
- `fine_tune.py` - Script for fine-tuning the Phi-2 model
- `fine_tune_data.json` - Training data for fine-tuning
- `demo.txt` - Demo text file
- `requirements.txt` - Python dependencies
- `db/` - ChromaDB database files (excluded from git)
- `phi2-finetuned/` - Fine-tuned model files (excluded from git)

## Setup

1. Clone this repository:
```bash
git clone https://github.com/ManeetChhabra/RAG.git
cd RAG
```

2. Create a virtual environment:
```bash
python -m venv rag-env
```

3. Activate the virtual environment:
   - Windows: `rag-env\Scripts\activate`
   - macOS/Linux: `source rag-env/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the fine-tuning script (if needed):
```bash
python fine_tune.py
```

2. Run the main application:
```bash
python main.py
```

## Notes

- The `rag-env/` directory contains the virtual environment and is excluded from version control
- Model checkpoints and database files are excluded from git due to their large size
- Make sure to activate the virtual environment before running any scripts

## Requirements

- Python 3.12+
- CUDA-capable GPU (recommended for fine-tuning)
- Sufficient disk space for model files and database
