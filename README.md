# AI Research Assistant

A RAG-based question-answering system for the Llama 2 research paper, powered by a fine-tuned LLM and Streamlit.

## Features

- PDF document processing and semantic search
- Fine-tuned AI researcher model via Ollama
- Interactive web interface with source citations

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create the Ollama model
ollama create aiResearcher -f llm_model/Modelfile
```

## Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Project Structure

```
├── app.py              # Streamlit web interface
├── rag.py              # RAG pipeline script
├── requirements.txt    # Python dependencies
├── llm_model/
│   └── Modelfile       # Ollama model configuration
└── source_data/
    └── llama2.pdf      # Research paper
```

## Requirements

- Python 3.10+
- Ollama installed and running
- 8GB+ RAM recommended
