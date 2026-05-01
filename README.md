# Stock ML Trading Pipeline

A comprehensive machine learning pipeline for stock analysis, recommendation, and AI-driven insights.

## Project Structure

```text
stock-ml-trading-pipeline/
├── app/                        # User interface components
│   ├── ai_chat_app.py          # Streamlit AI Chat Assistant
│   └── main.py                 # Main dashboard application
├── data/                       # Data storage (Raw and Processed)
├── models_saved/               # Serialized models and artifacts (FAISS index, etc.)
├── notebooks/                  # Experimental notebooks and EDA
├── src/                        # Core source code
│   ├── api/                    # API endpoints (FastAPI)
│   ├── data/                   # Data ingestion and preprocessing scripts
│   ├── features/               # Feature engineering and transformation logic
│   └── models/                 # Model implementations (Recommender, AI Assistant, Training)
├── run_pipeline.py             # Main entry point to execute the pipeline
├── setup_ai_assistant.py       # Configuration and setup for the AI Assistant
└── requirements.txt            # Project dependencies
```

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Setup environment variables in `.env`
3. Run the pipeline: `python run_pipeline.py`
4. Launch the dashboard: `streamlit run app/main.py`