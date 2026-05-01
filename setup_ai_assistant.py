from src.models.ai_assistant import AmazonAIAssistant
import os

def run_setup(sample_size=2000):
    """
    Initializes the AI Assistant by loading data and building the FAISS index.
    Using a sample of the data to speed up the process for the demo.
    """
    print(f"--- AI Assistant Setup ---")
    assistant = AmazonAIAssistant()
    
    # Load and prepare data
    if assistant.prepare_data():
        # Optional: Downsample for faster embedding during the first run
        if sample_size and len(assistant.df) > sample_size:
            print(f"[INFO] Sampling {sample_size} records for faster indexing...")
            assistant.df = assistant.df.sample(sample_size, random_state=42)
        
        # Build index
        assistant.build_index()
        print("\n--- Setup Complete! ---")
        print("You can now run the chat app: streamlit run app/ai_chat_app.py")

if __name__ == "__main__":
    run_setup()
