import pandas as pd
import numpy as np
import os
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

class AmazonAIAssistant:
    def __init__(self, data_path="data/processed/amazon_sales_cleaned_latest.csv", model_name='all-MiniLM-L6-v2'):
        self.data_path = data_path
        self.df = None
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.model = genai.GenerativeModel('gemini-3-flash-preview')
        
    def prepare_data(self):
        """Loads data and creates a descriptive text column for each product."""
        if not os.path.exists(self.data_path):
            print(f"[ERROR] Data file not found: {self.data_path}")
            return False
            
        self.df = pd.read_csv(self.data_path)
        
        # Create a rich description for indexing
        self.df['search_context'] = self.df.apply(lambda row: 
            f"Product ID: {row['product_id']}. "
            f"Category: {row['product_category']}. "
            f"Price: ${row['price']:.2f}. "
            f"Rating: {row['rating']} stars based on {row['review_count']} reviews. "
            f"Top region: {row['customer_region']}.", axis=1)
        
        print(f"[INFO] Prepared context for {len(self.df)} products.")
        return True

    def build_index(self, index_path="models_saved/faiss_index.bin"):
        """Generates embeddings and builds a FAISS index for similarity search."""
        if self.df is None:
            self.prepare_data()
            
        print("[INFO] Generating embeddings (this may take a minute)...")
        embeddings = self.embedding_model.encode(self.df['search_context'].tolist(), show_progress_bar=True)
        
        # Convert to float32 for FAISS
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Save index
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        print(f"[SUCCESS] Index built and saved to {index_path}")

    def load_index(self, index_path="models_saved/faiss_index.bin"):
        """Loads a pre-built FAISS index."""
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.df = pd.read_csv(self.data_path)
            return True
        return False

    def search_products(self, query, top_k=5):
        """Finds the most relevant products for a natural language query."""
        if self.index is None:
            if not self.load_index():
                print("[ERROR] Index not found. Please build the index first.")
                return []
        
        # Embed query
        query_vector = self.embedding_model.encode([query]).astype('float32')
        
        # Search index
        distances, indices = self.index.search(query_vector, top_k)
        
        # Return matched product rows
        results = self.df.iloc[indices[0]].copy()
        return results.to_dict(orient='records')

    def ask_assistant(self, user_query):
        """The main RAG pipeline: Search -> Augment -> Generate."""
        # 1. Retrieve
        relevant_products = self.search_products(user_query, top_k=4)
        
        if not relevant_products:
            return "Xin lỗi, tôi không tìm thấy sản phẩm nào phù hợp trong kho dữ liệu."

        # 2. Construct Context
        context_str = "\n".join([
            f"- Product {p['product_id']} ({p['product_category']}): ${p['price']}, Rating {p['rating']} sao"
            for p in relevant_products
        ])
        
        # 3. Generate with Gemini
        prompt = f"""
        Bạn là một trợ lý mua sắm thông minh của Amazon. Dưới đây là danh sách các sản phẩm phù hợp nhất với yêu cầu của khách hàng:
        
        {context_str}
        
        Câu hỏi của khách hàng: "{user_query}"
        
        Nhiệm vụ của bạn:
        1. Trả lời câu hỏi của khách hàng một cách thân thiện, chuyên nghiệp.
        2. Phân tích các sản phẩm trên và giải thích tại sao chúng lại phù hợp với yêu cầu của họ.
        3. Nếu có sản phẩm nào nổi bật về giá hoặc đánh giá, hãy nhấn mạnh nó.
        4. Trả lời bằng tiếng Việt.
        """
        
        response = self.model.generate_content(prompt)
        return response.text, relevant_products
