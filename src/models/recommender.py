import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class HybridRecommender:
    def __init__(self):
        self.product_df = None
        self.similarity_matrix = None
        self.region_matrix = None
        self.product_index_map = {}

    def train(self, product_df: pd.DataFrame, features_df: pd.DataFrame, region_matrix: pd.DataFrame):
        """
        Calculates similarity matrix and stores necessary structures.
        """
        print("[INFO] Training Recommender Engine...")
        self.product_df = product_df
        self.region_matrix = region_matrix
        
        # Mapping product_id to matrix index
        self.product_index_map = {prod_id: idx for idx, prod_id in enumerate(features_df['product_id'])}
        
        # Calculate Cosine Similarity on feature vectors (drop product_id col)
        feature_vectors = features_df.drop(columns=['product_id']).values
        self.similarity_matrix = cosine_similarity(feature_vectors)
        
        print("[INFO] Training completed.")

    def recommend(self, product_id: int, customer_region: str, top_n: int = 5) -> list:
        """
        Returns a list of recommended product_ids.
        Combines Content-Based similarity and Region-Based popularity.
        """
        if product_id not in self.product_index_map:
            return []
            
        # 1. Content-Based: Get similar products
        idx = self.product_index_map[product_id]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort by similarity (descending)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get Top 50 similar items first to apply region filter
        top_50_indices = [i[0] for i in sim_scores[1:51]] # Skip the first one (itself)
        
        top_50_products = self.product_df.iloc[top_50_indices].copy()
        top_50_products['similarity_score'] = [i[1] for i in sim_scores[1:51]]
        
        # 2. Region-Based Hybridization
        # Boost products whose category is popular in the customer's region
        if customer_region in self.region_matrix.index:
            region_prefs = self.region_matrix.loc[customer_region]
            
            # Function to get boost score based on category popularity
            def get_boost(category):
                if category in region_prefs:
                    return region_prefs[category]
                return 0
                
            # Apply boost: Final Score = Similarity + (Popularity * weight)
            # Weight determines how much region affects the recommendation
            weight = 0.5 
            top_50_products['region_boost'] = top_50_products['product_category'].apply(get_boost)
            top_50_products['final_score'] = top_50_products['similarity_score'] + (top_50_products['region_boost'] * weight)
            
            # Re-sort by final score
            top_50_products = top_50_products.sort_values(by='final_score', ascending=False)
            
        # Return the top N product IDs
        return top_50_products.head(top_n).to_dict(orient='records')

    def save_model(self, model_dir: str = "models_saved"):
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "hybrid_recommender.pkl")
        joblib.dump(self, model_path)
        print(f"[SUCCESS] Model saved to {model_path}")

    @staticmethod
    def load_model(model_path: str = "models_saved/hybrid_recommender.pkl"):
        return joblib.load(model_path)
