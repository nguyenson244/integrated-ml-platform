from src.data.loader import load_local_dataset
from src.data.processing import clean_ecommerce_data, save_processed_data
from src.features.build_features import build_product_features, build_region_popularity
from src.models.recommender import HybridRecommender
import pandas as pd
import os

def main():
    print("[INFO] Starting Recommendation Pipeline...")
    
    # 1. Load Data
    print("\n--- Phase 1: Data Ingestion ---")
    raw_df = load_local_dataset("amazon_sales_dataset.csv")
    if raw_df.empty:
        print("[ERROR] Pipeline stopped.")
        return

    # 2. Clean Data
    print("\n--- Phase 2: Data Cleaning ---")
    cleaned_df = clean_ecommerce_data(raw_df)
    
    # Optional: Save cleaned data if not exists
    processed_path = os.path.join("data", "processed", "amazon_sales_cleaned_latest.csv")
    cleaned_df.to_csv(processed_path, index=False)
    print(f"[SUCCESS] Cleaned data saved to {processed_path}")

    # 3. Feature Engineering
    print("\n--- Phase 3: Feature Engineering ---")
    product_df, features_df = build_product_features(cleaned_df)
    region_matrix = build_region_popularity(cleaned_df)
    print(f"Total Unique Products: {product_df.shape[0]}")
    print(f"Feature Matrix Shape: {features_df.shape}")
    print(f"Region Matrix Shape: {region_matrix.shape}")

    # 4. Model Training
    print("\n--- Phase 4: Model Training ---")
    recommender = HybridRecommender()
    recommender.train(product_df, features_df, region_matrix)
    
    # 5. Export Model
    print("\n--- Phase 5: Exporting Model ---")
    recommender.save_model()
    
    # Quick Test
    print("\n--- Phase 6: Quick Test ---")
    test_product_id = product_df['product_id'].iloc[0]
    test_region = "Asia"
    print(f"Testing Recommendation for Product ID: {test_product_id} in Region: {test_region}")
    recs = recommender.recommend(test_product_id, test_region, top_n=3)
    for i, rec in enumerate(recs):
        print(f"  {i+1}. Product {rec['product_id']} ({rec['product_category']}) - Rating: {rec['rating']:.1f}")

    print("\n[SUCCESS] Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
