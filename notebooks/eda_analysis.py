import pandas as pd
import os

def run_eda():
    print("=== EXPLORATORY DATA ANALYSIS (EDA) ===")
    file_path = os.path.join("data", "processed", "amazon_sales_cleaned_20260430_021324.csv")
    
    # Try to find the latest processed file if the hardcoded one doesn't exist
    if not os.path.exists(file_path):
        processed_dir = os.path.join("data", "processed")
        files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
        if not files:
            print("[ERROR] No processed data found.")
            return
        # Get the most recent file
        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(processed_dir, x)))
        file_path = os.path.join(processed_dir, latest_file)
        
    df = pd.read_csv(file_path)
    print(f"[INFO] Loaded data from: {file_path}")
    print(f"[INFO] Dataset Shape: {df.shape}")
    
    print("\n--- 1. Top 5 Best-Selling Product Categories ---")
    top_cats = df['product_category'].value_counts().head(5)
    print(top_cats)
    
    print("\n--- 2. Sales by Customer Region ---")
    region_sales = df.groupby('customer_region')['total_revenue'].sum().sort_values(ascending=False)
    print(region_sales)
    
    print("\n--- 3. Top Category per Region (Region-based Insight) ---")
    region_cat = df.groupby(['customer_region', 'product_category']).size().reset_index(name='count')
    # Get max count category for each region
    idx = region_cat.groupby('customer_region')['count'].idxmax()
    top_per_region = region_cat.loc[idx]
    print(top_per_region)

    print("\n--- 4. Numerical Columns Summary ---")
    print(df[['price', 'quantity_sold', 'rating', 'total_revenue']].describe().round(2))
    
    print("\n=== EDA COMPLETED ===")

if __name__ == "__main__":
    run_eda()
