import os
import pandas as pd
from datetime import datetime

def clean_ecommerce_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans e-commerce dataset.
    
    Requirements:
    - Convert order_date to datetime
    - Remove duplicate rows
    - Handle missing values
    - Ensure numeric columns are correct type
    - Do NOT shuffle data
    """
    # Create a copy to avoid mutating the original dataframe
    df_clean = df.copy()
    
    # 1. Convert order_date to datetime
    if 'order_date' in df_clean.columns:
        df_clean['order_date'] = pd.to_datetime(df_clean['order_date'], errors='coerce')
        
    # 2. Remove duplicate rows (maintains order)
    df_clean = df_clean.drop_duplicates()
    
    # 3. Ensure numeric columns are correct type
    # Includes the standard required columns and common ones in the dataset
    numeric_cols = ['price', 'quantity', 'revenue', 'quantity_sold', 'total_revenue', 'discount_percent', 'rating', 'review_count', 'discounted_price']
    
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
    # 4. Handle missing values
    # Drop rows with NaN values resulting from incorrect formats or inherently missing data
    df_clean = df_clean.dropna()
    
    # Optional: Reset index after dropping rows to keep the dataframe continuous
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def save_processed_data(df: pd.DataFrame, base_filename: str = "cleaned_data") -> str:
    """
    Saves processed DataFrame to data/processed/ with a timestamp.
    
    Args:
        df (pd.DataFrame): The processed data.
        base_filename (str): Base name for the saved file.
        
    Returns:
        str: Path to the saved file.
    """
    # 1. Define output directory
    output_dir = os.path.join("data", "processed")
    
    # 2. Handle folder creation if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Create timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_filename}_{timestamp}.csv"
    
    file_path = os.path.join(output_dir, filename)
    
    # 4. Save DataFrame to CSV
    df.to_csv(file_path, index=False)
    print(f"Data successfully saved to: {file_path}")
    
    return file_path
