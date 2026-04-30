import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates order data into a unique products dataset and extracts features
    for Content-Based Filtering.
    """
    # 1. Aggregate to get unique products
    # A product should have one category, but might have slightly varying prices/ratings in orders
    product_df = df.groupby('product_id').agg({
        'product_category': 'first',
        'price': 'mean',
        'rating': 'mean',
        'review_count': 'mean'
    }).reset_index()
    
    # 2. Setup transformers
    # One-Hot Encode 'product_category'
    # Scale 'price', 'rating', 'review_count'
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['price', 'rating', 'review_count']),
            ('cat', OneHotEncoder(), ['product_category'])
        ])
    
    # 3. Fit and transform
    feature_matrix = preprocessor.fit_transform(product_df)
    
    # Convert sparse matrix back to dataframe for easier handling (if not too large)
    # Get feature names for OHE
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(['product_category'])
    all_features = ['price_scaled', 'rating_scaled', 'review_count_scaled'] + list(cat_features)
    
    features_df = pd.DataFrame(feature_matrix, columns=all_features)
    features_df['product_id'] = product_df['product_id']
    
    # Reorder to keep product_id first
    cols = ['product_id'] + all_features
    features_df = features_df[cols]
    
    return product_df, features_df

def build_region_popularity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a Region-Category popularity matrix for Region-Based Filtering.
    """
    # Calculate how many items of each category are sold in each region
    region_cat = df.groupby(['customer_region', 'product_category'])['quantity_sold'].sum().reset_index()
    
    # Calculate total sold in the region to get a percentage/score
    region_total = region_cat.groupby('customer_region')['quantity_sold'].sum().reset_index()
    region_total.rename(columns={'quantity_sold': 'total_region_sold'}, inplace=True)
    
    popularity_df = pd.merge(region_cat, region_total, on='customer_region')
    popularity_df['popularity_score'] = popularity_df['quantity_sold'] / popularity_df['total_region_sold']
    
    # Pivot to make it a matrix: index=region, columns=category
    region_matrix = popularity_df.pivot(index='customer_region', columns='product_category', values='popularity_score').fillna(0)
    
    return region_matrix
