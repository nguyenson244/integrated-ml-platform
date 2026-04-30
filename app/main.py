import streamlit as st
import pandas as pd
import requests
import os

# Set page config
st.set_page_config(page_title="E-Commerce Hybrid Recommender", layout="wide", page_icon="🛍️")

st.title("🛍️ E-Commerce Cross-Selling Dashboard")
st.markdown("This dashboard suggests similar products using **Content-Based Filtering** (Price, Rating, Category) and boosts recommendations using **Region-Based Popularity**.")

# 1. Load basic product info for the dropdowns
@st.cache_data
def load_data():
    file_path = os.path.join("data", "processed", "amazon_sales_cleaned_latest.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Get unique products for the dropdown
        products = df.drop_duplicates(subset=['product_id']).sort_values('product_id')
        regions = df['customer_region'].unique().tolist()
        return products, regions
    return pd.DataFrame(), []

products_df, regions = load_data()

if products_df.empty:
    st.error("Processed data not found. Please run the data pipeline first.")
    st.stop()

# 2. Sidebar for User Input
st.sidebar.header("User Context")

# Select a Product
product_list = products_df['product_id'].astype(str) + " - " + products_df['product_category']
selected_str = st.sidebar.selectbox("Select Product Currently Viewing:", product_list)
selected_product_id = int(selected_str.split(" - ")[0])

# Select Region
selected_region = st.sidebar.selectbox("Customer Region:", regions)

# Number of recommendations
top_n = st.sidebar.slider("Number of Recommendations:", 3, 10, 5)

# Show current product details
current_product = products_df[products_df['product_id'] == selected_product_id].iloc[0]
st.subheader("🛒 You are currently viewing:")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Product ID", current_product['product_id'])
col2.metric("Category", current_product['product_category'])
col3.metric("Price ($)", round(current_product['price'], 2))
col4.metric("Rating", round(current_product['rating'], 1))

st.markdown("---")

# 3. Fetch Recommendations from API
st.subheader("✨ Frequently Bought Together (Recommendations)")

if st.button("Get Recommendations"):
    with st.spinner("Calculating hybrid recommendations..."):
        # Make a POST request to our FastAPI backend
        api_url = "http://localhost:8000/recommend"
        payload = {
            "product_id": selected_product_id,
            "customer_region": selected_region,
            "top_n": top_n
        }
        
        try:
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                recs = response.json().get("recommendations", [])
                
                if not recs:
                    st.warning("No recommendations found.")
                else:
                    # Display recommendations in columns
                    cols = st.columns(len(recs))
                    for i, rec in enumerate(recs):
                        with cols[i]:
                            st.info(f"**Top {i+1}**")
                            st.write(f"**Product ID:** {rec['product_id']}")
                            st.write(f"**Category:** {rec['product_category']}")
                            st.write(f"**Price:** ${rec['price']:.2f}")
                            st.write(f"**Rating:** {rec['rating']:.1f} ⭐️")
                            
                            # Show the final hybrid score
                            st.caption(f"Match Score: {rec['final_score']:.2f}")
                            
            else:
                st.error(f"API Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the Recommendation API. Ensure the FastAPI server is running on port 8000.")
            st.code("uvicorn src.api.main:app --reload")

