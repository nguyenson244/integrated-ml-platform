import streamlit as st
import sys
import os

# Add the project root to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ai_assistant import AmazonAIAssistant

# Page Config
st.set_page_config(page_title="Amazon AI Assistant", page_icon="🛍️", layout="wide")

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
    }
    .product-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff9900;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .product-title {
        font-weight: bold;
        color: #232f3e;
    }
    .product-price {
        color: #B12704;
        font-size: 1.1em;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Session State for AI Assistant
if 'assistant' not in st.session_state:
    assistant = AmazonAIAssistant()
    if assistant.load_index():
        st.session_state.assistant = assistant
        st.session_state.ready = True
    else:
        st.session_state.ready = False

# Sidebar
with st.sidebar:
    st.title("🛍️ Amazon AI")
    st.info("Trợ lý mua sắm thông minh sử dụng kỹ thuật RAG (Retrieval-Augmented Generation) và Gemini AI.")
    
    if not st.session_state.ready:
        st.warning("⚠️ Chỉ mục (Index) chưa được tạo!")
        if st.button("Khởi tạo Index ngay (Cần 1-2 phút)"):
            with st.spinner("Đang xử lý dữ liệu..."):
                from setup_ai_assistant import run_setup
                run_setup(sample_size=5000)
                st.session_state.assistant = AmazonAIAssistant()
                st.session_state.assistant.load_index()
                st.session_state.ready = True
                st.rerun()
    
    st.divider()
    st.markdown("### Ví dụ câu hỏi:")
    st.caption("- Tìm món quà công nghệ dưới 100$")
    st.caption("- Tôi muốn mua quần áo thời trang đánh giá tốt")
    st.caption("- Gợi ý đồ dùng nhà bếp phổ biến ở châu Á")

# Main Chat Interface
st.title("💬 Amazon Shopping Assistant")

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "products" in message:
            cols = st.columns(len(message["products"]))
            for idx, prod in enumerate(message["products"]):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="product-card">
                        <div class="product-title">📦 ID: {prod['product_id']}</div>
                        <div>🏷️ {prod['product_category']}</div>
                        <div class="product-price">${prod['price']}</div>
                        <div>⭐ {prod['rating']} stars</div>
                    </div>
                    """, unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Hỏi tôi bất cứ điều gì về sản phẩm..."):
    if not st.session_state.ready:
        st.error("Vui lòng khởi tạo Index ở thanh bên trước khi chat.")
    else:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI Response
        with st.chat_message("assistant"):
            with st.spinner("AI đang suy nghĩ..."):
                try:
                    response_text, products = st.session_state.assistant.ask_assistant(prompt)
                    st.markdown(response_text)
                    
                    # Display products as cards
                    cols = st.columns(len(products))
                    for idx, prod in enumerate(products):
                        with cols[idx]:
                            st.markdown(f"""
                            <div class="product-card">
                                <div class="product-title">📦 ID: {prod['product_id']}</div>
                                <div>🏷️ {prod['product_category']}</div>
                                <div class="product-price">${prod['price']}</div>
                                <div>⭐ {prod['rating']} stars</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "products": products
                    })
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi: {e}")
