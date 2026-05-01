# 📈 Stock ML Trading Pipeline & Hybrid Recommendation System

Một hệ thống tích hợp đa nhiệm kết hợp giữa **Pipeline phân tích chứng khoán** và **Hệ thống gợi ý sản phẩm Hybrid**, được tối ưu hóa cho hiệu năng cao và khả năng phản hồi thời gian thực.

## 🚀 Tính năng nổi bật (Key Highlights)

### 🛍️ Hệ thống gợi ý Hybrid (Hybrid Recommendation Engine)
*   **Kiến trúc đa tầng**: Phát triển mô hình gợi ý kết hợp (**Hybrid Recommender**) giữa lọc nội dung (**Content-based Filtering**) và phân tích hành vi theo ngữ cảnh.
*   **Xử lý dữ liệu lớn**: Trình xử lý ETL tối ưu cho hơn **50,000 bản ghi sản phẩm Amazon**, trích xuất đặc trưng (Feature Extraction) và xây dựng ma trận tương đồng sử dụng **Cosine Similarity**.
*   **Cá nhân hóa theo khu vực**: Tích hợp thuật toán phân tích xu hướng theo vị trí địa lý (**Regional Trend Analysis**), giúp tăng độ liên quan của gợi ý dựa trên sở thích cộng đồng địa phương.
*   **Real-time API**: Triển khai hệ thống thông qua **FastAPI**, cung cấp các endpoint RESTful có độ trễ thấp, hỗ trợ gợi ý sản phẩm theo thời gian thực.

### 🤖 Trợ lý AI Assistant & Trading Pipeline
*   **RAG Integration**: Tích hợp công nghệ Retrieval-Augmented Generation để truy xuất dữ liệu mô hình và phản hồi thông tin thị trường thông minh.
*   **Technical Analysis**: Tự động tính toán các chỉ số kỹ thuật (RSI, MACD, Bollinger Bands) cho pipeline dự báo chứng khoán.
*   **Interactive Dashboard**: Giao diện Streamlit hiện đại giúp trực quan hóa dữ liệu và tương tác trực tiếp với mô hình.

## 🛠️ Công nghệ sử dụng (Tech Stack)
*   **Language**: Python 3.9+
*   **Data Science**: Pandas, NumPy, Scikit-learn
*   **Web Framework**: FastAPI, Streamlit
*   **AI/ML**: Google Gemini API (GenAI), FAISS (Vector Database)
*   **Tools**: Joblib, Uvicorn, Git

## 📂 Cấu trúc dự án
```text
├── app/                        # Giao diện người dùng (Streamlit & Dashboard)
│   ├── ai_chat_app.py          # Trợ lý AI Chat Assistant
│   └── main.py                 # Dashboard phân tích chính
├── src/                        # Mã nguồn lõi
│   ├── api/                    # RESTful API Endpoints (FastAPI)
│   ├── models/                 # Logic mô hình (Hybrid Recommender, AI Assistant)
│   └── features/               # Feature Engineering & Technical Indicators
├── data/                       # Lưu trữ dữ liệu (Raw/Processed)
├── models_saved/               # Lưu trữ artifacts mô hình (.pkl, vector index)
└── run_pipeline.py             # Entry point thực thi toàn bộ pipeline
```

## ⚙️ Cài đặt & Sử dụng
1. **Clone repository**:
   ```bash
   git clone https://github.com/nguyenson244/integrated-ml-platform.git
   ```
2. **Cài đặt thư viện**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Thiết lập môi trường**: Tạo tệp `.env` và thêm `GEMINI_API_KEY`.
4. **Khởi chạy hệ thống**:
   ```bash
   python run_pipeline.py         # Chạy pipeline huấn luyện & xử lý dữ liệu
   streamlit run app/main.py      # Mở Dashboard tương tác
   ```