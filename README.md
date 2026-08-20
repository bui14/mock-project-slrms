# Smart Learning Resource Management System (SLRMS)

Nền tảng Quản lý & Tra cứu Tài liệu Học tập Thông minh tích hợp AI, Hybrid Search & Chat RAG Đa tài liệu.

---

## 1. Kiến trúc Hệ thống

Dự án áp dụng mô hình kiến trúc phân lớp **Clean / Layered Architecture 4 lớp**:

```
API Layer (FastAPI Routers) ──> Service Layer ──> Repository Layer ──> Database (PostgreSQL 16 + pgvector)
```

### Cấu trúc Thư mục Dự án:

```
.
├── docker-compose.yml           # Cấu hình container hóa 5 dịch vụ (DB, Redis, Migrate, Backend, Frontend)
├── .env.example                 # File mẫu cấu hình biến môi trường
├── db/
│   └── init.sql                 # Khởi tạo CSDL PostgreSQL và extension pgvector
├── backend/                     # FastAPI Backend (Python 3.11)
│   ├── alembic/                 # Quản lý Database Migrations
│   └── app/
│       ├── api/routers/         # Các API Endpoints (Auth, Search, Chat, Documents, Folders, Dashboard, Social)
│       ├── core/                # Cấu hình hệ thống, Database session, Bảo mật JWT/Bcrypt
│       ├── models/              # SQLAlchemy Models (User, Document, DocumentChunk, Folder, Bookmark...)
│       ├── repositories/        # Tầng truy vấn CSDL (Repository Pattern)
│       ├── schemas/             # Pydantic Schemas request/response
│       ├── services/            # Tầng xử lý logic nghiệp vụ & tích hợp AI Gemini
│       ├── utils/               # Trích xuất văn bản PDF, Chunking, Vector Embeddings, Stopwords
│       ├── seed.py              # Khởi tạo tài khoản Admin mặc định
│       └── main.py              # Khởi chạy FastAPI Application
└── frontend/                    # Web Client (Next.js 14 App Router, TypeScript, Tailwind CSS)
    ├── app/                     # Next.js Pages & Routes (Login, Dashboard, Documents, Search, Bookmarks...)
    ├── components/              # UI Components & Chat Panel
    └── lib/                     # API Client, Auth Context, Toast Context, Types & Utilities
```

---

## 2. Công nghệ Sử dụng

| Hạng mục | Công nghệ / Thư viện | Vai trò & Mô tả |
|---|---|---|
| **Backend Framework** | Python FastAPI, SQLAlchemy 2.0, Pydantic v2 | Xây dựng RESTful API bất đồng bộ (Async) hiệu năng cao |
| **Database & Vector** | PostgreSQL 16 + `pgvector` extension | Lưu trữ dữ liệu quan hệ & Vector Embeddings 384 chiều |
| **Artificial Intelligence** | Gemini 2.5 Flash API & Local Embedder | Chat RAG đa tài liệu, Tóm tắt & Tự động sinh câu hỏi gợi ý |
| **Cache & Task Queue** | Redis 7 & FastAPI BackgroundTasks | Lưu trữ tạm & xử lý tác vụ nền cho tài liệu PDF |
| **Frontend Framework** | Next.js 14 (App Router), TypeScript, Tailwind CSS | Giao diện Web tương tác mượt mà, phản hồi nhanh |
| **Containerization** | Docker & Docker Compose | Đóng gói toàn bộ ứng dụng chạy đồng bộ |

---

## 3. Hướng dẫn Cài đặt & Khởi chạy Dự án

### 🚀 Cách 1: Khởi chạy nhanh bằng Docker Compose (Khuyên dùng)

#### Bước 1: Thao tác file cấu hình môi trường `.env`
Sao chép file `.env.example` thành `.env`:
```bash
cp .env.example .env
```
*(Nếu muốn sử dụng AI Gemini cho Chat RAG & Sinh câu hỏi gợi ý, hãy điền khóa `GEMINI_API_KEY=your_api_key` vào file `.env`)*.

#### Bước 2: Khởi chạy toàn bộ hệ thống bằng Docker Compose
```bash
docker compose up --build
```
> **Lưu ý:** Container `slrms-migrate` sẽ tự động khởi chạy trước để thực hiện Database Migration (`alembic upgrade head`) và tạo sẵn tài khoản Admin mặc định (`app.seed`).

#### Bước 3: Truy cập hệ thống
Sau khi các container báo trạng thái chạy thành công:
- **Frontend Web UI:** [http://localhost:3000](http://localhost:3000)
- **Backend API Server:** [http://localhost:8000](http://localhost:8000)
- **Tài liệu API (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)

#### Bước 4: Đăng nhập tài khoản Admin mẫu
- **Email:** `admin@slrms.com`
- **Mật khẩu:** `Admin@123`

---

### 💻 Cách 2: Khởi chạy thủ công từng phần (Local Development)

#### 1. Khởi chạy CSDL PostgreSQL (với extension pgvector)
Bạn có thể chạy riêng CSDL bằng Docker:
```bash
docker compose up -d db redis
```

#### 2. Khởi chạy Backend (FastAPI)
```bash
cd backend
python -m venv .venv
# Trên Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Trên Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
alembic upgrade head
python -m app.seed
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 3. Khởi chạy Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
```
Giao diện sẽ chạy tại [http://localhost:3000](http://localhost:3000).

---

## 4. Các Tính năng Nổi bật đã Triển khai

1. **Tìm kiếm Ngữ nghĩa Thông minh (Dual-Path RRF Hybrid Search):**
   - Dung hợp thứ hạng **Reciprocal Rank Fusion (RRF)** giữa So khớp từ khóa SQL và Vector Similarity (`pgvector`).
   - Tích hợp thuật toán **Adaptive RankCut** cắt đứt nhiễu đuôi dài khi tìm kiếm câu hỏi phức tạp.
   - Xử lý dứt điểm lỗi trượt kết quả do dính chữ PDF Tiếng Việt (`%vận%trù%học%`).
2. **Trợ lý Chat AI RAG Đa tài liệu (Multi-Document Chat):**
   - Truy vấn ngữ cảnh chính xác từ các khối văn bản (chunks).
   - Tự động trích dẫn tên tài liệu nguồn (`citations`) trong câu trả lời.
3. **Tự động Sinh Câu hỏi Gợi ý cho Tài liệu (Suggested Questions Generation):**
   - Trích mẫu đại diện 3 phân vùng (Head-Middle-Tail) cho file PDF dài.
   - AI Gemini tự động gợi ý 3 câu hỏi học tập định hướng kèm bộ câu hỏi mặc định dự phòng.
4. **Quản lý Tài liệu & Thư mục:**
   - Phân loại tài liệu theo Thư mục/Môn học/Tag.
   - Quyền truy cập công khai tài liệu và tính năng Bookmark cá nhân độc lập.

---

## 5. Chạy Kiểm thử (Unit Testing)

Để chạy toàn bộ bộ kiểm thử tự động của Backend:

```bash
# Kiểm thử trên môi trường local venv:
$env:PYTHONPATH="backend"; .\.venv\Scripts\python -m unittest backend/tests/test_search.py backend/tests/test_retrieval_service.py backend/tests/test_chat.py backend/tests/test_chat_service.py

# Hoặc kiểm thử trực tiếp trong Docker container:
docker compose exec backend python -m unittest discover -s tests
```
*(Kết quả kiểm thử: **26/26 Unit Tests Passed 100%**)*.
