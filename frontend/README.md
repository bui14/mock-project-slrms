# SLRMS Frontend

Next.js 14 (App Router) + TypeScript + Tailwind CSS + Components.

## Cấu trúc Chính
- `app/login`, `app/register` — Trang xác thực (đăng ký luôn tạo tài khoản STUDENT).
- `app/(app)/*` — Khu vực đã đăng nhập, bảo vệ bởi `AppShell` (redirect nếu chưa login / sai role).
  - `dashboard` — Thống kê tổng quan hoạt động hệ thống.
  - `documents/upload` — Upload tài liệu (Giáo viên / Admin).
  - `documents/[id]` — Xem chi tiết tài liệu, tải xuống, xem tóm tắt AI và danh sách câu hỏi gợi ý.
  - `folders` — Quản lý cây thư mục và môn học.
  - `search` — Tìm kiếm thông minh Hybrid Search.
  - `bookmarks` — Danh sách tài liệu yêu thích đã lưu.
  - `admin/users` — Quản lý người dùng, phân quyền Giáo viên / Sinh viên.
- `components/chat/chat-panel.tsx` — Trợ lý Chat AI RAG Đa tài liệu với trích dẫn nguồn.
- `lib/api.ts` — API client dùng chung, tự động đính kèm Token và Refresh Token khi hết hạn.
- `lib/auth-context.tsx` — Quản lý phiên đăng nhập (React Context).

## Chạy Thử (Local Development)
```bash
npm install
cp .env.local.example .env.local   # Chỉnh NEXT_PUBLIC_API_URL nếu cần (mặc định: http://localhost:8000)
npm run dev
```

## Khởi chạy qua Docker Compose
Service `frontend` đã được tích hợp sẵn trong `docker-compose.yml` ở thư mục gốc. Bạn chỉ cần chạy:
```bash
docker compose up --build
```

## Build Production
```bash
npm run build && npm start
```
