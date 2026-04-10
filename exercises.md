# Day 7 — Exercises
## Data Foundations: Embedding & Vector Store | Lab Worksheet

---

## Part 1 — Warm-up (Cá nhân)

### Exercise 1.1 — Cosine Similarity in Plain Language

No math required — explain conceptually:

- What does it mean for two text chunks to have high cosine similarity?
- Give a concrete example of two sentences that would have HIGH similarity and two that would have LOW similarity.
- Why is cosine similarity preferred over Euclidean distance for text embeddings?

> **Ghi kết quả vào:** Report — Section 1 (Warm-up)

---

### Exercise 1.2 — Chunking Math

- A document is 10,000 characters. You chunk it with `chunk_size=500`, `overlap=50`. How many chunks do you expect?
- Formula: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
- If overlap is increased to 100, how does this change the chunk count? Why would you want more overlap?

> **Ghi kết quả vào:** Report — Section 1 (Warm-up)

---

## Part 2 — Core Coding (Cá nhân)

Implement all TODOs in `src/chunking.py`, `src/store.py`, và `src/agent.py`. `Document` dataclass và `FixedSizeChunker` đã được implement sẵn làm ví dụ — đọc kỹ để hiểu pattern trước khi implement phần còn lại.

Run `pytest tests/` to check progress.

### Checklist
- [x] `Document` dataclass — ĐÃ IMPLEMENT SẴN
- [x] `FixedSizeChunker` — ĐÃ IMPLEMENT SẴN
- [ ] `SentenceChunker` — split on sentence boundaries, group into chunks
- [ ] `RecursiveChunker` — try separators in order, recurse on oversized pieces
- [ ] `compute_similarity` — cosine similarity formula with zero-magnitude guard
- [ ] `ChunkingStrategyComparator` — call all three, compute stats
- [ ] `EmbeddingStore.__init__` — initialize store (in-memory or ChromaDB)
- [ ] `EmbeddingStore.add_documents` — embed and store each document
- [ ] `EmbeddingStore.search` — embed query, rank by dot product
- [ ] `EmbeddingStore.get_collection_size` — return count
- [ ] `EmbeddingStore.search_with_filter` — filter by metadata, then search
- [ ] `EmbeddingStore.delete_document` — remove all chunks for a doc_id
- [ ] `KnowledgeBaseAgent.answer` — retrieve + build prompt + call LLM

> **Nộp code:** `src/`
> **Ghi approach vào:** Report — Section 4 (My Approach)

---

## Part 3 — So Sánh Retrieval Strategy (Nhóm)

### Exercise 3.0 — Chuẩn Bị Tài Liệu (Giờ đầu tiên)

Mỗi nhóm chọn một domain và chuẩn bị bộ tài liệu:

**Step 1 — Chọn domain:** FAQ, SOP, policy, docs kỹ thuật, recipes, luật, y tế, v.v.

**Step 2 — Thu thập 5-10 tài liệu.** Lưu dưới dạng `.txt` hoặc `.md` vào thư mục `data/`.

> **Tip chuyển PDF sang Markdown:**
> - `pip install marker-pdf` → `marker_single input.pdf output/` (chất lượng cao, giữ cấu trúc)
> - `pip install pymupdf4llm` → `pymupdf4llm.to_markdown("input.pdf")` (nhanh, đơn giản)
> - Hoặc copy-paste nội dung từ PDF/web vào file `.txt`

Ghi vào bảng:

| # | Query | Gold Answer | Chunk nào chứa thông tin? |
|---|-------|-------------|--------------------------|
| 1 | Tiêu chuẩn về trình độ đào tạo, bồi dưỡng đối với giáo viên hạng I | Có bằng thạc sĩ trở lên thuộc ngành đào tạo giáo viên hoặc có bằng thạc sĩ trở lên chuyên ngành phù hợp với môn học giảng dạy và có chứng chỉ nghiệp vụ sư phạm đối với giáo viên trung học phổ thông. Có chứng chỉ bồi dưỡng chuẩn nghề nghiệp giáo viên cơ sở giáo dục thường xuyên. | **Điều 6, khoản 3 (Hạng I)** — `thong-tu-23-2026-tt-bgddt-quy-dinh-ma-so-chuan-nghe-nghiep-va-luong-giao-vien.md` |
| 2 | Xử lý vi phạm QUY ĐỊNH VỀ CÔNG TÁC CHÍNH TRỊ, TƯ TƯỞNG TRONG ĐẢNG như thế nào? | 1. Tổ chức đảng, cán bộ, đảng viên vi phạm Quy định này, tùy theo tính chất, mức độ và hậu quả phải bị xem xét, xử lý kỷ luật theo quy định của Đảng và pháp luật của Nhà nước. 2. Người đứng đầu cấp ủy, tổ chức đảng, cơ quan, đơn vị nếu để xảy ra vi phạm nghiêm trọng, kéo dài trong lĩnh vực công tác chính trị, tư tưởng phải chịu trách nhiệm hoặc trách nhiệm liên đới và bị xem xét xử lý theo quy định của Đảng. | **Điều 14 (Xử lý vi phạm)** — `QUY ĐỊNH VỀ CÔNG TÁC CHÍNH TRỊ, TƯ TƯỞNG TRONG ĐẢNG.md` |
| 3 | Theo Thông tư 05/2026/TT-BKHCN, khi hệ thống trí tuệ nhân tạo bị tấn công đầu độc dữ liệu, tổ chức/cá nhân cần thực hiện những biện pháp gì? | Bảo đảm an ninh của hệ thống trí tuệ nhân tạo: Tổ chức, cá nhân áp dụng biện pháp bảo vệ phù hợp để phòng ngừa, phát hiện, ngăn chặn và ứng phó với các hành vi xâm nhập, chiếm quyền điều khiển, đầu độc dữ liệu, đầu độc mô hình, tấn công đối nghịch, khai thác lỗ hổng, rò rỉ dữ liệu và lạm dụng hệ thống trí tuệ nhân tạo; bảo đảm tính bí mật, toàn vẹn và sẵn sàng của dữ liệu, mô hình, thuật toán và hạ tầng liên quan. | ⚠️ Tài liệu gốc "Thông tư 05/2026/TT-BKHCN" **không có trong data/**. Nội dung gần nhất: **Điều 14, khoản 1** — `luat-tri-tue-nhan-tạo-2025.txt` |
| 4 | Những hành vi nào bị nghiêm cấm trong hoạt động trí tuệ nhân tạo theo Luật AI 2025? | Luật nghiêm cấm việc lợi dụng hệ thống AI để vi phạm pháp luật; sử dụng yếu tố giả mạo để lừa dối hoặc thao túng hành vi con người; lợi dụng điểm yếu của nhóm người dễ bị tổn thương; tạo ra nội dung giả mạo gây nguy hại đến an ninh quốc gia; và thu thập dữ liệu trái phép để phát triển hệ thống AI. | **Điều 7 (Các hành vi bị nghiêm cấm)** — `luat-tri-tue-nhan-tạo-2025.txt` |
| 5 | Theo Nghị định 129/2026/NĐ-CP, dự toán ngân sách hàng năm của các Cơ quan Việt Nam ở nước ngoài được lập bằng đồng tiền nào và căn cứ vào tỷ giá tại thời điểm nào? | Dự toán được lập bằng đồng Việt Nam quy đổi ra đô la Mỹ theo tỷ giá hạch toán tháng 6 năm hiện hành do Bộ Tài chính quy định. | **Điều 8, khoản 1 (Lập dự toán ngân sách)** — `nghi-dinh-129-2026-nd-cp-quan-ly-su-dung-ngan-sach-nha-nuoc-cho-hoat-dong-doi-ngoai.md` |
| 6 | Cơ quan nhà nước sử dụng hệ thống trí tuệ nhân tạo có được để hệ thống tự động đưa ra quyết định cuối cùng không? | Bảo đảm quyết định cuối cùng thuộc thẩm quyền của con người theo quy định của pháp luật; hệ thống trí tuệ nhân tạo không thay thế trách nhiệm của người ra quyết định. | **Điều 27, khoản 2 (Trách nhiệm đạo đức khi ứng dụng AI trong QLNN)** — `luat-tri-tue-nhan-tạo-2025.txt` |

**Step 3 — Thiết kế metadata schema:** Mỗi tài liệu cần ít nhất 2 trường metadata hữu ích (e.g., `category`, `date`, `source`, `language`, `difficulty`).

> **Ghi kết quả vào:** Report — Section 2 (Document Selection)

---

### Exercise 3.1 — Thiết Kế Retrieval Strategy (Mỗi người thử riêng)

Mỗi thành viên **tự chọn strategy riêng** để thử trên cùng bộ tài liệu nhóm.

**Step 1 — Baseline:** Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu. Ghi kết quả.

**Step 2 — Chọn hoặc thiết kế strategy của bạn:**
- Dùng 1 trong 3 built-in strategies với tham số tối ưu, HOẶC
- Thiết kế custom strategy cho domain (ví dụ: chunk by Q&A pairs, by sections, by headers)
- Mỗi thành viên nên thử strategy **khác nhau** để có gì so sánh

```python
class CustomChunker:
    """Your custom chunking strategy for [your domain].

    Design rationale: [explain why this strategy fits your data]
    """

    def chunk(self, text: str) -> list[str]:
        # Your implementation here
        ...
```

**Step 3 — So sánh:** Custom/tuned strategy vs baseline trên cùng tài liệu.

> **Ghi kết quả vào:** Report — Section 3 (Chunking Strategy)

---

### Exercise 3.2 — Chuẩn Bị Benchmark Queries

Mỗi nhóm viết **đúng 5 benchmark queries** kèm **gold answers**.

| # | Query | Gold Answer (câu trả lời đúng) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |

**Yêu cầu:**
- Queries phải đa dạng (không hỏi 5 câu giống nhau)
- Gold answers phải cụ thể và có thể verify từ tài liệu
- Ít nhất 1 query yêu cầu metadata filtering để trả lời tốt

> **Ghi kết quả vào:** Report — Section 6 (Results — Benchmark Queries & Gold Answers)

---

### Exercise 3.3 — Cosine Similarity Predictions (Cá nhân)

Call `compute_similarity()` on 5 pairs of sentences. **Before running**, predict which pairs will have highest/lowest similarity. Record your predictions and the actual results. Reflect on what surprised you most.

> **Ghi kết quả vào:** Report — Section 5 (Similarity Predictions)

---

### Exercise 3.4 — Chạy Benchmark & So Sánh Trong Nhóm

**Step 1:** Mỗi thành viên chạy 5 benchmark queries với strategy riêng. Ghi kết quả top-3 cho mỗi query.

**Step 2:** So sánh kết quả trong nhóm:
- Strategy nào cho retrieval tốt nhất? Tại sao?
- Có query nào mà strategy A tốt hơn B nhưng ngược lại ở query khác?
- Metadata filtering có giúp ích không?

**Step 3:** Thảo luận và rút ra bài học — chuẩn bị cho phần demo với các nhóm khác.

> **Ghi kết quả vào:** Report — Section 6 (Results)
> **Gợi ý đánh giá:** xem checklist ngắn trong `README.md` mục **Cách Tự Đánh Giá Kết Quả Retrieval** hoặc chi tiết hơn trong `docs/EVALUATION.md`.

---

### Exercise 3.5 — Failure Analysis

Tìm ít nhất **1 failure case** trong quá trình so sánh. Mô tả:
- Query nào retrieval thất bại?
- Tại sao? (chunk quá nhỏ/lớn, metadata thiếu, query mơ hồ, v.v.)
- Đề xuất cải thiện?

> **Ghi kết quả vào:** Report — Section 7 (What I Learned)
> **Gợi ý:** failure analysis nên tham chiếu các góc nhìn như precision, chunk coherence, metadata utility, và grounding quality.

---

## Submission Checklist

- [ ] All tests pass: `pytest tests/ -v`
- [ ] `src/` updated (cá nhân)
- [ ] Report completed (`report/REPORT.md` — 1 file/sinh viên)
