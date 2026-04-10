# REPORT — Day 7: Data Foundations: Embedding & Vector Store

**Sinh viên:** Trần Ngọc Huy  
**MSSV:** 2A202600298  
**Ngày nộp:** 10/04/2026

---

## Section 1 - Warm-up

### Exercise 1.1 - Cosine Similarity in Plain Language

**High cosine similarity nghĩa là gì?**

Hai đoạn văn có cosine similarity cao nghĩa là các vector embedding của chúng *hướng về cùng một phía* trong không gian nhiều chiều. Cụ thể hơn, chúng chia sẻ nhiều khái niệm, từ ngữ và ý nghĩa tương tự nhau. Cosine similarity không đo độ dài vector (tức độ dài câu) mà chỉ đo **góc** giữa hai vector — nên câu ngắn và câu dài nói về cùng chủ đề vẫn có similarity cao.

**Ví dụ HIGH similarity:**
- *"Hệ thống AI bị nghiêm cấm thu thập dữ liệu trái phép."*
- *"Việc thu thập dữ liệu bất hợp pháp bằng trí tuệ nhân tạo là vi phạm pháp luật."*

→ Hai câu khác cấu trúc nhưng cùng ý nghĩa, embedding sẽ gần nhau.

**Ví dụ LOW similarity:**
- *"Dự toán ngân sách nhà nước cho hoạt động đối ngoại."*
- *"Tiêu chuẩn về trình độ đào tạo của giáo viên hạng I."*

→ Hai câu hoàn toàn khác chủ đề, embedding sẽ xa nhau.

**Tại sao cosine similarity ưu việt hơn Euclidean distance?**

Euclidean distance bị ảnh hưởng bởi **độ lớn** của vector. Tài liệu dài (vector magnitude lớn) và tài liệu ngắn nói cùng chủ đề sẽ có khoảng cách Euclidean lớn dù nội dung giống nhau. Cosine similarity chuẩn hóa (chia cho magnitude), chỉ giữ lại thông tin về **hướng** — tức là chủ đề và ý nghĩa — nên phù hợp hơn cho text.

---

### Exercise 1.2 — Chunking Math

**Tài liệu 10.000 ký tự, chunk_size=500, overlap=50:**

```
num_chunks = ceil((10000 - 50) / (500 - 50))
           = ceil(9950 / 450)
           = ceil(22.11)
           = 23 chunks
```

**Nếu tăng overlap lên 100:**

```
num_chunks = ceil((10000 - 100) / (500 - 100))
           = ceil(9900 / 400)
           = ceil(24.75)
           = 25 chunks
```

Tăng overlap → tăng số chunks (thêm 2 chunks). Bước nhảy `step = chunk_size - overlap` nhỏ hơn nên nhiều chunks hơn được tạo ra từ cùng lượng văn bản. **Tại sao muốn overlap lớn hơn?** Để thông tin tại ranh giới giữa hai chunks không bị mất — với văn bản pháp luật, một quy định có thể trải dài qua điểm cắt, overlap giúp cả hai chunks kề nhau đều "biết" về phần đó.

---

## Section 2 — Document Selection

### Domain đã chọn: Văn bản Pháp luật Việt Nam

Domain này phù hợp để benchmark vì cấu trúc rõ ràng theo Chương → Điều → Khoản → Điểm; mỗi Điều là đơn vị ý nghĩa độc lập; câu hỏi pháp lý thường hỏi về một Điều cụ thể nên dễ đánh giá retrieval precision.

### Bảng tài liệu

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata |
|---|---|---|---|---|
| 1 | Luật Trí tuệ nhân tạo 2025 | Quốc hội (134/2025/QH15) | 49.163 | category=luat, issuer=Quoc hoi, year=2025 |
| 2 | Quy định về Công tác Chính trị, Tư tưởng trong Đảng | BCH TW Đảng (19-QĐ/TW) | 52.266 | category=quy-dinh, issuer=BCH TW Dang, year=2026 |
| 3 | Thông tư 23/2026/TT-BGDĐT — Chuẩn nghề nghiệp Giáo viên | Bộ GD&ĐT | 30.075 | category=thong-tu, issuer=Bo GD&DT, year=2026 |
| 4 | Nghị định 129/2026/NĐ-CP — Quản lý Ngân sách Đối ngoại | Chính phủ | 17.512 | category=nghi-dinh, issuer=Chinh phu, year=2026 |
| 5 | Nghị định 102/2021/NĐ-CP — Xử phạt Vi phạm Hành chính | Chính phủ | 36.219 | category=nghi-dinh, issuer=Chinh phu, year=2021 |

### Metadata Schema

```python
{
    "category": str,   # "luat" | "nghi-dinh" | "thong-tu" | "quy-dinh"
    "issuer":   str,   # cơ quan ban hành
    "year":     str,   # năm ban hành
    "lang":     str,   # "vi"
    "source":   str,   # tên file gốc
    "doc_id":   str,   # id tài liệu cha (dùng để delete/filter)
}
```

---

## Section 3 — Chunking Strategy

### Strategy đã chọn: `LegalArticleChunker` (Custom)

**Lý do thiết kế:**

Các chunker có sẵn không "biết" cấu trúc pháp lý. FixedSize với chunk_size=300 cắt ngang giữa một Điều luật, làm mất ngữ cảnh pháp lý. `LegalArticleChunker` tách đúng tại ranh giới `Điều X.` — giữ nguyên mỗi điều luật như một đơn vị chunk hoàn chỉnh, đây là đơn vị truy vấn tự nhiên nhất trong văn bản pháp lý.

**Cơ chế hoạt động:**

```python
ARTICLE_PATTERN = re.compile(r"(?=\n\s*Điều\s+\d+[\.\.])")

def chunk(self, text: str) -> list[str]:
    raw_parts = self.ARTICLE_PATTERN.split(text)
    chunks = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= self.max_article_chars:   # vừa → giữ nguyên
            chunks.append(part)
        else:                                      # quá dài → fallback recursive
            chunks.extend(self._fallback.chunk(part))
    return chunks
```

1. Regex lookahead tách văn bản tại mỗi ranh giới `Điều X.`
2. Điều quá dài (> 1500 chars) → dùng `RecursiveChunker` để tách nhỏ hơn
3. Strip whitespace, bỏ chunk rỗng

**So sánh Baseline vs Custom (chunk_size=300):**

| Strategy | Số Chunks (Luật AI) | Avg Length | Số Chunks (Quy định Đảng) | Avg Length |
|---|---|---|---|---|
| FixedSizeChunker | 182 | 300.0 | 194 | 299.3 |
| SentenceChunker | 121 | 403.9 | 111 | 468.1 |
| RecursiveChunker | 215 | 226.8 | 239 | 216.9 |
| **LegalArticleChunker** | **52** | **943.5** | **53** | **984.2** |

**Toàn bộ 5 tài liệu:**

| Tài liệu | LegalArticle chunks | Avg len |
|---|---|---|
| Luật AI 2025 | 52 | 943.5 |
| Quy định Đảng | 53 | 984.2 |
| Thông tư 23 | 24 | 1251.2 |
| NĐ 129 | 14 | 1249.3 |
| NĐ 102 | 28 | 1291.6 |

**Nhận xét:** LegalArticleChunker tạo ít chunks hơn nhiều (52 vs 182-215) nhưng mỗi chunk **trọn nghĩa hơn** vì bằng 1 Điều luật. Avg length ~950-1291 ký tự phù hợp với văn bản pháp lý. RecursiveChunker tạo quá nhiều chunk nhỏ (~220 chars), dễ mất context liên điều.

---

## Section 4 — My Approach (Phase 1)

### Tổng quan kiến trúc đã implement

**`SentenceChunker`:** Dùng regex `(?<=[.!?])\s+` (lookbehind) để tách câu — giữ dấu câu ở cuối câu, sau đó nhóm theo `max_sentences_per_chunk`. Ưu điểm: không cắt giữa câu; nhược điểm: chunk size biến đổi nhiều.

**`RecursiveChunker`:** Thử separator theo thứ tự ưu tiên `["\n\n", "\n", ". ", " ", ""]`. Với mỗi separator: tách → kiểm tra từng mảnh → nếu vừa chunk_size thì giữ, nếu quá lớn thì đệ quy với separator tiếp theo → cuối cùng **merge** các mảnh nhỏ lại để tận dụng chunk_size. Merge là cải tiến quan trọng, tránh tạo quá nhiều chunk cực nhỏ.

**`compute_similarity`:** Áp dụng công thức cosine chuẩn với zero-magnitude guard (`return 0.0` nếu một vector có magnitude = 0).

**`EmbeddingStore`:** In-memory store dùng `list[dict]`. Mỗi record: `{id, content, embedding, metadata}`. Search dùng dot product (= cosine với unit vector). `search_with_filter` filter metadata trước, rồi gọi `_search_records` trên tập đã lọc — tăng precision khi domain rõ ràng.

**`KnowledgeBaseAgent`:** RAG 3 bước: retrieve top-k → build prompt với context đánh số + score → gọi `llm_fn`. Prompt hướng dẫn LLM chỉ dùng context đã cung cấp, tránh hallucination.

**Kết quả:** 42/42 tests pass.

---

## Section 5 — Similarity Predictions

Dùng `_mock_embed` (deterministic hash-based, không phải semantic embedding thật).

| # | Cặp câu | Dự đoán | Thực tế |
|---|---|---|---|
| 1 | "Hành vi bị nghiêm cấm trong AI" vs "Các hành vi cấm khi dùng hệ thống AI" | **Cao** (paraphrase) | **0.1421** |
| 2 | "Tiêu chuẩn trình độ giáo viên hạng I" vs "Yêu cầu bằng cấp bổ nhiệm giáo viên" | **Cao** (cùng chủ đề) | **0.0180** |
| 3 | "Dự toán ngân sách đối ngoại" vs "Đào tạo nghề nghiệp giáo viên" | **Thấp** (khác domain) | **-0.1496** |
| 4 | "Xử lý vi phạm trong Đảng" vs "Kỷ luật đảng viên vi phạm tư tưởng" | **Cao** (paraphrase) | **-0.0215** |
| 5 | "Cơ quan nhà nước dùng AI" vs "Hạ tầng kho bạc nhà nước" | **Thấp** (khác domain) | **0.1344** |

**Phân tích:**
- Cặp 3 đúng hướng (âm = không tương đồng) — tuy nhiên cặp 4 cùng chủ đề lại âm, hoàn toàn ngược với dự đoán
- Mock embedder dùng MD5 hash → vector hoàn toàn ngẫu nhiên, không capture ngữ nghĩa
- **Điều bất ngờ nhất:** Cặp 2 (paraphrase rõ ràng về giáo viên hạng I) chỉ đạt 0.018, gần bằng 0 — tệ hơn cả kỳ vọng cho hai câu hoàn toàn không liên quan. Điều này chứng minh mock embedder không thể dùng để benchmark retrieval thực tế.

---

## Section 6 — Results

### Benchmark Queries & Gold Answers

| # | Query | Gold Answer |
|---|---|---|
| 1 | Tiêu chuẩn về trình độ đào tạo, bồi dưỡng đối với giáo viên hạng I | Có bằng thạc sĩ trở lên; có chứng chỉ nghiệp vụ sư phạm (THPT); có chứng chỉ bồi dưỡng chuẩn nghề nghiệp giáo viên GDTX. |
| 2 | Xử lý vi phạm QUY ĐỊNH VỀ CÔNG TÁC CHÍNH TRỊ, TƯ TƯỞNG TRONG ĐẢNG? | Bị xem xét kỷ luật theo quy định Đảng và pháp luật; người đứng đầu để xảy ra vi phạm nghiêm trọng kéo dài phải chịu trách nhiệm liên đới. |
| 3 | Khi hệ thống AI bị tấn công đầu độc dữ liệu, cần làm gì? | Áp dụng biện pháp phòng ngừa, phát hiện, ngăn chặn, ứng phó với đầu độc dữ liệu/mô hình, tấn công đối nghịch, rò rỉ dữ liệu; bảo đảm bí mật, toàn vẹn, sẵn sàng của dữ liệu và hạ tầng. |
| 4 | Những hành vi nào bị nghiêm cấm trong hoạt động AI (Luật AI 2025)? | Lợi dụng AI vi phạm pháp luật; dùng yếu tố giả mạo lừa dối; lợi dụng điểm yếu nhóm dễ bị tổn thương; tạo nội dung giả mạo gây hại an ninh quốc gia; thu thập dữ liệu trái phép. |
| 5 | Theo NĐ 129/2026, dự toán ngân sách cơ quan VN ở nước ngoài lập bằng đồng tiền gì? | Lập bằng đồng Việt Nam quy đổi ra đô la Mỹ theo tỷ giá hạch toán tháng 6 năm hiện hành do Bộ Tài chính quy định. |
| 6 | Cơ quan nhà nước dùng AI có được để hệ thống tự ra quyết định cuối không? | Không — quyết định cuối cùng thuộc thẩm quyền của con người; hệ thống AI không thay thế trách nhiệm người ra quyết định. |

---

### Kết quả chạy Benchmark

**LegalArticleChunker** (143 chunks tổng, avg_len ~990 chars):

| Q | Rank 1 | Score | Rank 2 | Score | Rank 3 | Score | Hit? |
|---|---|---|---|---|---|---|---|
| Q1 | luat-AI | 0.307 | luat-AI | 0.306 | quy-dinh-dang | 0.252 | ❌ |
| Q2 | luat-AI | 0.318 | nghi-dinh-129 | 0.292 | luat-AI | 0.247 | ❌ |
| Q3 | luat-AI ✓ | 0.279 | thong-tu-23 | 0.266 | quy-dinh-dang | 0.244 | ⚠️ |
| Q4 | quy-dinh-dang | 0.409 | quy-dinh-dang | 0.409 | luat-AI ✓ | 0.286 | ⚠️ |
| Q5 | luat-AI | 0.315 | luat-AI | 0.255 | nghi-dinh-129 ✓ | 0.228 | ⚠️ |
| Q6 | nghi-dinh-129 | 0.368 | luat-AI ✓ | 0.276 | luat-AI ✓ | 0.274 | ⚠️ |

**RecursiveChunker baseline** (664 chunks tổng, avg_len ~226 chars):

| Q | Rank 1 | Score | Rank 2 | Score | Rank 3 | Score | Hit? |
|---|---|---|---|---|---|---|---|
| Q1 | luat-AI | 0.364 | quy-dinh-dang | 0.304 | quy-dinh-dang | 0.304 | ❌ |
| Q2 | luat-AI | 0.370 | luat-AI | 0.335 | quy-dinh-dang ✓ | 0.335 | ⚠️ |
| Q3 | quy-dinh-dang | 0.384 | luat-AI ✓ | 0.373 | luat-AI ✓ | 0.373 | ⚠️ |
| Q4 | luat-AI ✓ | 0.392 | thong-tu-23 | 0.360 | thong-tu-23 | 0.351 | ✅ |
| Q5 | nghi-dinh-129 ✓ | 0.341 | thong-tu-23 | 0.337 | luat-AI | 0.333 | ✅ |
| Q6 | thong-tu-23 | 0.374 | thong-tu-23 | 0.367 | luat-AI ✓ | 0.345 | ⚠️ |

**Tổng kết:**

| | LegalArticleChunker | RecursiveChunker |
|---|---|---|
| ✅ Hit (rank 1) | 0/6 | 2/6 |
| ⚠️ Partial (rank 2-3) | 4/6 | 3/6 |
| ❌ Miss | 2/6 | 1/6 |

### Metadata Filter Test

Query: *"tiêu chuẩn trình độ giáo viên"* + `metadata_filter={"category": "thong-tu"}`

```
[1] score=0.2799 | thong-tu-23 | ...phương pháp dạy học đa dạng...
[2] score=0.2199 | thong-tu-23 | ...hệ số lương viên chức loại A1...
[3] score=0.1815 | thong-tu-23 | ...bồi dưỡng học viên kết quả học tập...
```

→ Filter hoạt động đúng: 100% kết quả từ đúng tài liệu. Tuy nhiên chunk cụ thể về "hạng I" chưa xuất hiện top-3 do mock embedder limitation.

---

## Section 7 — What I Learned

### Failure Analysis

**Failure case chính: Cả hai strategy đều fail Q1 — "Giáo viên hạng I"**

- **Query:** "Tiêu chuẩn về trình độ đào tạo, bồi dưỡng đối với giáo viên hạng I"
- **Expected source:** `thong-tu-23-2026-tt-bgddt...md`
- **Thực tế:** Cả hai strategy trả về chunks từ Luật AI và Quy định Đảng

**Nguyên nhân:**

1. **Mock embedder không phải semantic embedder** — Hash MD5 tạo vector ngẫu nhiên, hoàn toàn không phản ánh nghĩa. Từ "giáo viên", "hạng I", "đào tạo" không kéo vector về phía tài liệu giáo dục.

2. **Vocabulary mismatch** — Query dùng "đào tạo, bồi dưỡng" nhưng tài liệu dùng "trình độ chuyên môn", "chứng chỉ bồi dưỡng chuẩn nghề nghiệp". Ngay cả semantic embedder cũng cần query expansion.

3. **Chunk too large (LegalArticle) vs too small (Recursive)** — LegalArticle chunk ~1251 chars/chunk cho thong-tu-23 dẫn đến vector bị "pha loãng" bởi nhiều khái niệm trong 1 chunk. Recursive chunk ~220 chars tạo ra vector focused hơn nhưng lại miss context liên khoản.

**Đề xuất cải thiện:**
- Dùng **semantic embedder thật** (`paraphrase-multilingual-MiniLM-L12-v2` cho tiếng Việt)
- **Metadata filter trước**: `category=thong-tu` → giảm candidate pool từ 143 xuống 24 chunks
- **Query expansion**: "giáo viên hạng I" → tự thêm "V.07.05.16" (mã số theo Điều 3 Thông tư)

---

### Bài học tổng hợp

**1. Mock embedder không đủ để đánh giá retrieval**

Với MD5 hash embedding, kết quả benchmark gần như ngẫu nhiên. RecursiveChunker thắng không phải vì chunking tốt hơn mà vì 664 chunks > 143 chunks → xác suất may mắn cao hơn. Bài học: *phải test với real semantic embedder trước khi kết luận về chunking strategy*.

**2. LegalArticleChunker đúng hướng, cần đúng embedder**

Thiết kế mỗi Điều = 1 chunk là hợp lý cho văn bản pháp lý. Với semantic embedder, kỳ vọng precision sẽ tăng rõ rệt vì mỗi chunk mang trọn 1 ý pháp lý. Nhược điểm hiện tại: Điều quá dài (>1500 chars) phải fallback sang Recursive, có thể làm mất ranh giới ý nghĩa.

**3. Metadata filtering bù đắp được phần lớn embedding kém**

Ngay cả với mock embedder, `search_with_filter(category="thong-tu")` cho 100% results từ đúng tài liệu. Trong hệ thống multi-domain thực tế, thiết kế metadata schema tốt là yếu tố quan trọng không kém chất lượng embedding.

**4. Chunk coherence vs Chunk count — trade-off thực sự**

Nhiều chunks nhỏ → noise cao, mỗi chunk thiếu context đầy đủ. Ít chunks lớn → precision tốt hơn về semantic nhưng khó match hơn với embedding yếu. Điểm tối ưu phụ thuộc vào embedder và domain — không có "best" cố định.

**5. Domain-specific strategy cần domain-specific evaluation**

Văn bản pháp luật tiếng Việt có đặc thù: câu dài, từ pháp lý chuyên ngành, cấu trúc Chương-Điều-Khoản. Model embedding tiếng Anh (`all-MiniLM-L6-v2`) có thể không xử lý tốt. Lý tưởng nên dùng `paraphrase-multilingual-MiniLM-L12-v2` hoặc model đã fine-tune trên văn bản pháp lý tiếng Việt.
