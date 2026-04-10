# Giải Thích Code — Day 7: Data Foundations (Phase 1)

> Tài liệu này giải thích chi tiết logic, đầu vào, đầu ra của toàn bộ code đã implement trong Phase 1.

---

## Tổng Quan Kiến Trúc

```
src/
├── models.py       ← Document dataclass (có sẵn)
├── embeddings.py   ← MockEmbedder, LocalEmbedder, OpenAIEmbedder (có sẵn)
├── chunking.py     ← Chunking + Similarity (đã implement)
├── store.py        ← EmbeddingStore / Vector Store (đã implement)
└── agent.py        ← KnowledgeBaseAgent / RAG (đã implement)
```

### Luồng dữ liệu tổng thể

```
Tài liệu thô (str)
      │
      ▼
 [Chunker] ──────────────────► danh sách các đoạn văn (list[str])
      │
      ▼
 [Embedder] ─────────────────► vector số thực (list[float])
      │
      ▼
 [EmbeddingStore] ───────────► lưu {id, content, embedding, metadata}
      │
      ▼ (khi có query)
 [Search] ────────────────────► top-k kết quả theo cosine/dot-product
      │
      ▼
 [KnowledgeBaseAgent] ────────► prompt → LLM → câu trả lời
```

---

## 1. `src/models.py` — Document Dataclass

> File này đã có sẵn, đây là nền tảng cho toàn bộ hệ thống.

```python
@dataclass
class Document:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
```

| Trường | Kiểu | Ý nghĩa |
|--------|------|---------|
| `id` | `str` | Định danh duy nhất của tài liệu (vd: `"doc1"`, `"python_intro"`) |
| `content` | `str` | Nội dung văn bản thô |
| `metadata` | `dict` | Thông tin phụ tùy ý (vd: `{"source": "file.txt", "lang": "vi"}`) |

---

## 2. `src/chunking.py` — Chia Văn Bản & Tính Tương Đồng

### 2.1 `FixedSizeChunker` (có sẵn — tham khảo)

**Ý tưởng:** Cắt văn bản thành các đoạn có độ dài cố định, với phần chồng lấp (overlap) giữa các đoạn liền kề để không mất ngữ cảnh tại ranh giới.

```python
FixedSizeChunker(chunk_size=500, overlap=50)
```

**Đầu vào:** `text: str`

**Đầu ra:** `list[str]` — danh sách các đoạn văn

**Logic từng bước:**

```
text = "ABCDEFGHIJ..."  (dài hơn chunk_size)

step = chunk_size - overlap  (bước nhảy thực sự)

Lần 1: text[0 : chunk_size]          → chunk 1
Lần 2: text[step : step+chunk_size]  → chunk 2  (chồng overlap ký tự với chunk 1)
Lần 3: text[2*step : 2*step+chunk_size] → chunk 3
...
```

**Ví dụ cụ thể:**

```python
chunker = FixedSizeChunker(chunk_size=10, overlap=2)
result  = chunker.chunk("abcdefghijklmnopqrst")
# step = 10 - 2 = 8
# chunk 1: text[0:10]  = "abcdefghij"
# chunk 2: text[8:18]  = "ijklmnopqr"   ← "ij" là phần chồng lấp
# chunk 3: text[16:26] = "qrst"
```

**Khi nào nên dùng:** Văn bản dài không có cấu trúc rõ ràng; cần chunks có kích thước đồng đều.

---

### 2.2 `SentenceChunker` (đã implement)

**Ý tưởng:** Tôn trọng ranh giới câu — không cắt giữa chừng một câu. Nhóm các câu lại thành chunks.

```python
SentenceChunker(max_sentences_per_chunk=3)
```

**Đầu vào:** `text: str`

**Đầu ra:** `list[str]` — mỗi phần tử là 1 đến `max_sentences_per_chunk` câu ghép lại

**Logic từng bước:**

```
Bước 1 — Tách câu:
  "Câu A. Câu B! Câu C? Câu D."
       ↓  regex: (?<=[.!?])\s+
  ["Câu A.", "Câu B!", "Câu C?", "Câu D."]

Bước 2 — Nhóm theo max_sentences_per_chunk=2:
  Nhóm 1: ["Câu A.", "Câu B!"]  → "Câu A. Câu B!"
  Nhóm 2: ["Câu C?", "Câu D."] → "Câu C? Câu D."
```

**Chi tiết Regex:**

```
(?<=[.!?])\s+
     │          │
     │          └── khớp với 1+ ký tự khoảng trắng (space, \n...)
     └── lookbehind: chỉ khớp nếu NGAY TRƯỚC là . hoặc ! hoặc ?
```

Regex này **không xóa dấu câu** — nó chỉ tách tại khoảng trắng sau dấu câu, giữ nguyên `.!?` ở cuối câu.

**Ví dụ cụ thể:**

```python
chunker = SentenceChunker(max_sentences_per_chunk=2)
text = "Python is easy. It is powerful! Use it well."
result = chunker.chunk(text)
# → ["Python is easy. It is powerful!", "Use it well."]
```

**Khi nào nên dùng:** Văn bản hội thoại, FAQ, tài liệu viết theo câu — nơi ý nghĩa nằm trong từng câu hoàn chỉnh.

---

### 2.3 `RecursiveChunker` (đã implement)

**Ý tưởng:** Thử chia bằng separator "thô" trước (paragraph), nếu vẫn còn đoạn quá dài thì dùng separator "mịn" hơn (câu → từ → ký tự). Sau đó gộp các mảnh nhỏ lại để chunks có kích thước hợp lý.

```python
RecursiveChunker(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=500
)
```

**Đầu vào:** `text: str`

**Đầu ra:** `list[str]`

**Logic từng bước:**

```
Bước 1 — chunk() gọi _split(text, ["\n\n", "\n", ". ", " ", ""])

Bước 2 — _split() kiểm tra:
  ├── Nếu len(text) <= chunk_size → trả về [text]  (base case)
  ├── Nếu không còn separator    → hard-cut bằng chunk_size
  └── Nếu còn separator → dùng separator[0] để split

Bước 3 — Với separator "\n\n", tách thành các paragraph:
  [para1, para2, para3, ...]

Bước 4 — Với mỗi paragraph:
  ├── Nếu len(para) <= chunk_size → giữ nguyên
  └── Nếu quá dài → đệ quy _split(para, ["\n", ". ", " ", ""])

Bước 5 — Merge: gộp các đoạn nhỏ lại miễn là tổng <= chunk_size
  [A, B, C, D] → [A+"\n\n"+B, C+"\n\n"+D]  nếu A+B vừa chunk_size
```

**Ví dụ cụ thể (chunk_size=50):**

```
Input:
  "Python is great.\n\nIt handles text well.\n\nUse it for AI."

separator[0] = "\n\n" → split thành:
  ["Python is great.", "It handles text well.", "Use it for AI."]

Mỗi mảnh đều <= 50 → giữ nguyên

Merge:
  "Python is great." + "\n\n" + "It handles text well." = 40 chars ≤ 50 ✓
  → "Python is great.\n\nIt handles text well." + "Use it for AI." = 57 > 50 ✗
  → Kết quả: ["Python is great.\n\nIt handles text well.", "Use it for AI."]
```

**Cây đệ quy minh họa:**

```
_split("rất dài...", ["\n\n", "\n", ". ", " ", ""])
 ├── Tách theo "\n\n"
 │    ├── para1 (ngắn) ✓
 │    ├── para2 (dài) → _split(para2, ["\n", ". ", " ", ""])
 │    │                   ├── line1 (ngắn) ✓
 │    │                   └── line2 (dài) → _split(line2, [". ", " ", ""])
 │    │                                        └── ...
 │    └── para3 (ngắn) ✓
 └── Merge tất cả kết quả
```

**Khi nào nên dùng:** Tài liệu có cấu trúc phân cấp (Markdown, tài liệu kỹ thuật, sách) — chunker tự động tôn trọng cấu trúc đó.

---

### 2.4 `compute_similarity()` (đã implement)

**Ý tưởng:** Đo mức độ "hướng giống nhau" của hai vector trong không gian nhiều chiều. Giá trị nằm trong `[-1, 1]`.

```python
compute_similarity(vec_a: list[float], vec_b: list[float]) -> float
```

**Đầu vào:** 2 vector số thực cùng chiều

**Đầu ra:** `float` trong khoảng `[-1.0, 1.0]`

**Công thức:**

```
                  a · b
cos(θ) = ──────────────────
           ‖a‖ × ‖b‖

Trong đó:
  a · b  = Σ(a_i × b_i)          ← tích vô hướng (dot product)
  ‖a‖    = √(Σ a_i²)             ← độ dài vector a (magnitude)
  ‖b‖    = √(Σ b_i²)             ← độ dài vector b
```

**Triển khai từng bước:**

```python
def compute_similarity(vec_a, vec_b):
    dot_product = sum(x * y for x, y in zip(vec_a, vec_b))  # a · b
    magnitude_a = math.sqrt(sum(x * x for x in vec_a))      # ‖a‖
    magnitude_b = math.sqrt(sum(x * x for x in vec_b))      # ‖b‖

    if magnitude_a == 0.0 or magnitude_b == 0.0:            # tránh chia 0
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)
```

**Ý nghĩa của giá trị trả về:**

| Giá trị | Ý nghĩa |
|---------|---------|
| `1.0` | Hai vector cùng hướng hoàn toàn (nội dung giống hệt) |
| `0.0` | Vuông góc — không có sự tương đồng về hướng |
| `-1.0` | Ngược hướng hoàn toàn |

**Tại sao dùng cosine thay vì Euclidean distance?**
- Cosine đo **hướng**, không đo **độ lớn** → câu ngắn và câu dài nói cùng ý vẫn có similarity cao
- Euclidean sẽ cho khoảng cách lớn giữa vector ngắn và dài, dù nội dung giống nhau

**Ví dụ:**

```python
v1 = [1.0, 0.0, 0.0]
v2 = [1.0, 0.0, 0.0]
compute_similarity(v1, v2)  # → 1.0 (giống hệt)

v3 = [0.0, 1.0, 0.0]
compute_similarity(v1, v3)  # → 0.0 (vuông góc)

v4 = [-1.0, 0.0, 0.0]
compute_similarity(v1, v4)  # → -1.0 (ngược hướng)

v5 = [0.0, 0.0, 0.0]
compute_similarity(v1, v5)  # → 0.0 (zero-magnitude guard)
```

---

### 2.5 `ChunkingStrategyComparator` (đã implement)

**Ý tưởng:** Chạy cả 3 strategy trên cùng một text để so sánh khách quan.

```python
ChunkingStrategyComparator().compare(text, chunk_size=200)
```

**Đầu vào:**
- `text: str` — văn bản cần so sánh
- `chunk_size: int` — kích thước chunk mục tiêu (mặc định 200)

**Đầu ra:** `dict` với 3 key là tên strategy:

```python
{
    "fixed_size": {
        "count": 12,        # số lượng chunks
        "avg_length": 185.3, # độ dài trung bình mỗi chunk (ký tự)
        "chunks": [...]     # danh sách các chunk thực tế
    },
    "by_sentences": { ... },
    "recursive":    { ... }
}
```

**Logic:**

```python
overlap = chunk_size // 10   # 10% overlap cho FixedSizeChunker

strategies = {
    "fixed_size":   FixedSizeChunker(chunk_size, overlap).chunk(text),
    "by_sentences": SentenceChunker(max_sentences_per_chunk=3).chunk(text),
    "recursive":    RecursiveChunker(chunk_size=chunk_size).chunk(text),
}

for name, chunks in strategies.items():
    count      = len(chunks)
    avg_length = total_chars / count
    result[name] = {"count": count, "avg_length": avg_length, "chunks": chunks}
```

---

## 3. `src/store.py` — EmbeddingStore (Vector Store)

**Ý tưởng tổng thể:** Lưu trữ các documents dưới dạng vector (embedding). Khi có query, chuyển query thành vector rồi tìm các vectors "gần nhất" trong store.

### 3.1 `__init__()` — Khởi tạo

```python
EmbeddingStore(
    collection_name="documents",
    embedding_fn=None   # mặc định dùng _mock_embed
)
```

**Các thuộc tính được khởi tạo:**

| Thuộc tính | Kiểu | Ý nghĩa |
|------------|------|---------|
| `_embedding_fn` | `Callable` | Hàm chuyển text → vector |
| `_collection_name` | `str` | Tên collection (dùng nếu kết nối ChromaDB) |
| `_use_chroma` | `bool` | Có dùng ChromaDB hay không (hiện luôn `False`) |
| `_store` | `list[dict]` | Danh sách records lưu trong bộ nhớ |
| `_next_index` | `int` | Bộ đếm index nội bộ |

**Về ChromaDB:** Code thử import `chromadb`, nhưng trong lab này vẫn giữ `_use_chroma = False` để hành vi nhất quán trên mọi môi trường. Để thật sự dùng ChromaDB, cần cài `pip install chromadb` và chỉnh code.

---

### 3.2 `_make_record()` — Tạo Record

```python
_make_record(doc: Document) -> dict[str, Any]
```

**Đầu vào:** 1 `Document`

**Đầu ra:** `dict` đã chuẩn hóa:

```python
{
    "id":        "python_intro",        # doc.id
    "content":   "Python is a...",      # doc.content
    "embedding": [0.12, -0.34, ...],    # vector 64 chiều (mock)
    "metadata":  {"source": "file.txt"} # doc.metadata (copy)
}
```

**Lưu ý:** `embedding_fn` được gọi tại đây — đây là bước tốn kém nhất (gọi API hoặc chạy model).

---

### 3.3 `add_documents()` — Thêm Documents

```python
add_documents(docs: list[Document]) -> None
```

**Đầu vào:** danh sách `Document`

**Đầu ra:** `None` (thay đổi `_store` in-place)

**Logic:**

```
For mỗi doc trong docs:
  record = _make_record(doc)   ← gọi embedding_fn
  _store.append(record)
  _next_index += 1
```

**Ví dụ:**

```python
store = EmbeddingStore(embedding_fn=_mock_embed)
store.add_documents([
    Document("d1", "Python is great", {"lang": "en"}),
    Document("d2", "Machine learning is AI", {"lang": "en"}),
])
# _store giờ có 2 records
```

---

### 3.4 `_search_records()` — Tìm Kiếm Trong Memory

```python
_search_records(
    query: str,
    records: list[dict[str, Any]],
    top_k: int
) -> list[dict[str, Any]]
```

**Đầu vào:**
- `query` — câu hỏi/từ khóa tìm kiếm
- `records` — danh sách records cần tìm (có thể là toàn bộ `_store` hoặc đã filter)
- `top_k` — số kết quả muốn lấy

**Đầu ra:** `list[dict]` top-k kết quả, sắp xếp theo `score` giảm dần:

```python
[
    {"id": "d2", "content": "...", "metadata": {...}, "score": 0.87},
    {"id": "d1", "content": "...", "metadata": {...}, "score": 0.43},
]
```

**Logic:**

```
1. query_embedding = embedding_fn(query)    ← embed câu hỏi

2. For mỗi record trong records:
       score = dot_product(query_embedding, record["embedding"])
       scored.append({...record, "score": score})

3. Sắp xếp scored theo score giảm dần

4. Trả về scored[:top_k]
```

**Tại sao dùng dot product (không phải cosine)?**

Vì `MockEmbedder` và `LocalEmbedder` đều trả về vector đã **normalize** (độ dài = 1), nên:

```
cosine(a, b) = a·b / (‖a‖ × ‖b‖) = a·b / (1 × 1) = a·b
```

Với unit vector, dot product **tương đương** cosine similarity.

---

### 3.5 `search()` — Tìm Kiếm Toàn Bộ Store

```python
search(query: str, top_k: int = 5) -> list[dict[str, Any]]
```

**Đầu vào:** `query` (str), `top_k` (int)

**Đầu ra:** top-k records phù hợp nhất

**Logic:** Gọi `_search_records(query, self._store, top_k)` — tìm trên toàn bộ store.

---

### 3.6 `get_collection_size()` — Đếm Records

```python
get_collection_size() -> int
```

**Đầu ra:** số lượng records hiện có trong store.

```python
store.get_collection_size()  # → 0 ban đầu
store.add_documents([doc1, doc2, doc3])
store.get_collection_size()  # → 3
```

---

### 3.7 `search_with_filter()` — Tìm Kiếm Với Bộ Lọc Metadata

```python
search_with_filter(
    query: str,
    top_k: int = 3,
    metadata_filter: dict = None
) -> list[dict]
```

**Đầu vào:**
- `query` — câu hỏi
- `top_k` — số kết quả
- `metadata_filter` — dict điều kiện lọc, vd: `{"department": "engineering", "lang": "vi"}`

**Đầu ra:** top-k records phù hợp (trong tập đã lọc)

**Logic 2 bước:**

```
Bước 1 — Filter (nếu có metadata_filter):
  filtered = [record for record in _store
              if record["metadata"]["department"] == "engineering"
              AND record["metadata"]["lang"] == "vi"]

Bước 2 — Search trong tập đã lọc:
  return _search_records(query, filtered, top_k)
```

**Ví dụ:**

```python
store.add_documents([
    Document("d1", "Python tutorial", {"department": "engineering"}),
    Document("d2", "Marketing plan",  {"department": "marketing"}),
    Document("d3", "Python advanced", {"department": "engineering"}),
])

# Chỉ tìm trong engineering
results = store.search_with_filter(
    "Python",
    top_k=5,
    metadata_filter={"department": "engineering"}
)
# → chỉ trả về d1 và d3 (d2 bị loại vì department khác)
```

---

### 3.8 `delete_document()` — Xóa Document

```python
delete_document(doc_id: str) -> bool
```

**Đầu vào:** `doc_id` — id của document cần xóa

**Đầu ra:** `True` nếu có record bị xóa, `False` nếu không tìm thấy

**Logic:**

```python
size_before = len(self._store)

self._store = [
    record for record in self._store
    if record["id"] != doc_id               # xóa theo id trực tiếp
    and record["metadata"].get("doc_id") != doc_id  # hoặc theo metadata
]

return len(self._store) < size_before  # True nếu đã xóa ít nhất 1
```

**Ví dụ:**

```python
store.add_documents([Document("d1", "..."), Document("d2", "...")])
store.delete_document("d1")  # → True, _store còn 1 record
store.delete_document("d1")  # → False, không còn gì để xóa
```

---

## 4. `src/agent.py` — KnowledgeBaseAgent (RAG Pattern)

**Ý tưởng:** RAG (Retrieval-Augmented Generation) — thay vì để LLM trả lời từ "trí nhớ" của nó, agent lấy thông tin liên quan từ store rồi đưa vào prompt. LLM chỉ cần tổng hợp thông tin đã cho.

### 4.1 `__init__()` — Khởi tạo

```python
KnowledgeBaseAgent(
    store: EmbeddingStore,    # vector store đã có dữ liệu
    llm_fn: Callable[[str], str]  # hàm gọi LLM: prompt → answer
)
```

Chỉ lưu tham chiếu, không làm gì thêm:

```python
self.store  = store
self.llm_fn = llm_fn
```

---

### 4.2 `answer()` — Trả Lời Câu Hỏi

```python
answer(question: str, top_k: int = 3) -> str
```

**Đầu vào:**
- `question` — câu hỏi của người dùng
- `top_k` — số chunks context lấy từ store

**Đầu ra:** `str` — câu trả lời từ LLM

**Logic 3 bước (RAG Pattern):**

```
Bước 1 — RETRIEVE: tìm top-k chunks liên quan
  results = store.search(question, top_k=top_k)

Bước 2 — BUILD PROMPT: đưa context vào prompt
  prompt = """
  You are a helpful assistant. Use the context below...
  
  Context:
  [1] (score=0.87) Python is a high-level programming language...
  [2] (score=0.65) Machine learning uses algorithms to learn...
  [3] (score=0.42) Vector databases store embeddings for...
  
  Question: What is Python?
  
  Answer:
  """

Bước 3 — GENERATE: gọi LLM
  return llm_fn(prompt)
```

**Cấu trúc Prompt chi tiết:**

```
"You are a helpful assistant. Use the context below to answer the question.\n"
"If the context does not contain the answer, say you do not know.\n\n"
"Context:\n"
"[1] (score=0.870) <nội dung chunk 1>\n\n"
"[2] (score=0.650) <nội dung chunk 2>\n\n"
"[3] (score=0.420) <nội dung chunk 3>\n\n"
"Question: <câu hỏi>\n\n"
"Answer:"
```

**Ví dụ đầy đủ:**

```python
# Chuẩn bị store
store = EmbeddingStore(embedding_fn=_mock_embed)
store.add_documents([
    Document("d1", "Python is a high-level programming language.", {}),
    Document("d2", "Machine learning uses algorithms.", {}),
    Document("d3", "Vector databases store embeddings.", {}),
])

# Tạo agent với mock LLM
def my_llm(prompt: str) -> str:
    return f"Based on context: {prompt[:100]}..."

agent = KnowledgeBaseAgent(store=store, llm_fn=my_llm)

# Hỏi
answer = agent.answer("What is Python?", top_k=2)
print(answer)
# → "Based on context: You are a helpful assistant..."
```

---

## 5. Tóm Tắt Nhanh — Bảng So Sánh 3 Chunking Strategy

| Tiêu chí | FixedSizeChunker | SentenceChunker | RecursiveChunker |
|----------|-----------------|-----------------|-----------------|
| **Đơn vị chia** | Ký tự | Câu | Paragraph → câu → từ |
| **Kích thước chunk** | Chính xác (±overlap) | Biến đổi theo câu | Gần với chunk_size |
| **Tôn trọng cấu trúc** | Không | Câu | Paragraph + câu + từ |
| **Overlap** | Có | Không | Không |
| **Phù hợp với** | Code, log, văn bản không cấu trúc | Hội thoại, FAQ | Markdown, tài liệu có cấu trúc |
| **Nhược điểm** | Có thể cắt giữa câu | Chunks có thể rất ngắn | Phức tạp hơn |

---

## 6. Sơ Đồ Tương Tác Giữa Các Class

```
Document("d1", "Python is...", {"lang": "en"})
     │
     ▼
EmbeddingStore.add_documents([doc1, doc2, doc3])
     │
     ├── _make_record(doc1)
     │     └── embedding_fn("Python is...") → [0.12, -0.34, 0.89, ...]
     │         └── lưu vào _store: {"id": "d1", "content": "...", "embedding": [...], "metadata": {...}}
     │
     ├── _make_record(doc2) → lưu vào _store
     └── _make_record(doc3) → lưu vào _store

                          ┌─────────────────────────────────────────┐
KnowledgeBaseAgent.answer("What is Python?")                        │
     │                                                               │
     ├── store.search("What is Python?", top_k=3)                   │
     │     ├── embedding_fn("What is Python?") → query_vector       │
     │     └── _search_records():                                    │
     │           dot_product(query_vector, record["embedding"])      │
     │           sort by score → return top-3                        │
     │                                                               │
     ├── build prompt với context từ top-3                           │
     └── llm_fn(prompt) → "Python is a high-level language..."      │
                                                                     │
                          └─────────────────────────────────────────┘
```

---

## 7. Câu Hỏi Thường Gặp

**Q: Tại sao `SentenceChunker` không dùng `overlap`?**

A: Overlap theo ký tự sẽ phá vỡ ranh giới câu. Thay vào đó, nếu cần context liên tục, có thể implement "sliding window" theo câu (câu cuối của chunk trước trở thành câu đầu của chunk sau).

**Q: `RecursiveChunker` với `separators=[]` sẽ làm gì?**

A: Không có separator nào → `remaining_separators` là rỗng từ đầu → hard-cut theo `chunk_size`. Nếu text ngắn hơn `chunk_size` thì trả về `[text]`.

**Q: Tại sao `EmbeddingStore` không thật sự dùng ChromaDB?**

A: Để đảm bảo tất cả học viên có môi trường giống nhau. ChromaDB là optional — cần cài thêm và uncomment code nếu muốn thử.

**Q: `compute_similarity` và `_search_records` dùng `dot product` — có phải cosine không?**

A: Với vector **đã normalize** (unit vector, ‖v‖ = 1), dot product = cosine similarity. `MockEmbedder` và `LocalEmbedder` đều normalize output, nên kết quả tương đương.
