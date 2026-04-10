from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)

        if results:
            context_blocks = []
            for index, result in enumerate(results, start=1):
                content = result.get("content", "")
                score = result.get("score", 0.0)
                context_blocks.append(f"[{index}] (score={score:.3f}) {content}")
            context = "\n\n".join(context_blocks)
        else:
            context = "(no relevant context found)"

        prompt = (
            "You are a helpful assistant. Use the context below to answer the question.\n"
            "If the context does not contain the answer, say you do not know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        return self.llm_fn(prompt)
