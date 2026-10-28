INSTRUCTIONS = (
    "You are an assistant for Customer Success support representatives. "
    "Use only the approved context below to answer the question. "
    "Do not invent policies, numbers, timelines, or other details that are not "
    "supported by the approved context. If the context does not fully answer the "
    "question, say what information is missing."
)

RESPONSE_REQUIREMENTS = (
    "- Answer concisely in plain language a support representative can act on.\n"
    "- Base every statement on the approved context.\n"
    "- If information is missing or incomplete, say so clearly.\n"
    "- Cite source IDs (for example, [SUB-101]) when helpful."
)


def format_context_chunk(chunk):
    """Format one retrieved chunk for the prompt context block."""
    header = (
        f"[{chunk.get('source_id', 'unknown')}] "
        f"{chunk.get('title', 'Untitled')} | "
        f"Category: {chunk.get('category', 'N/A')} | "
        f"Section: {chunk.get('section', 'N/A')}"
    )
    return f"{header}\n{chunk.get('text', '').strip()}"


def build_rag_prompt(question, context_chunks):
    """Build a structured RAG prompt from a question and retrieved context."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be blank.")

    usable_chunks = [
        chunk
        for chunk in (context_chunks or [])
        if isinstance(chunk, dict)
        and isinstance(chunk.get("text"), str)
        and chunk["text"].strip()
    ]
    if not usable_chunks:
        raise ValueError("Context must include at least one chunk with text.")

    context_block = "\n\n".join(format_context_chunk(chunk) for chunk in usable_chunks)

    return (
        f"Instructions:\n{INSTRUCTIONS}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Response Requirements:\n{RESPONSE_REQUIREMENTS}\n"
    )
