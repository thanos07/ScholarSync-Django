from apps.retrieval.lexical import bm25_search
from .generator import generate_answer

def answer_workspace_question(question, chunks):
    hits = bm25_search(question, list(chunks), limit=6)
    answer, confidence, model = generate_answer(question, hits)
    return {"answer": answer, "confidence": confidence, "model": model, "hits": hits}
