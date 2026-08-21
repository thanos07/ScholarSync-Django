import json
import re
from dataclasses import dataclass
from functools import lru_cache

from django.conf import settings

from apps.rag.generator import generate_answer
from apps.retrieval.lexical import SearchHit, bm25_search, tokens


@dataclass(frozen=True)
class DemoDocument:
    id: str
    title: str
    filename: str
    pages: int
    description: str
    size_bytes: int | None = None
    sha256: str | None = None
    chunks: int | None = None


@dataclass
class DemoChunk:
    id: str
    document_id: str
    filename: str
    source: str
    page: int
    content: str


@dataclass(frozen=True)
class RetrievalPlan:
    intent: str
    per_document_queries: dict[str, tuple[str, ...]]
    global_queries: tuple[str, ...] = ()
    per_document_limit: int = 3
    total_limit: int = 9


@lru_cache(maxsize=1)
def load_demo_documents():
    payload = json.loads(
        (settings.DEMO_DATA_DIR / "documents.json").read_text(encoding="utf-8")
    )
    return [DemoDocument(**row) for row in payload]


@lru_cache(maxsize=1)
def load_demo_chunks():
    payload = json.loads(
        (settings.DEMO_DATA_DIR / "chunks.json").read_text(encoding="utf-8")
    )
    return [DemoChunk(**row) for row in payload]


def get_demo_document(document_id):
    return next((doc for doc in load_demo_documents() if doc.id == document_id), None)


def _find_chunk(document_id, page, contains=None):
    matches = [
        chunk for chunk in load_demo_chunks()
        if chunk.document_id == document_id and chunk.page == page
    ]
    if contains:
        needle = contains.lower()
        for chunk in matches:
            if needle in chunk.content.lower():
                return chunk
    return matches[0] if matches else None


def _normalize_question(question):
    return re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()


def _find_chunks(document_id, queries, limit=4):
    pool = [chunk for chunk in load_demo_chunks() if chunk.document_id == document_id]
    return [hit.item for hit in _fused_search(queries, pool, limit=limit)]


def _is_comparison(normalized):
    return any(
        term in normalized
        for term in (
            "compare", "comparison", "differ", "difference", "versus", " vs ",
            "all three", "three paper", "these paper", "across the paper",
        )
    )


def _retrieval_plan(question):
    normalized = f" {_normalize_question(question)} "

    # Formula/equation requests need formula-heavy pages rather than generic
    # architecture passages. In a referential follow-up such as "what formulas
    # does it use?", the preceding question is included in `question`, so an
    # earlier mention of the attention paper still routes here.
    if any(term in normalized for term in (" formula ", " formulas ", " equation ", " equations ", " mathematical ", " mathematics ", " math ")):
        return RetrievalPlan(
            intent="formula",
            per_document_queries={
                "attention-is-all-you-need": (
                    question,
                    "LayerNorm x Sublayer residual connection equation",
                    "Attention Q K V softmax Q K transpose square root d k equation",
                    "MultiHead Concat head attention projection matrices equation",
                    "FFN x max ReLU W1 b1 W2 b2 equation",
                    "positional encoding sine cosine PE pos equation",
                    "learning rate schedule d model step warmup equation",
                    "self attention recurrent convolution complexity O n d table",
                )
            },
            per_document_limit=12,
            total_limit=12,
        )

    # A request to compare global and local attention is a comparison *inside*
    # the Luong paper, not a request to compare all three papers.
    if any(term in normalized for term in ("global attention", "local attention", " luong ", "alignment function")):
        return RetrievalPlan(
            intent="recurrent-attention",
            per_document_queries={
                "effective-attention-nmt": (
                    question,
                    "global attention all source hidden states context vector alignment",
                    "local attention predicted aligned position source window differentiable",
                    "stacked LSTM encoder decoder neural machine translation",
                )
            },
            per_document_limit=8,
            total_limit=8,
        )

    if _is_comparison(normalized):
        return RetrievalPlan(
            intent="comparison",
            per_document_queries={
                "effective-attention-nmt": (
                    "stacked LSTM encoder decoder neural machine translation global local attention context vector alignment",
                    "global attention local attention source positions computational cost translation results",
                ),
                "convolutional-seq2seq": (
                    "fully convolutional sequence to sequence architecture gated linear units residual connections",
                    "multi-step attention each decoder layer parallel computation recurrent comparison translation speed",
                ),
                "attention-is-all-you-need": (
                    "Transformer architecture encoder decoder multi-head self-attention feed-forward residual layer normalization",
                    "positional encoding parallelization recurrence convolution translation BLEU training cost",
                ),
            },
            per_document_limit=3,
            total_limit=9,
        )

    if any(term in normalized for term in (" transformer ", " transformers ", "self attention", "multi head", "scaled dot product", "positional encoding")):
        return RetrievalPlan(
            intent="transformer",
            per_document_queries={
                "attention-is-all-you-need": (
                    question,
                    "Transformer model architecture encoder decoder multi-head self-attention feed-forward residual layer normalization",
                    "scaled dot-product attention queries keys values softmax",
                    "positional encoding parallelization recurrence convolution",
                )
            },
            per_document_limit=10,
            total_limit=10,
        )

    # Broad attention questions are a flagship demo path.
    # Keep this after the Transformer route so prompts such as
    # "How does multi-head self-attention work?" remain Transformer-specific.
    if any(
        term in normalized
        for term in (
            " attention mechanism ",
            " attention mechanisms ",
            " attention model ",
            " attention models ",
            " attention based ",
            " explain attention ",
            " what is attention ",
            " about attention ",
            " how does attention work ",
        )
    ):
        return RetrievalPlan(
            intent="attention-overview",
            per_document_queries={
                "effective-attention-nmt": (
                    question,
                    "attention based models encoder decoder hidden states context vector alignment",
                    "global attention local attention source hidden states context vector",
                ),
                "convolutional-seq2seq": (
                    question,
                    "separate attention mechanism decoder layer encoder representations",
                    "multiple attention convolutional sequence generation decoder",
                ),
                "attention-is-all-you-need": (
                    question,
                    "scaled dot product attention queries keys values softmax",
                    "multi head self attention encoder decoder Transformer",
                ),
            },
            per_document_limit=2,
            total_limit=6,
        )

    if any(term in normalized for term in (" cnn ", "convolution", "conv s2s", "convseq2seq", "gated linear", " glu ")):
        return RetrievalPlan(
            intent="convolutional",
            per_document_queries={
                "convolutional-seq2seq": (
                    question,
                    "fully convolutional sequence to sequence architecture gated linear units residual connections",
                    "multi-step attention separate attention each decoder layer encoder outputs",
                    "parallel computation recurrent network speed long-range dependencies",
                )
            },
            per_document_limit=8,
            total_limit=8,
        )

    if "neural network" in normalized or " nmt " in normalized or "sequence to sequence" in normalized:
        return RetrievalPlan(
            intent="neural-network",
            per_document_queries={
                "effective-attention-nmt": (
                    "neural machine translation neural network encoder decoder recurrent LSTM attention",
                ),
                "convolutional-seq2seq": (
                    "recurrent sequence to sequence fully convolutional encoder decoder neural network",
                ),
                "attention-is-all-you-need": (
                    "sequence transduction Transformer encoder decoder self-attention neural architecture",
                ),
            },
            per_document_limit=2,
            total_limit=6,
        )

    return RetrievalPlan(
        intent="general",
        per_document_queries={},
        global_queries=(question,),
        total_limit=8,
    )


CORE_PAGE_WEIGHTS = {
    ("attention-overview", "effective-attention-nmt"): {2: 1.05, 3: 1.45, 4: 1.40, 5: 1.00},
    ("attention-overview", "convolutional-seq2seq"): {1: 1.10, 2: 1.25, 3: 1.45, 4: 1.35},
    ("attention-overview", "attention-is-all-you-need"): {2: 1.20, 3: 1.30, 4: 1.50, 5: 1.40},
    ("formula", "attention-is-all-you-need"): {3: 1.20, 4: 1.65, 5: 1.60, 6: 1.55, 7: 1.55, 8: 0.82},
    ("recurrent-attention", "effective-attention-nmt"): {2: 1.10, 3: 1.35, 4: 1.45, 5: 1.05},
    ("transformer", "attention-is-all-you-need"): {1: 1.05, 2: 1.35, 3: 1.50, 4: 1.45, 5: 1.45, 6: 1.30, 8: 0.82, 9: 0.76},
    ("convolutional", "convolutional-seq2seq"): {1: 1.30, 2: 1.45, 3: 1.50, 4: 1.30, 5: 1.05, 6: 0.92, 7: 0.86, 8: 0.82, 9: 0.78},
    ("neural-network", "effective-attention-nmt"): {1: 1.15, 2: 1.45, 3: 1.25},
    ("neural-network", "convolutional-seq2seq"): {1: 1.25, 2: 1.45, 3: 1.20, 6: 0.88, 9: 0.75},
    ("neural-network", "attention-is-all-you-need"): {1: 1.20, 2: 1.50, 3: 1.35, 4: 1.10, 5: 1.10, 8: 0.78, 9: 0.72},
    ("comparison", "effective-attention-nmt"): {2: 1.20, 3: 1.45, 4: 1.40, 6: 0.95, 7: 0.90},
    ("comparison", "convolutional-seq2seq"): {1: 1.25, 2: 1.45, 3: 1.45, 4: 1.20, 6: 0.92, 7: 0.88},
    ("comparison", "attention-is-all-you-need"): {1: 1.10, 2: 1.40, 3: 1.45, 4: 1.25, 5: 1.30, 6: 1.25, 8: 0.88, 9: 0.78},
}


def _fused_search(queries, items, limit=8, page_weights=None):
    """Fuse several lexical searches by rank, keeping concise page passages."""
    items = list(items)
    by_id = {}
    for query_index, query in enumerate(queries):
        if not query or not query.strip():
            continue
        query_weight = 1.35 if query_index == 0 else 1.0
        hits = bm25_search(
            query,
            items,
            text_getter=lambda item: f"{item.source} {item.content}",
            limit=min(18, max(limit * 3, 10)),
        )
        for rank, hit in enumerate(hits, 1):
            key = hit.item.id
            # Reciprocal-rank fusion is more stable than comparing raw BM25
            # scores from queries of very different lengths.
            fused = query_weight / (22 + rank)
            fused += min(hit.score, 20.0) * 0.002
            if page_weights:
                fused *= page_weights.get(getattr(hit.item, "page", None), 1.0)
            current = by_id.get(key)
            if current is None:
                by_id[key] = SearchHit(item=hit.item, score=fused)
            else:
                current.score += fused

    ranked = sorted(by_id.values(), key=lambda hit: hit.score, reverse=True)

    # Avoid returning several near-duplicate overlapping passages from one page.
    selected = []
    seen_text = []
    for hit in ranked:
        terms = set(tokens(hit.item.content, remove_stopwords=True))
        if any(
            len(terms & prior) / max(1, len(terms | prior)) > 0.76
            for prior in seen_text
        ):
            continue
        selected.append(hit)
        seen_text.append(terms)
        if len(selected) >= limit:
            break
    return selected


def _intent_anchor_chunks(intent):
    """Pin core passages so broad queries cover every essential concept."""
    if intent == "attention-overview":
        return [
            _find_chunk("effective-attention-nmt", 3, "3 Attention-based Models"),
            _find_chunk("effective-attention-nmt", 4, "3.2 Local Attention"),
            _find_chunk("convolutional-seq2seq", 3, "separate attention mechanism"),
            _find_chunk("convolutional-seq2seq", 4, "multiple attention"),
            _find_chunk("attention-is-all-you-need", 4, "Scaled Dot-Product Attention"),
            _find_chunk("attention-is-all-you-need", 4, "Multi-head attention allows"),
        ]
    if intent == "transformer":
        return [
            _find_chunk("attention-is-all-you-need", 2, "3 Model Architecture"),
            _find_chunk("attention-is-all-you-need", 3, "Figure 1: The Transformer"),
            _find_chunk("attention-is-all-you-need", 4, "Scaled Dot-Product Attention"),
            _find_chunk("attention-is-all-you-need", 5, "3.5 Positional Encoding"),
            _find_chunk("attention-is-all-you-need", 6, "sine and cosine functions"),
            _find_chunk("attention-is-all-you-need", 6, "Maximum path lengths"),
        ]
    if intent == "formula":
        return [
            _find_chunk("attention-is-all-you-need", 3, "LayerNorm(x + Sublayer(x))"),
            _find_chunk("attention-is-all-you-need", 4, "Attention(Q, K, V )"),
            _find_chunk("attention-is-all-you-need", 5, "MultiHead(Q, K, V )"),
            _find_chunk("attention-is-all-you-need", 5, "FFN(x)"),
            _find_chunk("attention-is-all-you-need", 6, "sine and cosine functions"),
            _find_chunk("attention-is-all-you-need", 7, "lrate ="),
        ]
    return []


def _retrieve(plan):
    all_chunks = load_demo_chunks()
    selected = []

    if plan.per_document_queries:
        for document_id, queries in plan.per_document_queries.items():
            pool = [chunk for chunk in all_chunks if chunk.document_id == document_id]
            selected.extend(
                _fused_search(
                    queries,
                    pool,
                    limit=plan.per_document_limit,
                    page_weights=CORE_PAGE_WEIGHTS.get((plan.intent, document_id)),
                )
            )
    else:
        selected = _fused_search(plan.global_queries, all_chunks, limit=plan.total_limit)

    # Preserve the paper-balanced order for comparisons; otherwise sort by score.
    if plan.intent != "comparison":
        selected.sort(key=lambda hit: hit.score, reverse=True)

    anchors = [chunk for chunk in _intent_anchor_chunks(plan.intent) if chunk is not None]
    if anchors:
        existing = {hit.item.id for hit in selected}
        pinned = [SearchHit(item=chunk, score=100.0 - index) for index, chunk in enumerate(anchors)]
        selected = pinned + [hit for hit in selected if hit.item.id not in {chunk.id for chunk in anchors}]

    return selected[: plan.total_limit]


def _fallback_for_intent(intent):
    if intent == "formula":
        chunks = [
            _find_chunk("attention-is-all-you-need", 3, "LayerNorm(x + Sublayer(x))"),
            _find_chunk("attention-is-all-you-need", 4, "Attention(Q, K, V )"),
            _find_chunk("attention-is-all-you-need", 5, "MultiHead(Q, K, V )"),
            _find_chunk("attention-is-all-you-need", 5, "FFN(x)"),
            _find_chunk(
                "attention-is-all-you-need",
                6,
                "That is, each dimension of the positional encoding",
            ),
            _find_chunk("attention-is-all-you-need", 7, "lrate ="),
        ]
        answer = (
            "## Mathematical formulas in *Attention Is All You Need*\n\n"

            "### 1. Residual connection and layer normalization\n\n"
            "$$\\operatorname{LayerNorm}(x + \\operatorname{Sublayer}(x))$$\n\n"
            "The input $x$ is added to the sub-layer output before normalization. [1]\n\n"

            "### 2. Scaled dot-product attention\n\n"
            "$$\\operatorname{Attention}(Q,K,V)"
            "=\\operatorname{softmax}\\left("
            "\\frac{QK^{T}}{\\sqrt{d_k}}"
            "\\right)V$$\n\n"
            "The factor $\\sqrt{d_k}$ prevents large dot products from pushing "
            "softmax into very small-gradient regions. [2]\n\n"

            "### 3. Multi-head attention\n\n"
            "$$\\operatorname{MultiHead}(Q,K,V)"
            "=\\operatorname{Concat}(head_1,\\ldots,head_h)W^O$$\n\n"
            "$$head_i=\\operatorname{Attention}"
            "(QW_i^Q,KW_i^K,VW_i^V)$$\n\n"
            "Each head uses separate learned projections. [3]\n\n"

            "### 4. Position-wise feed-forward network\n\n"
            "$$\\operatorname{FFN}(x)"
            "=\\max(0,xW_1+b_1)W_2+b_2$$\n\n"
            "The paper applies this two-layer feed-forward network independently "
            "at each sequence position. [4]\n\n"

            "### 5. Sinusoidal positional encoding\n\n"
            "$$PE_{(pos,2i)}="
            "\\sin\\left(pos/10000^{2i/d_{model}}\\right)$$\n\n"
            "$$PE_{(pos,2i+1)}="
            "\\cos\\left(pos/10000^{2i/d_{model}}\\right)$$\n\n"
            "The paper uses sine and cosine functions of different frequencies "
            "to encode token positions. [5]\n\n"

            "### 6. Learning-rate schedule\n\n"
            "$$lrate=d_{model}^{-0.5}"
            "\\min(step_{num}^{-0.5},"
            "step_{num}\\,warmup_{steps}^{-1.5})$$\n\n"
            "The rate rises during warm-up and then decays with the inverse "
            "square root of the step number. [6]"
        )
        return answer, chunks

    if intent == "attention-overview":
        chunks = [
            _find_chunk(
                "effective-attention-nmt",
                3,
                "A global context vector",
            ),
            _find_chunk(
                "effective-attention-nmt",
                4,
                "avoiding the expensive computation",
            ),
            _find_chunk(
                "convolutional-seq2seq",
                4,
                "attention of the first layer determines a useful source context",
            ),
            _find_chunk(
                "attention-is-all-you-need",
                4,
                "Scaled Dot-Product Attention",
            ),
            _find_chunk(
                "attention-is-all-you-need",
                4,
                "Multi-Head Attention consists",
            ),
        ]

        answer = (
            "## Attention mechanism\n\n"
            "An **attention mechanism** lets a sequence model selectively weight source information "
            "that is relevant to the current prediction. In attention-based neural machine translation, "
            "the decoder derives a context vector from encoder hidden states using learned alignment "
            "weights. [1]\n\n"
            "### Global and local attention\n\n"
            "Luong et al. distinguish **global attention**, which considers the full set of source hidden "
            "states, from **local attention**, which focuses on a smaller window around an aligned source "
            "position. The local approach reduces the amount of source information considered for each "
            "target step. [1][2]\n\n"
            "### Attention in convolutional sequence models\n\n"
            "Gehring et al. retain attention inside their convolutional decoder. Attention computed by an "
            "earlier decoder layer provides source context that later layers can use when computing their "
            "own attention. [3]\n\n"
            "### Self-attention in the Transformer\n\n"
            "The Transformer makes attention the central mechanism for mixing sequence information. "
            "**Scaled dot-product attention** compares queries with keys and uses the resulting weights "
            "to combine values, while **multi-head attention** performs several learned attention "
            "operations in parallel. [4][5]\n\n"
            "Across the three papers, attention therefore develops from an alignment mechanism used with "
            "recurrent models, to repeated attention within a convolutional decoder, and finally to the "
            "Transformer's primary sequence-processing mechanism. [1][3][4]"
        )

        return answer, chunks

    if intent == "transformer":
        chunks = [
            _find_chunk("attention-is-all-you-need", 2, "Transformer follows this overall architecture"),
            _find_chunk("attention-is-all-you-need", 3, "Decoder: The decoder"),
            _find_chunk("attention-is-all-you-need", 3, "attention function can be described"),
            _find_chunk("attention-is-all-you-need", 4, "Multi-head attention allows"),
            _find_chunk("attention-is-all-you-need", 5, "Positional Encoding"),
            _find_chunk("attention-is-all-you-need", 6, "self-attention layer connects"),
        ]
        answer = (
            "The **Transformer** is a sequence-to-sequence architecture that removes recurrent and convolutional layers and uses attention as its main mechanism for combining information. Its encoder and decoder are built from repeated layers; the encoder uses multi-head self-attention followed by a position-wise feed-forward network, while the decoder adds masked self-attention and attention over encoder outputs. Residual connections and layer normalization surround these sub-layers. [1][2]\n\n"
            "Its basic attention operation forms weighted combinations of values from query-key compatibility scores. Multi-head attention repeats this with several learned projections, allowing the model to attend to different positions and representation subspaces in parallel. [3][4]\n\n"
            "Because attention alone does not encode token order, positional encodings are added to the input embeddings. Removing recurrence also allows positions in a sequence to be processed in parallel, which is a central efficiency advantage described by the paper. [5][6]"
        )
        return answer, chunks

    if intent == "convolutional":
        chunks = [
            _find_chunk("convolutional-seq2seq", 2, "entirely convolutional"),
            _find_chunk("convolutional-seq2seq", 3, "gated linear units"),
            _find_chunk("convolutional-seq2seq", 1, "allow parallelization"),
            _find_chunk("convolutional-seq2seq", 3, "separate attention mechanism"),
            _find_chunk("convolutional-seq2seq", 4, "multiple attention"),
        ]
        answer = (
            "In the ConvS2S paper, a **convolutional neural network** replaces the recurrent encoder and decoder with stacked one-dimensional convolutional blocks. Each block uses a gated linear unit, and residual connections make deeper stacks easier to optimize. [1][2]\n\n"
            "A convolution sees a fixed local window, but stacking layers expands the effective context so higher layers can represent longer-range relationships. Unlike an RNN, positions within the same layer do not depend on the previous time step, so they can be computed in parallel. [3]\n\n"
            "The decoder also has a separate attention mechanism at every layer. These repeated attention steps let successive layers refine which encoder information is useful, while retaining the speed advantage of convolutional computation. [4][5]"
        )
        return answer, chunks

    if intent == "recurrent-attention":
        chunks = [
            _find_chunk("effective-attention-nmt", 3, "3.1 Global Attention"),
            _find_chunk("effective-attention-nmt", 3, "A global context vector"),
            _find_chunk("effective-attention-nmt", 4, "3.2 Local Attention"),
            _find_chunk("effective-attention-nmt", 4, "avoiding the expensive computation"),
        ]
        answer = (
            "Global and local attention differ in how much of the source sequence they inspect for each target word.\n\n"
            "**Global attention** compares the current decoder state with all encoder hidden states, produces an alignment distribution over the entire source, and computes the context vector as a weighted average of all source states. [1][2]\n\n"
            "**Local attention** first selects or predicts an aligned source position and then attends only to a window around that position. This reduces the cost of attending over long source sequences while remaining differentiable and trainable end to end. [3][4]\n\n"
            "In simple terms: global attention searches the full source; local attention searches a focused neighborhood."
        )
        return answer, chunks

    if intent == "neural-network":
        chunks = [
            _find_chunk("effective-attention-nmt", 2, "neural machine translation system is a neural network"),
            _find_chunk("effective-attention-nmt", 3, "Attention-based Models"),
            _find_chunk("convolutional-seq2seq", 2, "entirely convolutional"),
            _find_chunk("convolutional-seq2seq", 3, "separate attention mechanism"),
            _find_chunk("attention-is-all-you-need", 2, "Transformer follows this overall architecture"),
            _find_chunk("attention-is-all-you-need", 5, "encoder contains self-attention"),
        ]
        answer = (
            "Across these papers, a **neural network** is the trainable encoder-decoder system used to map an input sequence to an output sequence. The papers differ mainly in how that network represents context and connects distant sequence positions. [1]\n\n"
            "Luong et al. use stacked recurrent networks: the encoder produces source hidden states, the decoder generates one target token at a time, and attention forms a context vector from the source states. [1][2]\n\n"
            "Gehring et al. replace recurrence with stacked convolutions, gated linear units, residual connections, and attention in every decoder layer, enabling more parallel computation. [3][4]\n\n"
            "Vaswani et al. remove both recurrence and convolution. Their Transformer uses multi-head self-attention and feed-forward layers, with positional encodings supplying sequence-order information. [5][6]"
        )
        return answer, chunks

    if intent == "comparison":
        chunks = [
            _find_chunk("effective-attention-nmt", 3, "3 Attention-based Models"),
            _find_chunk("effective-attention-nmt", 4, "3.2 Local Attention"),
            _find_chunk("convolutional-seq2seq", 2, "entirely convolutional"),
            _find_chunk("convolutional-seq2seq", 3, "separate attention mechanism"),
            _find_chunk("attention-is-all-you-need", 2, "Transformer follows this overall architecture"),
            _find_chunk("attention-is-all-you-need", 5, "encoder contains self-attention"),
        ]
        answer = (
            "The papers show three successive ways to build an encoder-decoder model.\n\n"
            "| Paper | Core sequence mechanism | Use of attention | Main computational characteristic |\n"
            "|---|---|---|---|\n"
            "| Luong et al. | Stacked LSTM recurrence | Global attention over all source states or local attention over a window | Sequential recurrent state updates [1][2] |\n"
            "| Gehring et al. | Stacked convolutions with gated linear units and residual connections | A separate attention module in every decoder layer | Positions in each convolutional layer can be processed in parallel [3][4] |\n"
            "| Vaswani et al. | Multi-head self-attention plus feed-forward layers | Self-attention in encoder/decoder and encoder-decoder attention | Removes recurrence and convolution, giving the most direct sequence-level parallelism [5][6] |\n\n"
            "In short, the progression is **recurrent state -> convolutional receptive fields -> self-attention**. Each paper keeps the encoder-decoder goal but changes the operation used to mix information across positions."
        )
        return answer, chunks

    return None


def _is_meta_question(normalized):
    phrases = (
        "why are answers irrelevant", "why is answer irrelevant", "response quality",
        "why did you answer", "why are almost all answers", "about scholarsync",
        "what can you do", "how does this demo work",
    )
    return any(phrase in normalized for phrase in phrases)


def _retrieval_query(question, history):
    referential_terms = {
        "it", "its", "this", "that", "they", "them", "their", "these",
        "those", "former", "latter", "above", "previous",
    }
    needs_context = bool(set(tokens(question)) & referential_terms)
    if not needs_context or not history:
        return question
    previous_questions = [
        row.get("content", "")
        for row in reversed(history)
        if row.get("role") == "USER" and row.get("content")
    ]
    return f"{previous_questions[0]} {question}" if previous_questions else question


def _is_capability_question(normalized):
    exact = {
        "what do you do", "what can you do", "what are you", "who are you",
        "what is scholarsync", "how can you help", "help", "show capabilities",
        "what else could you do", "what else can you do", "what else do you do", "what more can you do",
        "what other things can you do", "show me what you can do",
    }
    return normalized in exact or bool(
        re.fullmatch(r"what\s+(?:else|more|other things)\s+(?:can|could)\s+you\s+do", normalized)
    )


def _is_acknowledgement(normalized):
    exact = {
        "thanks", "thank you", "thanks a lot", "ok thanks", "okay thanks",
        "ok thank you", "okay thank you", "got it", "understood", "ok", "okay",
        "great", "awesome", "nice", "perfect", "cool", "good", "sounds good",
        "excellent", "amazing", "wonderful", "fantastic", "very good",
        "great thanks", "great thank you",
    }
    if normalized in exact:
        return True
    return bool(
        re.fullmatch(
            r"(?:(?:ok|okay|great|perfect|awesome|nice|cool|excellent|amazing|wonderful|fantastic)\s+)?"
            r"(?:thanks|thank you)(?:\s+scholarsync)?",
            normalized,
        )
    )


_ANSWER_CITATION_RE = re.compile(r"\[(\d+)\]")


def _has_claim_local_citations(answer):
    """Reject citation-summary lines that are detached from factual claims.

    Good:
        Global attention considers all source states. [1]

    Bad:
        Global attention considers all source states.

        [1][2][3]

    A trailing ``Sources: [1][2]`` line is also treated as detached.
    """
    saw_claim_local = False
    saw_detached = False

    lines = (
        str(answer or "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .split("\n")
    )

    for raw_line in lines:
        line = raw_line.strip()
        if not line or not _ANSWER_CITATION_RE.search(line):
            continue

        without_citations = _ANSWER_CITATION_RE.sub("", line)
        cleaned = re.sub(r"[*_`#>|]+", " ", without_citations)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t,:;.-()[]")

        if not cleaned:
            saw_detached = True
            continue

        if re.fullmatch(
            r"(?:sources?|citations?|evidence)(?:\s+used)?",
            cleaned,
            flags=re.I,
        ):
            saw_detached = True
            continue

        # Require a small amount of real prose so a label or heading followed
        # by citation markers is not mistaken for a grounded factual claim.
        if len(re.findall(r"[A-Za-z]{2,}", cleaned)) >= 2:
            saw_claim_local = True
        else:
            saw_detached = True

    return saw_claim_local and not saw_detached


def answer_demo(question, history=None):
    history = history or []
    normalized = _normalize_question(question)

    if normalized in {"hi", "hello", "hey", "hi scholarsync", "hello scholarsync"}:
        return {
            "answer": (
                "Hello! Ask me about the three demo papers, a concept such as global attention, "
                "ConvS2S, or the Transformer, or ask me to compare the architectures."
            ),
            "confidence": "high",
            "model": "conversation-router",
            "hits": [],
        }

    if _is_capability_question(normalized):
        return {
            "answer": (
                "I am ScholarSync, an evidence-grounded research assistant for this demo library. "
                "I can explain concepts from the three papers, summarize a paper, compare the recurrent, "
                "convolutional, and Transformer approaches, extract supported formulas, show page-level evidence, "
                "open the cited PDF pages, copy answers, and export the complete conversation as a PDF."
            ),
            "confidence": "high",
            "model": "conversation-router",
            "hits": [],
        }

    if _is_acknowledgement(normalized):
        return {
            "answer": "You’re welcome! Ask another question whenever you’re ready.",
            "confidence": "high",
            "model": "conversation-router",
            "hits": [],
        }

    if _is_meta_question(normalized):
        return {
            "answer": (
                "You are asking about ScholarSync itself rather than the papers. The demo answers from retrieved "
                "passages in its three-document library; a weak retrieval match can make an answer feel unrelated. "
                "Try naming the concept or paper—for example, “Explain multi-head attention” or “Compare the three architectures.”"
            ),
            "confidence": "high",
            "model": "conversation-router",
            "hits": [],
        }

    contextual_question = _retrieval_query(
        question,
        history,
    )

    plan = _retrieval_plan(
        contextual_question
    )

    # The public demo uses a fixed, curated three-paper library.
    #
    # These two flagship routes benefit from deterministic output:
    #
    # - attention-overview:
    #   guarantees strong cross-paper grounding.
    #
    # - formula:
    #   guarantees valid, tested LaTeX rather than allowing model
    #   formatting variance to corrupt equations in the browser/PDF.
    if plan.intent in {
        "attention-overview",
        "formula",
    }:
        curated = _fallback_for_intent(
            plan.intent
        )

        if curated is not None:
            answer, chunks = curated

            hits = [
                SearchHit(
                    item=chunk,
                    score=10.0 - index,
                )
                for index, chunk in enumerate(
                    chunks
                )
                if chunk is not None
            ]

            for hit in hits:
                hit.item.page_number = (
                    hit.item.page
                )

            return {
                "answer": answer,
                "confidence": "high",
                "model": "grounded-demo-curated",
                "hits": hits,
            }

    hits = _retrieve(plan)

    for hit in hits:
        hit.item.page_number = hit.item.page

    answer, confidence, model = generate_answer(
        question,
        hits,
        conversation_context=history[-4:],
        answer_mode=plan.intent,
    )

    # A public demo should not showcase a raw retrieval dump or an answer
    # whose citations are detached from the claims they are supposed to support.
    needs_curated_fallback = (
        model == "retrieval-only"
        or (
            bool(hits)
            and not _has_claim_local_citations(answer)
        )
    )

    if needs_curated_fallback:
        curated = _fallback_for_intent(plan.intent)
        if curated is not None:
            answer, chunks = curated
            hits = [
                SearchHit(item=chunk, score=10.0 - index)
                for index, chunk in enumerate(chunks)
                if chunk is not None
            ]
            for hit in hits:
                hit.item.page_number = hit.item.page
            confidence = "high"
            model = "grounded-demo-fallback"

    return {
        "answer": answer,
        "confidence": confidence,
        "model": model,
        "hits": hits,
    }
