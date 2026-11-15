import logging
import os
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_NAME = "nasa_fire_papers"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_CHUNK_LEN = 100
MAX_DISTANCE = 1.4
LLM_MAX_TOKENS = 1500
NO_ANSWER_MSG = "The ingested NASA papers do not contain enough information to answer this question."

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "does", "do", "did", "why", "what", "how", "when", "where", "which",
    "in", "on", "of", "to", "and", "or", "it", "this", "that", "these",
    "those", "for", "with", "by", "as", "at", "from", "about", "into",
    "has", "have", "had", "will", "would", "can", "could", "should",
    "than", "then", "so", "if", "not", "no", "but", "also", "very",
}


def _extract_text_pymupdf(pdf_path):
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def _extract_text_pypdf(pdf_path):
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_pdf_text(pdf_path):
    for name, extractor in (("pymupdf", _extract_text_pymupdf), ("pypdf", _extract_text_pypdf)):
        try:
            text = extractor(pdf_path)
            if text.strip():
                return text, name
        except Exception as e:
            logger.debug(f"{name} failed for {pdf_path.name}: {e}")
    return "", "failed"


def _keywords(text):
    return set(re.findall(r"[a-z]{3,}", text.lower())) - STOPWORDS


def _synthesize_fallback(question, context, sources):
    q_words = _keywords(question)
    if not q_words:
        return (
            "Please ask a more specific question (e.g. 'why does flame quench "
            "in microgravity?' or 'what is LOI?').\n\n"
            f"Available sources: {', '.join(sources)}"
        )

    sentences = re.split(r"(?<=[.!?])\s+", context)
    scored = []
    for s in sentences:
        s = s.strip().replace("\n", " ")
        if len(s) < 50 or len(s) > 420:
            continue
        if re.match(r"^\d+\.", s):
            continue
        if s.count(" ") < 8:
            continue
        overlap = len(q_words & _keywords(s))
        if overlap == 0:
            continue
        scored.append((overlap / max(len(q_words), 1), s))

    scored.sort(key=lambda x: x[0], reverse=True)

    seen, top = set(), []
    for _, s in scored:
        key = s[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        top.append(s)
        if len(top) >= 3:
            break

    if not top:
        return (
            "No direct match found in the NASA paper excerpts for this question. "
            "Try terms like 'radiative quench', 'opposed flow', 'LOI', "
            "'Saffire', 'buoyancy', or 'microgravity'.\n\n"
            f"Available sources: {', '.join(sources)}"
        )

    body = "\n\n".join(f"{i+1}. {s}" for i, s in enumerate(top))
    return (
        f"Based on NASA microgravity combustion literature, "
        f"the relevant findings are:\n\n{body}\n\nSources: {', '.join(sources)}"
    )


class NASAKnowledgeBase:
    def __init__(self, papers_dir=None, persist_dir=None, force_reingest=False):
        if papers_dir is None:
            papers_dir = Path(__file__).parent / "papers"
        if persist_dir is None:
            persist_dir = Path(__file__).parent / "chroma_db"

        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )

        if force_reingest:
            try:
                self.client.delete_collection(COLLECTION_NAME)
                logger.info("Existing collection deleted for re-ingestion.")
            except Exception as e:
                logger.debug(f"No existing collection to delete: {e}")

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=self.embed_fn
        )

        self.llm_client = self._init_llm_client()

        if self.collection.count() == 0:
            self._ingest(papers_dir)

    def _init_llm_client(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set - using fallback synthesizer.")
            return None
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            logger.info("RAG: LLM enabled (Groq).")
            return client
        except ImportError:
            logger.warning("groq package not installed - using fallback synthesizer.")
            return None

    def _ingest(self, papers_dir):
        path = Path(papers_dir)
        if not path.exists():
            logger.warning(f"Papers folder not found: {papers_dir}")
            return

        pdf_files = sorted(path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDFs found in {papers_dir}")
            return

        logger.info(f"Ingesting {len(pdf_files)} PDFs...")
        all_chunks, all_metas, all_ids = [], [], []

        for pdf_file in pdf_files:
            text, extractor = extract_pdf_text(pdf_file)
            if not text:
                logger.warning(f"  {pdf_file.name}: FAILED")
                continue

            file_chunks = 0
            for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk = text[i:i + CHUNK_SIZE].strip()
                if len(chunk) > MIN_CHUNK_LEN:
                    all_chunks.append(chunk)
                    all_metas.append({"source": pdf_file.stem, "chunk_id": i})
                    all_ids.append(f"{pdf_file.stem}_{i}")
                    file_chunks += 1

            logger.info(f"  {pdf_file.name}: {file_chunks} chunks ({extractor})")

        if not all_chunks:
            return

        for i in range(0, len(all_chunks), 100):
            self.collection.add(
                documents=all_chunks[i:i + 100],
                metadatas=all_metas[i:i + 100],
                ids=all_ids[i:i + 100],
            )
        logger.info(f"RAG: ingested {len(all_chunks)} chunks.")

    def query(self, question, n_results=5):
        fetch_n = max(n_results * 3, 15)
        results = self.collection.query(query_texts=[question], n_results=fetch_n)

        if not results["documents"] or not results["documents"][0]:
            return {"answer": "No relevant papers found.", "sources": []}

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = (results.get("distances") or [[None] * len(docs)])[0]

        if any(d is not None for d in dists):
            kept = [(d, m, dist) for d, m, dist in zip(docs, metas, dists)
                    if dist is None or dist <= MAX_DISTANCE]
            logger.info("RAG distances (top %d, kept %d): %s", 
                        len(dists), len(kept), 
                        [round(d, 3) for d in dists if d is not None])
        else:
            kept = list(zip(docs, metas, dists))

        if not kept:
            return {"answer": NO_ANSWER_MSG, "sources": []}

        kept = kept[:n_results]

        docs = [k[0] for k in kept]
        metas = [k[1] for k in kept]

        context = "\n\n---\n\n".join(docs)
        sources = sorted({m["source"] for m in metas})

        if not self.llm_client:
            return {
                "answer": _synthesize_fallback(question, context, sources),
                "sources": sources,
            }

        prompt = (
            "You are a NASA fire safety researcher. Answer using ONLY the context "
            "below from NASA microgravity combustion papers. Be concise (3-5 sentences) "
            "and cite sources in the form [source_name]. "
            "If the context does not contain the information needed to answer, reply exactly: "
            f"\"{NO_ANSWER_MSG}\" Do not use outside knowledge and do not guess.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer with citations:"
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=LLM_MAX_TOKENS,
            )
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                logger.warning("LLM returned empty content - using fallback synthesizer.")
                return {
                    "answer": _synthesize_fallback(question, context, sources),
                    "sources": sources,
                }
            return {
                "answer": answer,
                "sources": sources,
            }
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {
                "answer": _synthesize_fallback(question, context, sources),
                "sources": sources,
            }