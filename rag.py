import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from logger import setup_logger

logger = setup_logger()

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
DOCS_PATH = os.path.join(DATA_DIR, "docs.pkl")
KB_PATH = "knowledge_base"


class RAG:
    def __init__(self):
        logger.info("Initializing RAG pipeline")

        os.makedirs(DATA_DIR, exist_ok=True)

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.docs = []
        self.index = None

        if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
            self._load_index()
        else:
            self._build_index()


    def _load_index(self):
        logger.info("Loading FAISS index from %s", INDEX_PATH)

        self.index = faiss.read_index(INDEX_PATH)

        with open(DOCS_PATH, "rb") as f:
            self.docs = pickle.load(f)

        logger.info("Loaded FAISS index with %d documents", len(self.docs))

    
    def _build_index(self):
        logger.info("Building FAISS index from knowledge base")

        for file_name in os.listdir(KB_PATH):
            file_path = os.path.join(KB_PATH, file_name)
            if os.path.isfile(file_path):
                with open(file_path, encoding="utf-8") as f:
                    self.docs.append(f.read())

        if not self.docs:
            raise ValueError("Knowledge base is empty")

        embeddings = self.embedder.encode(
            self.docs,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        # Persist
        faiss.write_index(self.index, INDEX_PATH)
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(self.docs, f)

        logger.info(
            "FAISS index built and saved at %s (%d docs, dim=%d)",
            INDEX_PATH,
            len(self.docs),
            dim
        )


    def retrieve(self, query, k=3):
        logger.info("Retrieving context for query")

        query_emb = self.embedder.encode(
            [query],
            convert_to_numpy=True
        )

        _, idxs = self.index.search(query_emb, k)
        return [self.docs[i] for i in idxs[0]]