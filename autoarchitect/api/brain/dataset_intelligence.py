"""
dataset_intelligence.py — ChromaDB-powered Dataset Memory

Stage 0 of the discovery pipeline — runs BEFORE everything else.

Flow:
    1. Query ChromaDB for similar past problems (instant)
    2. If found → return proven dataset immediately
    3. If not found → Crawl4AI scouts HuggingFace for candidates
    4. Validate candidates → run training
    5. Store successful results back in ChromaDB (learns forever)

Every successful training makes the system smarter.
No ceiling. No human. Compounds forever.
"""

import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime

# ChromaDB
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR  = Path(__file__).parent.parent.parent
CHROMA_DIR = BASE_DIR / "datasets" / "chromadb"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


class DatasetIntelligence:
    """
    ChromaDB-powered dataset memory.
    Remembers every successful dataset-problem pair forever.
    """

    def __init__(self, groq_api_key: str = ""):
        self.groq_key = groq_api_key or os.getenv("GROQ_API_KEY", "")

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        # Use sentence transformers for embeddings (free, local)
        self.ef = embedding_functions.DefaultEmbeddingFunction()

        # Main collection — stores proven dataset solutions
        self.collection = self.client.get_or_create_collection(
            name="dataset_solutions",
            embedding_function=self.ef,
            metadata={"description": "Proven dataset-problem pairs"}
        )

        count = self.collection.count()
        print(f"DatasetIntelligence ready — {count} proven solutions in memory")

    # ── Main entry — Stage 0 ───────────────────────────────────────────────

    def recall(self, problem: str, domain: str) -> dict:
        """
        Query ChromaDB for similar past problems.
        Returns proven dataset if found, None if not.
        """
        if self.collection.count() == 0:
            return None

        try:
            results = self.collection.query(
                query_texts=[problem],
                n_results=min(3, self.collection.count()),
                where={"domain": domain} if domain else None,
            )

            if not results["documents"] or not results["documents"][0]:
                return None

            # Check similarity threshold
            distances = results["distances"][0]
            if not distances or distances[0] > 0.4:
                return None

            # Best match
            metadata = results["metadatas"][0][0]
            distance = distances[0]
            similarity = round((1 - distance) * 100, 1)

            print(f"   ChromaDB recall: {similarity}% match")
            print(f"   Proven dataset: {metadata['dataset_id']}")
            print(f"   Past accuracy:  {metadata.get('accuracy', 0)}%")

            return {
                "dataset_id":  metadata["dataset_id"],
                "accuracy":    metadata.get("accuracy", 0),
                "domain":      metadata.get("domain", domain),
                "similarity":  similarity,
                "from_memory": True,
                "past_problem": results["documents"][0][0],
            }

        except Exception as e:
            print(f"   ChromaDB recall error: {e}")
            return None

    def store(self, problem: str, domain: str, dataset_id: str,
              accuracy: float, method: str = "resnet18"):
        """
        Store a successful training result in ChromaDB.
        Called after every successful training run.
        """
        if accuracy <= 0:
            return  # Don't store failed runs

        try:
            # Use problem as document, metadata stores the solution
            doc_id = f"{domain}_{hash(problem) % 100000}"

            self.collection.upsert(
                documents=[problem],
                ids=[doc_id],
                metadatas=[{
                    "dataset_id":  dataset_id,
                    "domain":      domain,
                    "accuracy":    accuracy,
                    "method":      method,
                    "stored_at":   datetime.now().isoformat(),
                    "problem_preview": problem[:100],
                }]
            )

            count = self.collection.count()
            print(f"   ChromaDB stored: {dataset_id} ({accuracy}%) — {count} total solutions")

        except Exception as e:
            print(f"   ChromaDB store error: {e}")

    # ── Crawl4AI Scout ─────────────────────────────────────────────────────

    async def scout_huggingface(self, problem: str, domain: str) -> list:
        """
        Use Crawl4AI to scrape HuggingFace dataset search results.
        Returns list of candidate dataset IDs.
        """
        try:
            from crawl4ai import AsyncWebCrawler
            from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

            # Build search query from problem
            query = self._build_search_query(problem, domain)
            url   = f"https://huggingface.co/datasets?search={query}&sort=downloads"

            print(f"   Crawl4AI scouting: {url[:60]}")

            schema = {
                "name": "HuggingFace Datasets",
                "baseSelector": "article",
                "fields": [
                    {"name": "dataset_id", "selector": "a", "type": "attribute", "attribute": "href"},
                    {"name": "title",      "selector": "h4", "type": "text"},
                    {"name": "downloads",  "selector": "[data-downloads]", "type": "attribute", "attribute": "data-downloads"},
                ]
            }

            strategy = JsonCssExtractionStrategy(schema)

            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(
                    url=url,
                    extraction_strategy=strategy,
                    bypass_cache=True,
                )

            if not result.success:
                return []

            candidates = []
            extracted  = json.loads(result.extracted_content or "[]")

            for item in extracted[:10]:
                ds_id = item.get("dataset_id", "")
                if ds_id and ds_id.startswith("/"):
                    ds_id = ds_id.lstrip("/")
                if ds_id and "/" in ds_id:
                    candidates.append(ds_id)

            print(f"   Crawl4AI found: {len(candidates)} candidates")
            return candidates

        except Exception as e:
            print(f"   Crawl4AI scout failed: {e}")
            return []

    def scout_huggingface_sync(self, problem: str, domain: str) -> list:
        """Synchronous wrapper for scout_huggingface."""
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.scout_huggingface(problem, domain))
            loop.close()
            return result
        except Exception as e:
            print(f"   Scout sync error: {e}")
            return []

    def _build_search_query(self, problem: str, domain: str) -> str:
        """Build a focused search query from the problem description."""
        stop = {"detect", "identify", "classify", "monitor", "analyze",
                "build", "find", "using", "from", "in", "on", "at", "to",
                "for", "and", "or", "the", "a", "an", "with", "that"}
        words = [w.strip(".,!?") for w in problem.lower().split()
                 if w not in stop and len(w) > 3][:3]
        query = "+".join(words[:2]) if words else domain
        return query

    # ── Validate candidate ─────────────────────────────────────────────────

    def validate_candidate(self, dataset_id: str) -> bool:
        """Quick streaming validation — checks dataset loads and has labels."""
        try:
            from datasets import load_dataset
            ds    = load_dataset(dataset_id, streaming=True)
            split = ds.get("train", list(ds.values())[0])
            row   = next(iter(split))
            cols  = list(row.keys())

            has_label = any(c in cols for c in
                            ["label", "labels", "class", "category", "target"])
            has_image = any("image" in c.lower() for c in cols)
            has_text  = any(c in cols for c in
                            ["text", "sentence", "content", "message"])

            return has_label and (has_image or has_text)

        except Exception:
            return False

    # ── Stats ──────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        count = self.collection.count()
        return {
            "total_solutions": count,
            "storage_path":    str(CHROMA_DIR),
        }

    def list_solutions(self) -> list:
        """List all stored solutions."""
        if self.collection.count() == 0:
            return []
        try:
            results = self.collection.get()
            solutions = []
            for i, doc in enumerate(results["documents"]):
                meta = results["metadatas"][i]
                solutions.append({
                    "problem":    doc[:60],
                    "dataset":    meta.get("dataset_id"),
                    "accuracy":   meta.get("accuracy"),
                    "domain":     meta.get("domain"),
                    "stored_at":  meta.get("stored_at"),
                })
            return solutions
        except Exception:
            return []