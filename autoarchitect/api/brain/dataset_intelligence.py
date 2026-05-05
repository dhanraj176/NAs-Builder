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
from concurrent.futures import ThreadPoolExecutor
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

    # Session-level blacklist — populated when a dataset is blocked (e.g. too many shards)
    _session_blacklist: set = set()

    @classmethod
    def blacklist_dataset(cls, name: str) -> None:
        """Add a dataset to the session blacklist — won't be selected again this session."""
        cls._session_blacklist.add(name)
        print(f"   [Blacklist] {name}")

    def purge_problem_cache(self, keywords: list) -> int:
        """Delete ChromaDB entries whose stored problem contains any of the keywords."""
        if self.collection.count() == 0:
            return 0
        try:
            results   = self.collection.get()
            to_delete = []
            for i, doc in enumerate(results["documents"]):
                if any(kw.lower() in doc.lower() for kw in keywords):
                    to_delete.append(results["ids"][i])
                    meta = results["metadatas"][i]
                    print(f"   [Purge] '{doc[:50]}' -> {meta.get('dataset_id')}")
            if to_delete:
                self.collection.delete(ids=to_delete)
                print(f"   [Purge] Removed {len(to_delete)} entries")
            return len(to_delete)
        except Exception as e:
            print(f"   [Purge] Error: {e}")
            return 0

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

    # ── Multi-source parallel discovery ────────────────────────────────────────

    def discover(self, problem: str, domain: str) -> dict:
        """
        Query Papers With Code, OpenML, Roboflow, and HuggingFace in parallel
        (10 s timeout per source). Rank by: sample count → open license →
        keyword overlap → SOTA benchmark bonus. Min 500 samples enforced.
        Cache best in ChromaDB. Fallback to HuggingFace if all 3 new sources fail.
        """
        search_term = self._build_search_query(problem, domain)
        print(f"   [Discover] Querying 4 sources for: '{search_term}'")

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {
                "pwc":         ex.submit(self._query_papers_with_code, search_term),
                "openml":      ex.submit(self._query_openml,           search_term),
                "roboflow":    ex.submit(self._query_roboflow,         search_term),
                "huggingface": ex.submit(self._query_huggingface_api,  search_term),
            }
            source_results = {}
            for key, fut in futures.items():
                try:
                    source_results[key] = fut.result(timeout=12)
                except Exception as e:
                    print(f"   [Discover] {key} error: {e}")
                    source_results[key] = []

        counts = {k: len(v) for k, v in source_results.items()}
        print(f"   [Discover] PWC:{counts['pwc']} OpenML:{counts['openml']} "
              f"Roboflow:{counts['roboflow']} HF:{counts['huggingface']}")

        all_candidates = []
        for items in source_results.values():
            all_candidates.extend(items)

        # Strip session-blacklisted datasets (blocked for shards etc. earlier this session)
        if DatasetIntelligence._session_blacklist:
            before        = len(all_candidates)
            all_candidates = [c for c in all_candidates
                              if c.get("name") not in DatasetIntelligence._session_blacklist]
            removed = before - len(all_candidates)
            if removed:
                print(f"   [Discover] {removed} blacklisted dataset(s) removed")

        # Fallback to HF Crawl4AI only if ALL 4 sources (incl. HF API) returned nothing
        if not all_candidates:
            print("   [Discover] All sources empty — HuggingFace Crawl4AI fallback")
            hf_ids = self.scout_huggingface_sync(problem, domain)
            return {
                "best": {"name": hf_ids[0], "source": "huggingface_crawl"} if hf_ids else None,
                "candidates": [{"name": h, "source": "huggingface_crawl"} for h in hf_ids],
                "source_counts": counts,
            }

        ranked = self._rank_candidates(all_candidates, problem)
        best   = ranked[0] if ranked else None

        if best:
            sota_acc = float(best.get("sota_accuracy") or 0)
            self.store(problem, domain, best["name"], sota_acc)

        return {
            "best":          best,
            "candidates":    ranked[:10],
            "source_counts": counts,
        }

    def _query_papers_with_code(self, search_term: str) -> list:
        """Papers With Code API — returns name, url, paper_citation, sota_accuracy."""
        resp = requests.get(
            "https://paperswithcode.com/api/v1/datasets/",
            params={"q": search_term.replace("+", " "), "limit": 10},
            headers={"Accept": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        if "json" not in resp.headers.get("content-type", ""):
            return []  # API returned HTML — likely needs auth or has changed
        results = []
        for item in resp.json().get("results", []):
            name = item.get("name", "")
            if not name:
                continue
            paper         = item.get("introduced_in_paper") or {}
            sota_accuracy = None
            if isinstance(paper, dict):
                for metric_row in paper.get("results", []):
                    if isinstance(metric_row, dict):
                        acc = metric_row.get("metrics", {}).get("Accuracy")
                        if acc is not None:
                            sota_accuracy = acc
                            break
            results.append({
                "source":         "papers_with_code",
                "name":           name,
                "url":            item.get("url", ""),
                "paper_citation": paper.get("title", "") if isinstance(paper, dict) else "",
                "sota_accuracy":  sota_accuracy,
                "num_instances":  1000,  # academic benchmark datasets always adequate
                "license":        "unknown",
            })
        return results

    def _query_openml(self, search_term: str) -> list:
        """OpenML — returns name, num_instances, num_features, task_type."""
        try:
            import openml
        except ImportError:
            return []  # skip gracefully if package not installed

        keywords = [k.strip() for k in search_term.split("+") if k.strip()]
        try:
            df = openml.datasets.list_datasets(output_format="dataframe")
        except Exception:
            return []

        # Search with combined term first, then individual keywords
        seen, all_rows = set(), []
        for kw in [" ".join(keywords)] + keywords:
            mask = df["name"].str.contains(kw, case=False, na=False)
            for idx in df[mask].index:
                if idx not in seen:
                    seen.add(idx)
                    all_rows.append(df.loc[idx])

        results = []
        for row in all_rows[:20]:
            raw_n = row.get("NumberOfInstances")
            try:
                n = int(raw_n) if raw_n is not None else 0
            except (ValueError, TypeError):
                continue  # NaN or non-numeric — skip
            if n < 500:
                continue
            did = row.get("did", "")
            results.append({
                "source":        "openml",
                "name":          str(row["name"]),
                "num_instances": n,
                "num_features":  int(row.get("NumberOfFeatures", 0) or 0),
                "task_type":     "classification",  # OpenML list doesn't expose task_type
                "license":       "unknown",          # per-dataset call needed for license
                "url":           f"https://www.openml.org/d/{did}",
                "sota_accuracy": None,
            })
        return results

    def _query_roboflow(self, search_term: str) -> list:
        """Roboflow Universe — returns name, url, num_images, classes."""
        RF_KEY = "VBDUPRj7iL5fDKVVJkNt"
        resp = requests.get(
            "https://universe.roboflow.com/search/datasets",
            params={"q": search_term.replace("+", " ")},
            headers={"Authorization": f"Bearer {RF_KEY}"},
            timeout=10,
        )
        resp.raise_for_status()
        results = []
        for item in resp.json().get("results", []):
            num_images = int(item.get("images", 0) or item.get("num_images", 0) or 0)
            if num_images < 500:
                continue
            classes = item.get("classes", [])
            if isinstance(classes, int):
                classes = [f"class_{i}" for i in range(classes)]
            results.append({
                "source":        "roboflow",
                "name":          item.get("name", "") or item.get("id", ""),
                "url":           item.get("url", ""),
                "num_images":    num_images,
                "classes":       classes,
                "num_instances": num_images,
                "license":       item.get("license", "unknown"),
                "sota_accuracy": None,
            })
        return results

    def _query_huggingface_api(self, search_term: str) -> list:
        """HuggingFace Hub API — tries combined query, then individual keywords."""
        keywords = [k.strip() for k in search_term.split("+") if k.strip()]
        # Try combined first, then each keyword separately (dedup by id)
        queries  = [" ".join(keywords)] + keywords
        seen, results = set(), []

        for q in queries:
            try:
                resp = requests.get(
                    "https://huggingface.co/api/datasets",
                    params={"search": q, "limit": 10, "sort": "downloads"},
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue
            except Exception:
                continue
            for ds in resp.json():
                ds_id = ds.get("id", "")
                if not ds_id or ds_id in seen:
                    continue
                seen.add(ds_id)
                downloads = ds.get("downloads", 0) or 0
                card      = ds.get("cardData") or {}
                if isinstance(card, dict):
                    license_   = card.get("license", "unknown")
                    size_cats  = card.get("size_categories", []) or []
                    task_cats  = card.get("task_categories", []) or []
                else:
                    license_ = "unknown"
                    size_cats, task_cats = [], []
                results.append({
                    "source":          "huggingface",
                    "name":            ds_id,
                    "url":             f"https://huggingface.co/datasets/{ds_id}",
                    "downloads":       downloads,
                    "num_instances":   max(downloads, 500),
                    "license":         license_,
                    "size_categories": size_cats,
                    "task_categories": task_cats,
                    "sota_accuracy":   None,
                })
        return results

    # Hard-excluded by name — removed BEFORE scoring, unconditionally
    _HARD_EXCLUDE_NAMES = (
        "vqa", "caption", "llava", "gpt", "chat", "merged",
    )
    # Soft penalty via task_categories (image tasks only)
    _BAD_IMAGE_TASK_CATS = (
        "visual-question-answering", "image-captioning", "image-to-text",
        "text-to-image", "question-answering", "text-generation",
    )
    _IMAGE_PROBLEM_WORDS = (
        "image", "photo", "picture", "visual", "detect", "scan",
        "leaf", "plant", "disease", "xray", "medical", "defect",
        "object", "classify", "recognition",
    )

    def _rank_candidates(self, candidates: list, problem: str) -> list:
        """
        Rank by: sample count (max 5) + license (max 3) + keyword overlap (max 9)
                 + SOTA bonus (2) − size penalty − shard penalty − format penalty.
        Candidates with < 500 samples or matching HARD_EXCLUDE_NAMES are removed first.
        """
        import math

        # Hard pre-filter — excluded before scoring, regardless of task
        filtered = []
        for c in candidates:
            name_lower = str(c.get("name", "")).lower()
            if any(excl in name_lower for excl in self._HARD_EXCLUDE_NAMES):
                print(f"   [Filter] Hard-excluded: {c['name']}")
                continue
            filtered.append(c)
        candidates = filtered

        p_lower = problem.lower()
        p_words = set(p_lower.split())
        is_image_task = any(w in p_lower for w in self._IMAGE_PROBLEM_WORDS)

        def score(c) -> float:
            n = max(int(c.get("num_instances", 0) or 0), 0)
            if n < 500:
                return -1.0

            # 1. Sample count (max 5 pts) — capped so relevance dominates
            sample_score = min(math.log10(max(n, 1)) * 1.0, 5.0)

            # 2. License (max 3 pts)
            lic = str(c.get("license", "")).lower()
            lic_score = 3.0 if any(
                k in lic for k in ("mit", "cc", "apache", "public", "open")
            ) else (1.0 if lic in ("unknown", "") else 0.5)

            # 3. Keyword overlap via prefix match (max 9 pts)
            name_words = set(
                str(c.get("name", "")).lower()
                .replace("/", " ").replace("-", " ").replace("_", " ").split()
            )
            overlap = sum(
                1 for pw in p_words
                if any(len(pw) > 3 and len(nw) > 3 and
                       (pw.startswith(nw[:4]) or nw.startswith(pw[:4]))
                       for nw in name_words)
            )
            bert_score = min(overlap * 3.0, 9.0)

            # 4. SOTA benchmark bonus (2 pts)
            benchmark = 2.0 if c.get("sota_accuracy") else 0.0

            # 5. Size penalty — prefer datasets under ~500 MB
            size_cats = " ".join(str(s).lower()
                                 for s in (c.get("size_categories") or []))
            if any(x in size_cats for x in (">10gb", "10g<n", "1b<n", ">1b")):
                size_penalty = -5.0   # multi-GB — likely many shards
            elif any(x in size_cats for x in ("100m<n", "1m<n<10m", ">100m")):
                size_penalty = -2.0   # 100 M+ rows — borderline large
            else:
                size_penalty = 0.0

            # 6. Parquet shard penalty (>5 shards = slow / disk-heavy)
            shards = int(c.get("parquet_shards", 0) or 0)
            shard_penalty = -3.0 if shards > 5 else 0.0

            # 7. Wrong-format penalty via task_categories (image tasks only)
            format_penalty = 0.0
            if is_image_task:
                task_cats = " ".join(str(t).lower()
                                     for t in (c.get("task_categories") or []))
                if any(b in task_cats for b in self._BAD_IMAGE_TASK_CATS):
                    format_penalty -= 4.0

            return (sample_score + lic_score + bert_score + benchmark
                    + size_penalty + shard_penalty + format_penalty)

        return sorted(candidates, key=score, reverse=True)

    # ── Crawl4AI Scout (HuggingFace) ───────────────────────────────────────

    def _build_search_query(self, problem: str, domain: str) -> str:
        """Build a focused search query from the problem description (up to 3 keywords)."""
        stop = {"detect", "identify", "classify", "monitor", "analyze",
                "build", "find", "using", "from", "in", "on", "at", "to",
                "for", "and", "or", "the", "a", "an", "with", "that"}
        words = [w.strip(".,!?") for w in problem.lower().split()
                 if w not in stop and len(w) > 3][:3]
        query = "+".join(words) if words else domain
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