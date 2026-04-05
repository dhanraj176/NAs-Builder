# ============================================
# AutoArchitect — Self Learning Cache System
# Now with BERT Semantic Similarity
# Same problem, different words = cache hit!
# ============================================

import os
import json
import hashlib
import time
import torch
import numpy as np
from datetime import datetime

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'cache'
)

# BERT loaded once, reused forever
_bert_model     = None
_bert_tokenizer = None

def _get_bert():
    """Load BERT once and reuse"""
    global _bert_model, _bert_tokenizer
    if _bert_model is None:
        print("🧠 Loading BERT for semantic cache...")
        from transformers import BertTokenizer, BertModel
        bert_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'models', 'bert'
        )
        _bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
        _bert_model     = BertModel.from_pretrained(bert_path)
        _bert_model.eval()
        print("✅ Semantic cache ready!")
    return _bert_tokenizer, _bert_model


def get_embedding(text: str) -> list:
    """Convert problem text to BERT embedding vector."""
    try:
        tokenizer, model = _get_bert()
        inputs = tokenizer(
            text,
            max_length     = 64,
            padding        = 'max_length',
            truncation     = True,
            return_tensors = 'pt'
        )
        with torch.no_grad():
            outputs   = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
            embedding = torch.nn.functional.normalize(
                embedding, dim=1)
        return embedding[0].tolist()
    except Exception as e:
        print(f"⚠️ Embedding failed: {e}")
        return []


def cosine_similarity(v1: list, v2: list) -> float:
    """Compare two embeddings — 1.0 = identical"""
    if not v1 or not v2:
        return 0.0
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def get_problem_hash(problem):
    """Convert problem description to unique ID"""
    import re
    cleaned    = re.sub(r'[^\w\s]', '', problem)
    normalized = ' '.join(cleaned.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()[:10]


def check_cache(problem):
    """Check exact match first, then semantic similarity"""
    h          = get_problem_hash(problem)
    cache_path = os.path.join(CACHE_DIR, h)
    meta_path  = os.path.join(cache_path, 'metadata.json')

    # 1. Exact match
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        return {
            'found':      True,
            'hash':       h,
            'path':       cache_path,
            'model_path': os.path.join(cache_path, 'model.pth'),
            'metadata':   metadata,
            'match_type': 'exact'
        }

    # 2. Semantic similarity
    semantic = find_semantic_match(problem)
    if semantic:
        print(f"🧠 Semantic match found! "
              f"'{problem[:30]}' ≈ '{semantic['problem'][:30]}'")
        sem_hash = get_problem_hash(semantic['problem'])
        return {
            'found':      True,
            'hash':       sem_hash,
            'path':       os.path.join(CACHE_DIR, sem_hash),
            'model_path': os.path.join(
                CACHE_DIR, sem_hash, 'model.pth'),
            'metadata':   semantic,
            'match_type': 'semantic',
            'similarity': semantic.get('_similarity', 0)
        }

    return {'found': False}


def find_semantic_match(problem: str,
                        threshold: float = 0.88) -> dict:
    """Find cached problem with similar meaning."""
    if not os.path.exists(CACHE_DIR):
        return None

    new_embedding = get_embedding(problem)
    if not new_embedding:
        return find_similar_cached(problem, 'image')

    best_match = None
    best_score = 0.0

    for folder in os.listdir(CACHE_DIR):
        meta_path = os.path.join(
            CACHE_DIR, folder, 'metadata.json')
        if not os.path.exists(meta_path):
            continue

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        cached_embedding = meta.get('embedding', [])
        if not cached_embedding:
            continue

        score = cosine_similarity(new_embedding, cached_embedding)

        if score > best_score and score >= threshold:
            best_score = score
            best_match = meta.copy()
            best_match['_similarity'] = round(score, 3)

    if best_match:
        print(f"  🧠 Semantic similarity: "
              f"{best_match['_similarity']} "
              f"(threshold: {threshold})")
    return best_match


def save_to_cache(problem, category, confidence,
                  architecture, parameters, search_time,
                  **kwargs):
    """Save solution + BERT embedding to cache"""
    h          = get_problem_hash(problem)
    cache_path = os.path.join(CACHE_DIR, h)
    os.makedirs(cache_path, exist_ok=True)

    print(f"  🧠 Computing BERT embedding for cache...")
    embedding = get_embedding(problem)

    metadata = {
        'problem':          problem,
        'category':         category,
        'confidence':       confidence,
        'architecture':     architecture,
        'parameters':       parameters,
        'search_time':      search_time,
        'trained_at':       datetime.now().isoformat(),
        'use_count':        1,
        # Multi-agent fields
        'result_type':      kwargs.get('result_type',     'single'),
        'agents_used':      kwargs.get('agents_used',     [category]),
        'self_trained':     kwargs.get('self_trained',    False),
        'avg_accuracy':     kwargs.get('avg_accuracy',    0),
        'all_accuracies':   kwargs.get('all_accuracies',  {}),
        'evaluation':       kwargs.get('evaluation',      {}),
        # User data fields ← FIXED
        'user_model_path':  kwargs.get('user_model_path', ''),
        'classes':          kwargs.get('classes',         []),
        # BERT embedding
        'embedding':        embedding,
    }

    with open(os.path.join(cache_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved to cache: {h} ({problem[:30]}...)")
    return h


def increment_use_count(cache_result):
    """Track how many times a cached solution is used"""
    meta_path = os.path.join(
        cache_result['path'], 'metadata.json')
    metadata  = cache_result['metadata']
    metadata['use_count'] = metadata.get('use_count', 1) + 1
    metadata['last_used'] = datetime.now().isoformat()
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def get_cache_stats():
    """Get statistics about knowledge base"""
    if not os.path.exists(CACHE_DIR):
        return {
            'total_solutions': 0,
            'total_uses':      0,
            'categories':      {},
            'recent':          []
        }

    solutions  = []
    categories = {}
    total_uses = 0

    for folder in os.listdir(CACHE_DIR):
        meta_path = os.path.join(
            CACHE_DIR, folder, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            meta_clean = {k: v for k, v in meta.items()
                         if k != 'embedding'}
            solutions.append(meta_clean)
            cat = meta.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
            total_uses      += meta.get('use_count', 1)

    solutions.sort(
        key=lambda x: x.get('trained_at', ''),
        reverse=True
    )

    return {
        'total_solutions': len(solutions),
        'total_uses':      total_uses,
        'categories':      categories,
        'recent':          solutions[:5]
    }


def find_similar_cached(problem, category):
    """Fallback: word overlap matching"""
    if not os.path.exists(CACHE_DIR):
        return None

    problem_words = set(problem.lower().split())
    best_match    = None
    best_score    = 0

    for folder in os.listdir(CACHE_DIR):
        meta_path = os.path.join(
            CACHE_DIR, folder, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            if meta.get('category') != category:
                continue

            cached_words = set(
                meta.get('problem', '').lower().split())
            overlap = len(problem_words & cached_words)
            score   = overlap / max(len(problem_words), 1)

            if score > best_score and score > 0.3:
                best_score = score
                best_match = meta

    return best_match
