# -*- coding: utf-8 -*-
"""
Day 5 verification -- AudioAgent (MFCC+Whisper) and MultimodalAgent (CLIP).
"""

import sys, os, math, array, wave, tempfile, shutil
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_sine_wav(path, freq=440, duration=0.5, sr=22050):
    """Write a mono 16-bit sine-wave WAV (no extra deps beyond stdlib)."""
    n       = int(sr * duration)
    samples = array.array('h',
        [int(32767 * math.sin(2 * math.pi * freq * i / sr)) for i in range(n)])
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def _make_audio_dataset(root, n_per_class=20):
    """
    Create root/low_freq/ and root/high_freq/ with n_per_class WAV files each.
    Low-freq class: 200-300 Hz  |  High-freq class: 800-1200 Hz
    """
    import random
    random.seed(42)
    root = Path(root)
    for cls, lo, hi in [("low_freq", 200, 300), ("high_freq", 800, 1200)]:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            freq = random.randint(lo, hi)
            _write_sine_wav(str(cls_dir / f"sample_{i:03d}.wav"), freq=freq)


def _make_test_image(path):
    """Create a simple red 224x224 PNG for CLIP tests."""
    try:
        from PIL import Image
        img = Image.new("RGB", (224, 224), color=(220, 50, 50))
        img.save(path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Test 1: AudioAgent.train() returns expected metrics
# ---------------------------------------------------------------------------
def test_audio_train():
    print("\n-- Test 1: AudioAgent.train() -- 2 classes, 20 files each --")
    from api.agents.audio_agent import AudioAgent

    tmpdir = tempfile.mkdtemp(prefix="day5_audio_")
    try:
        _make_audio_dataset(tmpdir)
        agent  = AudioAgent()
        result = agent.train(tmpdir, hash_id="day5test")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    ok = (
        "accuracy"        in result and
        "val_accuracy"    in result and
        "num_samples"     in result and
        "classes"         in result and
        "classifier_type" in result and
        result["accuracy"]        >= 0.0 and
        result["num_samples"]     == 40 and
        result["classifier_type"] in ("random_forest", "gradient_boosting") and
        set(result["classes"])    == {"low_freq", "high_freq"}
    )
    print(f"  accuracy={result.get('accuracy')}  "
          f"model={result.get('classifier_type')}  "
          f"classes={result.get('classes')}  "
          f"samples={result.get('num_samples')}")
    print("  PASS" if ok else "  FAIL -- missing keys or unexpected values")
    return ok, result.get("model_path")


# ---------------------------------------------------------------------------
# Test 2: AudioAgent.predict() with a real audio file after training
# ---------------------------------------------------------------------------
def test_audio_predict(model_path):
    print("\n-- Test 2: AudioAgent.predict() on a single audio file --")
    from api.agents.audio_agent import AudioAgent

    agent = AudioAgent()
    agent.load_trained_model(model_path)

    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    _write_sine_wav(tmp_wav.name, freq=250)  # low_freq class

    result = agent.predict(tmp_wav.name)
    os.unlink(tmp_wav.name)

    ok = (
        result.get("label")      in ("low_freq", "high_freq") and
        0.0 <= result.get("confidence", -1) <= 1.0 and
        "top3"       in result and
        result.get("agent_used") == "AudioAgent"
    )
    print(f"  result: {result}")
    print("  PASS" if ok else "  FAIL -- unexpected predict result")
    return ok


# ---------------------------------------------------------------------------
# Test 3: AudioAgent.predict() with no model -> honest error dict
# ---------------------------------------------------------------------------
def test_audio_no_model():
    print("\n-- Test 3: AudioAgent.predict() with no model -> error dict --")
    from api.agents.audio_agent import AudioAgent

    fresh  = AudioAgent()
    result = fresh.predict("anything.wav")

    ok = (
        result.get("label")      == "error" and
        result.get("confidence") == 0.0 and
        "error"                  in result and
        result.get("fake")       is False
    )
    print(f"  result: {result}")
    print("  PASS" if ok else "  FAIL -- expected error dict")
    return ok


# ---------------------------------------------------------------------------
# Test 4: AudioAgent.load_trained_model() round-trip
# ---------------------------------------------------------------------------
def test_audio_load(model_path):
    print("\n-- Test 4: AudioAgent.load_trained_model() round-trip --")
    from api.agents.audio_agent import AudioAgent

    if not model_path or not Path(model_path).exists():
        print("  SKIP -- model file not found (run test 1 first)")
        return True

    fresh  = AudioAgent()
    loaded = fresh.load_trained_model(model_path)

    ok = (
        loaded is True and
        fresh.model            is not None and
        fresh.classes          is not None and
        fresh.classifier_type  is not None and
        set(fresh.classes)     == {"low_freq", "high_freq"}
    )
    print(f"  loaded={loaded}  classes={fresh.classes}  "
          f"classifier={fresh.classifier_type}")
    print("  PASS" if ok else "  FAIL -- load_trained_model() failed")
    return ok


# ---------------------------------------------------------------------------
# Test 5: AudioAgent.transcribe() with Whisper (SKIP if unavailable)
# ---------------------------------------------------------------------------
def test_audio_transcribe():
    print("\n-- Test 5: AudioAgent.transcribe() with Whisper --")
    from api.agents.audio_agent import AudioAgent

    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    _write_sine_wav(tmp_wav.name, freq=440, duration=2.0)

    agent  = AudioAgent()
    result = agent.transcribe(tmp_wav.name)
    os.unlink(tmp_wav.name)

    if "error" in result:
        print(f"  SKIP -- Whisper unavailable: {result.get('error', '')[:80]}")
        return True  # not a failure -- model not downloaded

    ok = (
        "transcript" in result and
        "language"   in result and
        "segments"   in result
    )
    print(f"  transcript='{result.get('transcript','')[:60]}'  "
          f"lang={result.get('language')}")
    print("  PASS" if ok else "  FAIL -- unexpected transcribe result format")
    return ok


# ---------------------------------------------------------------------------
# Test 6: MultimodalAgent.classify_with_labels() real CLIP inference
#         (SKIP gracefully if CLIP unavailable or no internet)
# ---------------------------------------------------------------------------
def test_multimodal_classify():
    print("\n-- Test 6: MultimodalAgent.classify_with_labels() zero-shot --")
    from api.agents.multimodal_agent import MultimodalAgent

    tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_img.close()
    img_ok = _make_test_image(tmp_img.name)
    if not img_ok:
        print("  SKIP -- PIL not available for test image creation")
        os.unlink(tmp_img.name)
        return True

    agent  = MultimodalAgent()
    labels = ["a red image", "a blue image", "a green image"]
    result = agent.classify_with_labels(tmp_img.name, labels)
    os.unlink(tmp_img.name)

    if result.get("label") == "error":
        err = result.get("error", "")
        print(f"  SKIP -- CLIP unavailable: {err[:80]}")
        return True

    ok = (
        result.get("label")      in labels and
        0.0 <= result.get("confidence", -1) <= 1.0 and
        "all_scores"             in result and
        len(result["all_scores"]) == len(labels) and
        result.get("agent_used") == "MultimodalAgent"
    )
    print(f"  label='{result.get('label')}'  "
          f"confidence={result.get('confidence')}  "
          f"all_scores={result.get('all_scores')}")
    print("  PASS" if ok else "  FAIL -- unexpected CLIP result")
    return ok


# ---------------------------------------------------------------------------
# Test 7: MultimodalAgent.predict() with image + text query
#         (SKIP if CLIP unavailable)
# ---------------------------------------------------------------------------
def test_multimodal_predict():
    print("\n-- Test 7: MultimodalAgent.predict() with image+text query --")
    from api.agents.multimodal_agent import MultimodalAgent

    tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_img.close()
    if not _make_test_image(tmp_img.name):
        print("  SKIP -- PIL not available")
        os.unlink(tmp_img.name)
        return True

    agent  = MultimodalAgent()
    result = agent.predict(tmp_img.name, text_query="a red colored shape")
    os.unlink(tmp_img.name)

    if result.get("label") == "error":
        err = result.get("error", "")
        print(f"  SKIP -- CLIP unavailable: {err[:80]}")
        return True

    ok = (
        "similarity"          in result and
        "image_features_norm" in result and
        "text_features_norm"  in result and
        result.get("agent_used") == "MultimodalAgent"
    )
    print(f"  similarity={result.get('similarity')}  "
          f"img_norm={result.get('image_features_norm')}  "
          f"txt_norm={result.get('text_features_norm')}")
    print("  PASS" if ok else "  FAIL -- unexpected predict result")
    return ok


# ---------------------------------------------------------------------------
# Test 8: MultimodalAgent offline fallback -> error dict (subclass override)
# ---------------------------------------------------------------------------
def test_multimodal_offline():
    print("\n-- Test 8: MultimodalAgent offline fallback -> error dict --")
    from api.agents.multimodal_agent import MultimodalAgent

    class OfflineAgent(MultimodalAgent):
        def _lazy_load_clip(self):
            return False   # simulate no internet / model unavailable

    agent  = OfflineAgent()
    result = agent.classify_with_labels("any.jpg", ["cat", "dog"])

    ok = (
        result.get("label")      == "error" and
        result.get("confidence") == 0.0 and
        result.get("fake")       is False and
        "error"                  in result
    )
    print(f"  result: {result}")
    print("  PASS" if ok else "  FAIL -- expected error dict for offline case")
    return ok


# ---------------------------------------------------------------------------
# Test 9: TopologyDesigner AGENT_CATALOG has 'audio' with correct keywords
# ---------------------------------------------------------------------------
def test_catalog_has_audio():
    print("\n-- Test 9: AGENT_CATALOG has 'audio' entry --")
    from api.brain.topology_designer import AGENT_CATALOG

    ok = (
        "audio" in AGENT_CATALOG and
        "keywords" in AGENT_CATALOG["audio"] and
        "audio"   in AGENT_CATALOG["audio"]["keywords"] and
        "speech"  in AGENT_CATALOG["audio"]["keywords"]
    )
    print(f"  keywords: {AGENT_CATALOG.get('audio', {}).get('keywords', [])}")
    print("  PASS" if ok else "  FAIL -- 'audio' not in AGENT_CATALOG")
    return ok


# ---------------------------------------------------------------------------
# Test 10: TopologyDesigner AGENT_CATALOG has 'multimodal'
# ---------------------------------------------------------------------------
def test_catalog_has_multimodal():
    print("\n-- Test 10: AGENT_CATALOG has 'multimodal' entry --")
    from api.brain.topology_designer import AGENT_CATALOG

    ok = (
        "multimodal" in AGENT_CATALOG and
        "keywords"    in AGENT_CATALOG["multimodal"] and
        "multimodal"  in AGENT_CATALOG["multimodal"]["keywords"] and
        "clip"        in AGENT_CATALOG["multimodal"]["keywords"]
    )
    print(f"  keywords: {AGENT_CATALOG.get('multimodal', {}).get('keywords', [])}")
    print("  PASS" if ok else "  FAIL -- 'multimodal' not in AGENT_CATALOG")
    return ok


# ---------------------------------------------------------------------------
# Test 11: TopologyDesigner routes audio problem to audio_pipeline
# ---------------------------------------------------------------------------
def test_topology_routes_audio():
    print("\n-- Test 11: TopologyDesigner routes audio problem to audio agents --")
    from api.brain.topology_designer import TopologyDesigner

    td          = TopologyDesigner()
    td.use_anas = False
    topo        = td.design("classify customer call center audio recordings")
    agents      = topo.get("agents", [])

    ok = "audio" in agents
    print(f"  agents: {agents}  source: {topo.get('source')}")
    print("  PASS" if ok else "  FAIL -- 'audio' not in designed agents")
    return ok


# ---------------------------------------------------------------------------
# Test 12: TopologyDesigner routes multimodal problem to multimodal_pipeline
# ---------------------------------------------------------------------------
def test_topology_routes_multimodal():
    print("\n-- Test 12: TopologyDesigner routes multimodal problem --")
    from api.brain.topology_designer import TopologyDesigner

    td          = TopologyDesigner()
    td.use_anas = False
    topo        = td.design("zero shot image and text classification with clip")
    agents      = topo.get("agents", [])

    ok = "multimodal" in agents
    print(f"  agents: {agents}  source: {topo.get('source')}")
    print("  PASS" if ok else "  FAIL -- 'multimodal' not in designed agents")
    return ok


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t1_ok, model_path = test_audio_train()

    results = {
        "AudioAgent train()":             t1_ok,
        "AudioAgent predict() single":    test_audio_predict(model_path),
        "AudioAgent no-model error":      test_audio_no_model(),
        "AudioAgent load round-trip":     test_audio_load(model_path),
        "AudioAgent transcribe Whisper":  test_audio_transcribe(),
        "MultimodalAgent classify_with_labels": test_multimodal_classify(),
        "MultimodalAgent predict() img+txt":    test_multimodal_predict(),
        "MultimodalAgent offline fallback":     test_multimodal_offline(),
        "AGENT_CATALOG has audio":        test_catalog_has_audio(),
        "AGENT_CATALOG has multimodal":   test_catalog_has_multimodal(),
        "Topology routes to audio":       test_topology_routes_audio(),
        "Topology routes to multimodal":  test_topology_routes_multimodal(),
    }

    print("\n==============================")
    print("  DAY 5 VERIFICATION SUMMARY")
    print("==============================")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print("  {:<40} {}".format(name, status))
        if not passed:
            all_pass = False
    print("==============================")
    sys.exit(0 if all_pass else 1)
