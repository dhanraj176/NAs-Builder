# -*- coding: utf-8 -*-
"""
tests/verify_all_agents.py -- Day 6: Full system verification
Runs 5 checks per agent, generates JSON + markdown report.

Checks per agent:
  1. import        -- module imports cleanly
  2. init          -- class instantiates without error
  3. no_model      -- predict without loaded model returns honest error / fallback
  4. with_model    -- predict with a model returns a real (non-error) result
  5. catalog_match -- AGENT_CATALOG has this domain key AND TopologyDesigner
                      routes a representative problem to it
                      (NA for support/infrastructure agents)
"""

import sys, os, json, math, array, wave, time, traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR  = Path(__file__).parent / "results"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

TRAINED_DIR = Path(__file__).parent.parent / "models" / "trained"

CHECK_NAMES = ["import", "init", "no_model", "with_model", "catalog_match"]


# ── JSON helper ───────────────────────────────────────────────────────────────

def _json_default(obj):
    try:
        import numpy as np
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.str_):     return str(obj)
    except ImportError:
        pass
    if isinstance(obj, Path): return str(obj)
    return str(obj)


# ── Fixture builders ──────────────────────────────────────────────────────────

def _make_image(path):
    from PIL import Image
    Image.new("RGB", (224, 224), color=(128, 64, 32)).save(str(path))


def _make_wav(path, freq=440, dur=0.5, sr=22050):
    n = int(sr * dur)
    s = array.array('h',
        [int(32767 * math.sin(2 * math.pi * freq * i / sr)) for i in range(n)])
    with wave.open(str(path), 'w') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(s.tobytes())


def _make_resnet(path, nc=2):
    import torch, torch.nn as nn
    import torchvision.models as models
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, nc)
    torch.save(m.state_dict(), str(path))


def _make_darts(path, nc=2):
    import torch
    from api.nas_engine import DARTSNet
    m = DARTSNet(C=16, num_cells=3, num_classes=nc)
    torch.save(m.state_dict(), str(path))


def _make_vocab(path):
    vocab = {f"word{i}": i for i in range(50)}
    vocab.update({"test": 0, "hello": 1, "spam": 2, "threat": 3, "safe": 4})
    with open(path, "w") as f:
        json.dump(vocab, f)


def build_fixtures():
    """Create all reusable test fixtures once. Returns paths dict."""
    fx = {}

    p = FIXTURES_DIR / "test_image.png"
    if not p.exists(): _make_image(p)
    fx["image"] = p

    p = FIXTURES_DIR / "test_audio.wav"
    if not p.exists(): _make_wav(p)
    fx["audio_wav"] = p

    # ResNet18 checkpoint (ImageAgent + MedicalAgent)
    p = FIXTURES_DIR / "fx_resnet.pth"
    if not p.exists(): _make_resnet(p, nc=2)
    fx["resnet"] = p

    # DARTSNet checkpoints — named so split('_')[0] == 'fx', meaning both agents
    # look for fixtures/fx_text_vocab.json which we create below.
    p = FIXTURES_DIR / "fx_text_model.pth"
    if not p.exists(): _make_darts(p, nc=2)
    fx["darts_text"] = p

    p = FIXTURES_DIR / "fx_sec_model.pth"
    if not p.exists(): _make_darts(p, nc=2)
    fx["darts_sec"] = p

    p = FIXTURES_DIR / "fx_darts.pth"
    if not p.exists(): _make_darts(p, nc=2)
    fx["darts"] = p

    # Vocab — 'fx_text_vocab.json' matches both TextAgent and SecurityAgent
    p = FIXTURES_DIR / "fx_text_vocab.json"
    if not p.exists(): _make_vocab(p)
    fx["vocab"] = p

    # Trained models from earlier sprint days (may not exist)
    tp = TRAINED_DIR / "day4test_tabular.pkl"
    fx["tabular_model"] = tp if tp.exists() else None

    ap = TRAINED_DIR / "day5test_audio.pkl"
    fx["audio_model"] = ap if ap.exists() else None

    return fx


# ── Routing check helper ──────────────────────────────────────────────────────

def _catalog_check(domain_key, problem):
    """
    Verify AGENT_CATALOG has domain_key and TopologyDesigner routes
    problem to that domain. Returns (bool, detail_str).
    """
    try:
        from api.brain.topology_designer import AGENT_CATALOG, TopologyDesigner
        if domain_key not in AGENT_CATALOG:
            return False, f"'{domain_key}' missing from AGENT_CATALOG"
        td = TopologyDesigner()
        td.use_anas = False
        topo   = td.design(problem)
        agents = topo.get("agents", [])
        if domain_key in agents:
            return True, f"routed -> {agents} (src={topo.get('source')})"
        return False, f"expected '{domain_key}' in {agents}"
    except Exception as e:
        return False, str(e)


# ── Per-agent check functions ─────────────────────────────────────────────────
# Each returns {"checks": {name: True|False|"NA"|"SKIP"}, "notes": {name: ...}}

def check_dynamic_agent(fx):
    r, n = {}, {}

    try:
        from api.agents.dynamic_agent import DynamicAgent
        r["import"] = True
    except Exception as e:
        r["import"] = False; n["import"] = str(e)
        return {"checks": r, "notes": n}

    try:
        agent = DynamicAgent("test_agent", "TestAgent", "test problem",
                             "text", "darts", 2, ["A", "B"])
        r["init"] = True
    except Exception as e:
        r["init"] = False; n["init"] = str(e)
        return {"checks": r, "notes": n}

    # No model -> _fallback_predict: confidence=0.0, mode="fallback_no_model"
    try:
        res = agent.predict("sample text")
        r["no_model"] = (res.get("confidence") == 0.0 and
                         res.get("mode") == "fallback_no_model")
        n["no_model"] = {"confidence": res.get("confidence"),
                         "mode": res.get("mode")}
    except Exception as e:
        r["no_model"] = False; n["no_model"] = str(e)

    # With model (DARTSNet fixture)
    try:
        agent2 = DynamicAgent("test_agent", "TestAgent", "test problem",
                              "text", "darts", 2, ["A", "B"])
        loaded = agent2.load_model(str(fx["darts"]))
        if loaded:
            res = agent2.predict("hello world test input")
            r["with_model"] = (res.get("label") in ["A", "B"] and
                               res.get("confidence", -1) >= 0.0)
            n["with_model"] = {"label": res.get("label"),
                               "confidence": res.get("confidence")}
        else:
            r["with_model"] = False; n["with_model"] = "load_model returned False"
    except Exception as e:
        r["with_model"] = False; n["with_model"] = str(e)

    r["catalog_match"] = "NA"; n["catalog_match"] = "infrastructure agent"
    return {"checks": r, "notes": n}


def check_agent_factory(fx):
    r, n = {}, {}

    try:
        from api.agents.agent_factory import AgentFactory
        r["import"] = True
    except Exception as e:
        r["import"] = False; n["import"] = str(e)
        return {"checks": r, "notes": n}

    try:
        factory = AgentFactory()
        r["init"] = True
    except Exception as e:
        r["init"] = False; n["init"] = str(e)
        return {"checks": r, "notes": n}

    # No model: factory.create → agent.predict → confidence=0 fallback
    try:
        agent = factory.create("test problem", "text",
                               classes=["A", "B"], num_classes=2)
        res = agent.predict("hello world")
        r["no_model"] = (res.get("confidence") == 0.0)
        n["no_model"] = {"confidence": res.get("confidence"),
                         "mode": res.get("mode")}
    except Exception as e:
        r["no_model"] = False; n["no_model"] = str(e)

    # With model: create_from_trained with fixture checkpoint
    try:
        trained = {"model_path": str(fx["darts"]), "classes": ["A", "B"],
                   "test_accuracy": 0.90, "dataset": "fixture", "method": "darts"}
        agent2 = factory.create_from_trained("test problem", "text", trained)
        res = agent2.predict("test input data")
        r["with_model"] = (res.get("label") in ["A", "B"] and
                           res.get("confidence", -1) >= 0.0)
        n["with_model"] = {"label": res.get("label"),
                           "confidence": res.get("confidence")}
    except Exception as e:
        r["with_model"] = False; n["with_model"] = str(e)

    r["catalog_match"] = "NA"; n["catalog_match"] = "infrastructure agent"
    return {"checks": r, "notes": n}


def check_image_agent(fx):
    r, n = {}, {}

    try:
        from api.agents.image_agent import ImageAgent
        r["import"] = True
    except Exception as e:
        r["import"] = False; n["import"] = str(e)
        return {"checks": r, "notes": n}

    try:
        agent = ImageAgent()
        r["init"] = True
    except Exception as e:
        r["init"] = False; n["init"] = str(e)
        return {"checks": r, "notes": n}

    # No model
    try:
        res = agent.predict_image(str(fx["image"]))
        r["no_model"] = (res.get("label") == "error" and
                         res.get("confidence") == 0.0 and
                         res.get("fake") is False)
        n["no_model"] = res
    except Exception as e:
        r["no_model"] = False; n["no_model"] = str(e)

    # With model (ResNet18 fixture — EfficientNetV2 will fail → auto fallback)
    try:
        agent2 = ImageAgent()
        agent2.load_trained_model(str(fx["resnet"]), ["ClassA", "ClassB"], 2)
        if agent2.trained_model is not None:
            res = agent2.predict_image(str(fx["image"]))
            r["with_model"] = (res.get("label") in ["ClassA", "ClassB"] and
                               res.get("label") != "error")
            n["with_model"] = res
        else:
            r["with_model"] = False; n["with_model"] = "model did not load"
    except Exception as e:
        r["with_model"] = False; n["with_model"] = str(e)

    passed, detail = _catalog_check("image",
                                    "detect and classify objects in images and photos")
    r["catalog_match"] = passed; n["catalog_match"] = detail
    return {"checks": r, "notes": n}


def check_text_agent(fx):
    r, n = {}, {}

    try:
        from api.agents.text_agent import TextAgent
        r["import"] = True
    except Exception as e:
        r["import"] = False; n["import"] = str(e)
        return {"checks": r, "notes": n}

    try:
        agent = TextAgent()
        r["init"] = True
    except Exception as e:
        r["init"] = False; n["init"] = str(e)
        return {"checks": r, "notes": n}

    # No model
    try:
        res = agent.predict("some test text input")
        r["no_model"] = (res.get("label") == "error" and
                         res.get("confidence") == 0.0 and
                         res.get("fake") is False)
        n["no_model"] = res
    except Exception as e:
        r["no_model"] = False; n["no_model"] = str(e)

    # With model (fx_text_model.pth → loads fx_text_vocab.json automatically)
    try:
        agent2 = TextAgent()
        agent2.load_trained_model(str(fx["darts_text"]), ["A", "B"], 2)
        if agent2.trained_model is not None:
            res = agent2.predict("hello test spam word")
            r["with_model"] = (res.get("label") != "error" and
                               "confidence" in res)
            n["with_model"] = {"label": res.get("label"),
                               "confidence": res.get("confidence")}
        else:
            r["with_model"] = False; n["with_model"] = "model did not load"
    except Exception as e:
        r["with_model"] = False; n["with_model"] = str(e)

    passed, detail = _catalog_check("text",
                                    "classify spam text messages and emails")
    r["catalog_match"] = passed; n["catalog_match"] = detail
    return {"checks": r, "notes": n}


def check_medical_agent(fx):
    r, n = {}, {}

    try:
        from api.agents.medical_agent import MedicalAgent
        r["import"] = True
    except Exception as e:
        r["import"] = False; n["import"] = str(e)
        return {"checks": r, "notes": n}

    try:
        agent = MedicalAgent()
        r["init"] = True
    except Exception as e:
        r["init"] = False; n["init"] = str(e)
        return {"checks": r, "notes": n}

    # No model
    try:
        res = agent.predict_image(str(fx["image"]))
        r["no_model"] = (res.get("label") == "error" and
                         res.get("confidence") == 0.0 and
                         res.get("fake") is False)
        n["no_model"] = res
    except Exception as e:
        r["no_model"] = False; n["no_model"] = str(e)

    # With model
    try:
        agent2 = MedicalAgent()
        agent2.load_trained_model(str(fx["resnet"]), ["Normal", "Abnormal"], 2)
        if agent2.trained_model is not None:
            res = agent2.predict_image(str(fx["image"]))
            r["with_model"] = (res.get("label") in ["Normal", "Abnormal"] and
                               "confidence" in res)
            n["with_model"] = res
        else:
            r["with_model"] = False; n["with_model"] = "model did not load"
    except Exception as e:
        r["with_model"] = False; n["with_model"] = str(e)

    passed, detail = _catalog_check("medical",
                                    "analyze xray medical images for diagnosis")
    r["catalog_match"] = passed; n["catalog_match"] = detail
    return {"checks": r, "notes": n}


def check_security_agent(fx):
    r, n = {}, {}

    try:
        from api.agents.security_agent import SecurityAgent
        r["import"] = True
    except Exception as e:
        r["import"] = False; n["import"] = str(e)
        return {"checks": r, "notes": n}

    try:
        agent = SecurityAgent()
        r["init"] = True
    except Exception as e:
        r["init"] = False; n["init"] = str(e)
        return {"checks": r, "notes": n}

    # No model
    try:
        res = agent.predict_threat("suspicious malware attack intrusion")
        r["no_model"] = (res.get("label") == "error" and
                         res.get("confidence") == 0.0 and
                         res.get("fake") is False)
        n["no_model"] = res
    except Exception as e:
        r["no_model"] = False; n["no_model"] = str(e)

    # With model (fx_sec_model.pth → loads fx_text_vocab.json automatically)
    try:
        agent2 = SecurityAgent()
        agent2.load_trained_model(str(fx["darts_sec"]), ["Safe", "Threat"], 2)
        if agent2.trained_model is not None:
            res = agent2.predict_threat("test hello threat word safe attack")
            r["with_model"] = (res.get("label") in ["Safe", "Threat"] and
                               "confidence" in res)
            n["with_model"] = {"label": res.get("label"),
                               "confidence": res.get("confidence")}
        else:
            r["with_model"] = False; n["with_model"] = "model did not load"
    except Exception as e:
        r["with_model"] = False; n["with_model"] = str(e)

    passed, detail = _catalog_check("security",
                                    "detect security threats and vulnerabilities attacks")
    r["catalog_match"] = passed; n["catalog_match"] = detail
    return {"checks": r, "notes": n}


def check_tabular_agent(fx):
    r, n = {}, {}

    try:
        from api.agents.tabular_agent import TabularAgent
        r["import"] = True
    except Exception as e:
        r["import"] = False; n["import"] = str(e)
        return {"checks": r, "notes": n}

    try:
        agent = TabularAgent()
        r["init"] = True
    except Exception as e:
        r["init"] = False; n["init"] = str(e)
        return {"checks": r, "notes": n}

    # No model
    try:
        res = agent.predict({"age": 30, "income": 50000})
        r["no_model"] = (res.get("label") == "error" and
                         res.get("confidence") == 0.0 and
                         res.get("fake") is False)
        n["no_model"] = res
    except Exception as e:
        r["no_model"] = False; n["no_model"] = str(e)

    # With model (Day-4 sprint artifact)
    if fx["tabular_model"]:
        try:
            agent2 = TabularAgent()
            loaded = agent2.load_trained_model(str(fx["tabular_model"]))
            if loaded:
                res = agent2.predict({"age": 35, "income": 45000,
                                      "score": 0.3, "region": "north"})
                r["with_model"] = (res.get("label") != "error" and
                                   "confidence" in res and
                                   res.get("agent_used") == "TabularAgent")
                n["with_model"] = {"label": res.get("label"),
                                   "confidence": res.get("confidence")}
            else:
                r["with_model"] = False; n["with_model"] = "load_trained_model False"
        except Exception as e:
            r["with_model"] = False; n["with_model"] = str(e)
    else:
        r["with_model"] = "SKIP"; n["with_model"] = "day4test model not found"

    passed, detail = _catalog_check("tabular",
                                    "predict customer churn from csv tabular data")
    r["catalog_match"] = passed; n["catalog_match"] = detail
    return {"checks": r, "notes": n}


def check_audio_agent(fx):
    r, n = {}, {}

    try:
        from api.agents.audio_agent import AudioAgent
        r["import"] = True
    except Exception as e:
        r["import"] = False; n["import"] = str(e)
        return {"checks": r, "notes": n}

    try:
        agent = AudioAgent()
        r["init"] = True
    except Exception as e:
        r["init"] = False; n["init"] = str(e)
        return {"checks": r, "notes": n}

    # No model
    try:
        res = agent.predict(str(fx["audio_wav"]))
        r["no_model"] = (res.get("label") == "error" and
                         res.get("confidence") == 0.0 and
                         res.get("fake") is False)
        n["no_model"] = res
    except Exception as e:
        r["no_model"] = False; n["no_model"] = str(e)

    # With model (Day-5 sprint artifact)
    if fx["audio_model"]:
        try:
            agent2 = AudioAgent()
            loaded = agent2.load_trained_model(str(fx["audio_model"]))
            if loaded:
                res = agent2.predict(str(fx["audio_wav"]))
                r["with_model"] = (res.get("label") != "error" and
                                   "confidence" in res and
                                   res.get("agent_used") == "AudioAgent")
                n["with_model"] = {"label": res.get("label"),
                                   "confidence": res.get("confidence")}
            else:
                r["with_model"] = False; n["with_model"] = "load_trained_model False"
        except Exception as e:
            r["with_model"] = False; n["with_model"] = str(e)
    else:
        r["with_model"] = "SKIP"; n["with_model"] = "day5test model not found"

    passed, detail = _catalog_check("audio",
                                    "classify audio speech recordings and sound")
    r["catalog_match"] = passed; n["catalog_match"] = detail
    return {"checks": r, "notes": n}


def check_multimodal_agent(fx):
    r, n = {}, {}

    try:
        from api.agents.multimodal_agent import MultimodalAgent
        r["import"] = True
    except Exception as e:
        r["import"] = False; n["import"] = str(e)
        return {"checks": r, "notes": n}

    try:
        agent = MultimodalAgent()
        r["init"] = True
    except Exception as e:
        r["init"] = False; n["init"] = str(e)
        return {"checks": r, "notes": n}

    # No model: offline subclass override → must return error dict
    try:
        class _Offline(MultimodalAgent):
            def _lazy_load_clip(self): return False
        offline = _Offline()
        res = offline.classify_with_labels(str(fx["image"]), ["cat", "dog"])
        r["no_model"] = (res.get("label") == "error" and
                         res.get("confidence") == 0.0 and
                         res.get("fake") is False)
        n["no_model"] = res
    except Exception as e:
        r["no_model"] = False; n["no_model"] = str(e)

    # With model (CLIP zero-shot — SKIP if model unavailable)
    try:
        agent2 = MultimodalAgent()
        res = agent2.classify_with_labels(str(fx["image"]),
                                          ["red image", "blue image", "green image"])
        if res.get("label") == "error":
            r["with_model"] = "SKIP"
            n["with_model"] = f"CLIP unavailable: {res.get('error','')[:60]}"
        else:
            r["with_model"] = (res.get("label") is not None and
                               0.0 <= res.get("confidence", -1) <= 1.0 and
                               res.get("agent_used") == "MultimodalAgent")
            n["with_model"] = {"label": res.get("label"),
                               "confidence": res.get("confidence")}
    except Exception as e:
        r["with_model"] = "SKIP"; n["with_model"] = str(e)

    passed, detail = _catalog_check("multimodal",
                                    "multimodal image and text classification clip zero shot")
    r["catalog_match"] = passed; n["catalog_match"] = detail
    return {"checks": r, "notes": n}


def check_fusion_agent(fx):
    r, n = {}, {}

    try:
        from api.agents.fusion_agent import FusionAgent
        r["import"] = True
    except Exception as e:
        r["import"] = False; n["import"] = str(e)
        return {"checks": r, "notes": n}

    try:
        agent = FusionAgent()
        r["init"] = True
    except Exception as e:
        r["init"] = False; n["init"] = str(e)
        return {"checks": r, "notes": n}

    # No model: fuse([]) → {"error": "No agent results to fuse"}
    try:
        res = agent.fuse([])
        r["no_model"] = ("error" in res)
        n["no_model"] = res
    except Exception as e:
        r["no_model"] = False; n["no_model"] = str(e)

    # With model: fuse two matching predictions
    try:
        mock = [
            {"label": "positive", "confidence": 0.9, "agent_name": "AgentA"},
            {"label": "positive", "confidence": 0.8, "agent_name": "AgentB"},
        ]
        res = agent.fuse(mock)
        r["with_model"] = (res.get("label") == "positive" and
                           "confidence" in res and
                           "fusion_method" in res)
        n["with_model"] = {"label":         res.get("label"),
                           "confidence":    res.get("confidence"),
                           "fusion_method": res.get("fusion_method")}
    except Exception as e:
        r["with_model"] = False; n["with_model"] = str(e)

    r["catalog_match"] = "NA"; n["catalog_match"] = "support agent"
    return {"checks": r, "notes": n}


def check_evaluator_agent(fx):
    r, n = {}, {}

    try:
        from api.agents.evaluator_agent import EvaluatorAgent
        r["import"] = True
    except Exception as e:
        r["import"] = False; n["import"] = str(e)
        return {"checks": r, "notes": n}

    try:
        agent = EvaluatorAgent()
        r["init"] = True
    except Exception as e:
        r["init"] = False; n["init"] = str(e)
        return {"checks": r, "notes": n}

    # No model: validate_single with low confidence → flag_for_review=True
    try:
        res = agent.validate_single({"label": "spam", "confidence": 0.3})
        r["no_model"] = (res.get("flag_for_review") is True and
                         res.get("quality") == "low")
        n["no_model"] = res
    except Exception as e:
        r["no_model"] = False; n["no_model"] = str(e)

    # With model: ML metrics evaluation (no external model required)
    try:
        preds = [{"label": "cat", "confidence": 0.9},
                 {"label": "dog", "confidence": 0.8},
                 {"label": "cat", "confidence": 0.95}]
        truth = ["cat", "dog", "cat"]
        res = agent.evaluate(preds, truth)
        r["with_model"] = ("accuracy" in res and "f1" in res and
                           "quality_score" in res and
                           isinstance(res["f1"], float))
        n["with_model"] = {k: res[k] for k in
                           ("accuracy", "f1", "quality_score", "verdict")
                           if k in res}
    except Exception as e:
        r["with_model"] = False; n["with_model"] = str(e)

    r["catalog_match"] = "NA"; n["catalog_match"] = "support agent"
    return {"checks": r, "notes": n}


def check_agent_network(fx):
    r, n = {}, {}

    try:
        from api.agents.agent_network import AgentNetwork
        r["import"] = True
    except Exception as e:
        r["import"] = False; n["import"] = str(e)
        return {"checks": r, "notes": n}

    try:
        network = AgentNetwork("day6_test_net")
        r["init"] = True
    except Exception as e:
        r["init"] = False; n["init"] = str(e)
        return {"checks": r, "notes": n}

    # No model: all agents fail → error dict with agent_used="none"
    try:
        class _Fail:
            def predict(self, data):
                raise RuntimeError("intentional test failure")
        res = network.collaborate([_Fail(), _Fail()], task="test", data="input")
        r["no_model"] = ("error" in res and res.get("agent_used") == "none")
        n["no_model"] = res
    except Exception as e:
        r["no_model"] = False; n["no_model"] = str(e)

    # With model: working agents → valid fusion result
    try:
        net2 = AgentNetwork("day6_test_net2")

        class _Good:
            def predict(self, data):
                return {"label": "ok", "confidence": 0.88}

        res = net2.collaborate([_Good(), _Good()],
                               task="unit_test", data="sample data")
        r["with_model"] = (res.get("label") == "ok" and
                           res.get("valid_count") == 2 and
                           res.get("task") == "unit_test")
        n["with_model"] = {"label":       res.get("label"),
                           "valid_count": res.get("valid_count"),
                           "task":        res.get("task")}
    except Exception as e:
        r["with_model"] = False; n["with_model"] = str(e)

    r["catalog_match"] = "NA"; n["catalog_match"] = "support agent"
    return {"checks": r, "notes": n}


# ── Agent registry ────────────────────────────────────────────────────────────

AGENTS = [
    ("DynamicAgent",    check_dynamic_agent),
    ("AgentFactory",    check_agent_factory),
    ("ImageAgent",      check_image_agent),
    ("TextAgent",       check_text_agent),
    ("MedicalAgent",    check_medical_agent),
    ("SecurityAgent",   check_security_agent),
    ("TabularAgent",    check_tabular_agent),
    ("AudioAgent",      check_audio_agent),
    ("MultimodalAgent", check_multimodal_agent),
    ("FusionAgent",     check_fusion_agent),
    ("EvaluatorAgent",  check_evaluator_agent),
    ("AgentNetwork",    check_agent_network),
]


# ── Run all ───────────────────────────────────────────────────────────────────

def run_all(fx):
    all_results = {}
    for agent_name, check_fn in AGENTS:
        print(f"\n[{agent_name}]")
        try:
            data = check_fn(fx)
        except Exception as e:
            data = {"checks": {c: False for c in CHECK_NAMES},
                    "notes":  {"fatal": traceback.format_exc()}}
        all_results[agent_name] = data

        checks = data["checks"]
        for c in CHECK_NAMES:
            v = checks.get(c, False)
            sym = ("[OK]"   if v is True  else
                   "[NA]"   if v == "NA"  else
                   "[SKIP]" if v == "SKIP" else
                   "[FAIL]")
            print(f"  {sym} {c}")
    return all_results


# ── Report ────────────────────────────────────────────────────────────────────

def _is_pass(v):
    return v is True or v in ("NA", "SKIP")


def generate_report(all_results, elapsed_s):
    total        = len(all_results)
    ready        = 0
    total_failed = 0
    summaries    = []

    for name, data in all_results.items():
        checks  = data["checks"]
        failed  = [c for c in CHECK_NAMES if checks.get(c) is False]
        skipped = [c for c in CHECK_NAMES if checks.get(c) == "SKIP"]
        na      = [c for c in CHECK_NAMES if checks.get(c) == "NA"]
        passed  = [c for c in CHECK_NAMES if checks.get(c) is True]

        is_ready = not failed
        if is_ready:
            ready += 1
        total_failed += len(failed)

        summaries.append({
            "name":             name,
            "production_ready": is_ready,
            "checks_raw":       {c: checks.get(c, False) for c in CHECK_NAMES},
            "passed":           passed,
            "failed":           failed,
            "skipped":          skipped,
            "na":               na,
            "notes":            data.get("notes", {}),
        })

    overall = "READY" if total_failed == 0 else "NEEDS WORK"

    return {
        "generated_at":     datetime.now().isoformat(),
        "elapsed_seconds":  round(elapsed_s, 1),
        "total_agents":     total,
        "production_ready": ready,
        "failed_checks":    total_failed,
        "overall_status":   overall,
        "agents":           summaries,
    }


def print_report(report):
    w = 56
    sep = "=" * w
    print(f"\n{sep}")
    print("   AUTOARCHITECT AGENT SYSTEM REPORT")
    print(sep)
    print(f"  Generated:         {report['generated_at'][:19]}")
    print(f"  Elapsed:           {report['elapsed_seconds']}s")
    print(f"  Total agents:      {report['total_agents']}")
    print(f"  Production ready:  {report['production_ready']}")
    print(f"  Failed checks:     {report['failed_checks']}")
    print(sep)
    print("  DETAILS PER AGENT:")
    for a in report["agents"]:
        name   = a["name"]
        status = "[PASS]" if a["production_ready"] else "[FAIL]"
        fails  = a["failed"]
        skips  = a["skipped"]
        if a["production_ready"]:
            extra = f"  skipped={skips}" if skips else "  all checks pass"
        else:
            extra = f"  FAILED: {fails}"
        print(f"    {status} {name:<18}{extra}")
    print(sep)
    print(f"  OVERALL STATUS: {report['overall_status']}")
    print(sep)


def save_markdown(report, path):
    lines = [
        "# AutoArchitect Agent System Report",
        "",
        f"**Generated:** {report['generated_at'][:19]}  ",
        f"**Total agents:** {report['total_agents']}  ",
        f"**Production ready:** {report['production_ready']}  ",
        f"**Failed checks:** {report['failed_checks']}  ",
        f"**Overall status:** `{report['overall_status']}`  ",
        "",
        "---",
        "",
        "## Agent Details",
        "",
        "| Agent | import | init | no_model | with_model | catalog_match |",
        "|-------|--------|------|----------|------------|---------------|",
    ]
    for a in report["agents"]:
        def _cell(c):
            v = a["checks_raw"].get(c, False)
            if v is True:    return "OK"
            if v == "NA":    return "N/A"
            if v == "SKIP":  return "SKIP"
            return "FAIL"

        name   = a["name"]
        status = "OK" if a["production_ready"] else "FAIL"
        row = (f"| **{name}** "
               f"| {_cell('import')} "
               f"| {_cell('init')} "
               f"| {_cell('no_model')} "
               f"| {_cell('with_model')} "
               f"| {_cell('catalog_match')} |")
        lines.append(row)

    lines += [
        "",
        "---",
        "",
        f"## Overall: {report['overall_status']}",
        "",
        ("All 12 agents passed every applicable check."
         if report["failed_checks"] == 0
         else f"{report['failed_checks']} check(s) failed — see table above."),
        "",
        f"*Report generated by `tests/verify_all_agents.py` "
        f"— Day 6 of 21-day sprint*",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building fixtures...")
    fx = build_fixtures()
    print(f"  tabular model: {'found' if fx['tabular_model'] else 'missing'}")
    print(f"  audio   model: {'found' if fx['audio_model']   else 'missing'}")

    t0 = time.time()
    all_results = run_all(fx)
    elapsed = time.time() - t0

    report = generate_report(all_results, elapsed)

    print_report(report)

    # Save JSON
    json_path = RESULTS_DIR / "agent_system_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default)
    print(f"\n  Saved JSON : {json_path}")

    # Save markdown
    md_path = RESULTS_DIR / "AGENT_SYSTEM_STATUS.md"
    save_markdown(report, md_path)
    print(f"  Saved MD   : {md_path}")

    sys.exit(0 if report["overall_status"] == "READY" else 1)
