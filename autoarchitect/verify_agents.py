# -*- coding: utf-8 -*-
"""
Day 1 verification -- confirms all three agents honour the deterministic
error contract:
  - no model loaded  -> error dict (label="error", fake=False)
  - model loaded     -> real prediction OR honest vocab-missing error
Never a fake/random result.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

TRAINED = os.path.join(os.path.dirname(__file__), "models", "trained")


def _load_classes(path):
    with open(path) as f:
        meta = json.load(f)
    return meta["classes"], meta["num_classes"]


def _is_error(result):
    return (isinstance(result, dict)
            and result.get("label") == "error"
            and result.get("fake") is False)


def _is_real_prediction(result):
    return (isinstance(result, dict)
            and result.get("label") not in (None, "error")
            and isinstance(result.get("confidence"), (int, float))
            and result.get("confidence") >= 0.0)


# -- MedicalAgent -------------------------------------------------------------
def test_medical():
    print("\n-- MedicalAgent --")
    from api.agents.medical_agent import MedicalAgent

    agent = MedicalAgent()

    # 1. No model -> predict_image must return error dict
    r1 = agent.predict_image("nonexistent.png")
    ok1 = _is_error(r1)
    print("  no-model predict_image :", r1)
    print("  PASS" if ok1 else "  FAIL -- expected error dict, got " + repr(r1))

    # 2. No model -> _predict_scan must return error dict
    r2 = agent._predict_scan("")
    ok2 = _is_error(r2)
    print("  no-model _predict_scan :", r2)
    print("  PASS" if ok2 else "  FAIL -- expected error dict, got " + repr(r2))

    # 3. Load real model -> _predict_scan must return a real prediction
    model_path   = os.path.join(TRAINED, "3ece8af0e0_medical.pth")
    classes_path = os.path.join(TRAINED, "3ece8af0e0_medical_classes.json")
    if os.path.exists(model_path) and os.path.exists(classes_path):
        classes, num_classes = _load_classes(classes_path)
        agent.load_trained_model(model_path, classes, num_classes)

        try:
            import io, base64
            from PIL import Image as PILImage
            img = PILImage.new("RGB", (64, 64), color=(200, 200, 200))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            r3 = agent._predict_scan(b64)
            ok3 = _is_real_prediction(r3)
            print("  model loaded _predict_scan:", r3)
            print("  PASS" if ok3 else "  FAIL -- expected real prediction, got " + repr(r3))
        except ImportError:
            print("  SKIP loaded-model test (PIL not available)")
            ok3 = True
    else:
        print("  SKIP loaded-model test (model file not found)")
        ok3 = True

    return ok1 and ok2 and ok3


# -- SecurityAgent ------------------------------------------------------------
def test_security():
    print("\n-- SecurityAgent --")
    from api.agents.security_agent import SecurityAgent

    agent = SecurityAgent()

    # 1. No model -> predict_threat must return error dict
    r1 = agent.predict_threat("malicious payload attack")
    ok1 = _is_error(r1)
    print("  no-model predict_threat:", r1)
    print("  PASS" if ok1 else "  FAIL -- expected error dict, got " + repr(r1))

    # 2. Load real model (no vocab available for this security hash)
    model_path   = os.path.join(TRAINED, "0f1ec872e0_security.pth")
    classes_path = os.path.join(TRAINED, "0f1ec872e0_security_classes.json")
    if os.path.exists(model_path) and os.path.exists(classes_path):
        classes, num_classes = _load_classes(classes_path)
        agent.load_trained_model(model_path, classes, num_classes)

        r2 = agent.predict_threat("suspicious login attempt from unknown IP")
        # Real prediction or honest vocab-missing error are both acceptable;
        # a fake/random result (fake=True or no error key when label=="error") is not.
        ok2 = _is_real_prediction(r2) or _is_error(r2)
        print("  model loaded predict_threat:", r2)
        if _is_error(r2):
            print("  PASS -- honest vocab-missing error (no random prediction)")
        elif _is_real_prediction(r2):
            print("  PASS -- real prediction returned")
        else:
            print("  FAIL -- unexpected result:", r2)
            ok2 = False
    else:
        print("  SKIP loaded-model test (model file not found)")
        ok2 = True

    return ok1 and ok2


# -- TextAgent ----------------------------------------------------------------
def test_text():
    print("\n-- TextAgent --")
    from api.agents.text_agent import TextAgent

    agent = TextAgent()

    # 1. No model -> predict must return error dict
    r1 = agent.predict("this headline is completely fabricated")
    ok1 = _is_error(r1)
    print("  no-model predict:", r1)
    print("  PASS" if ok1 else "  FAIL -- expected error dict, got " + repr(r1))

    # 2. Load real model with vocab (71802a8ac0 has both .pth and _vocab.json)
    model_path   = os.path.join(TRAINED, "71802a8ac0_text.pth")
    classes_path = os.path.join(TRAINED, "71802a8ac0_text_classes.json")
    if os.path.exists(model_path) and os.path.exists(classes_path):
        classes, num_classes = _load_classes(classes_path)
        agent.load_trained_model(model_path, classes, num_classes)

        r2 = agent.predict("Scientists discover new vaccine that eliminates cancer")
        ok2 = _is_real_prediction(r2) and "top3_predictions" in r2
        print("  model loaded predict:", r2)
        print("  PASS" if ok2 else "  FAIL -- expected real prediction with top3, got " + repr(r2))
    else:
        print("  SKIP loaded-model test (model file not found)")
        ok2 = True

    return ok1 and ok2


# -- Run all ------------------------------------------------------------------
if __name__ == "__main__":
    results = {
        "MedicalAgent":  test_medical(),
        "SecurityAgent": test_security(),
        "TextAgent":     test_text(),
    }

    print("\n==============================")
    print("  VERIFICATION SUMMARY")
    print("==============================")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print("  {:<18} {}".format(name, status))
        if not passed:
            all_pass = False

    print("==============================")
    sys.exit(0 if all_pass else 1)
