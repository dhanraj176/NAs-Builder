# -*- coding: utf-8 -*-
"""
Day 2 verification -- ImageAgent EfficientNetV2-S upgrade + DynamicAgent routing.

Tests:
  1. EfficientNetV2-S architecture loads from torchvision
  2. ImageAgent.predict_image() returns valid prediction (ResNet18 fallback)
  3. DynamicAgent.route_to_agent("image")   -> ImageAgent
  4. DynamicAgent.route_to_agent("medical") -> MedicalAgent
  5. ResNet18 fallback: EfficientNetV2 fails on ResNet18 weights -> ResNet18 loaded
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

TRAINED = os.path.join(os.path.dirname(__file__), "models", "trained")


def _load_classes(path):
    with open(path) as f:
        meta = json.load(f)
    return meta["classes"], meta["num_classes"]


# -- Test 1: EfficientNetV2-S instantiates cleanly ----------------------------
def test_efficientnet_builds():
    print("\n-- Test 1: EfficientNetV2-S architecture loads --")
    try:
        import torch.nn as nn
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model   = efficientnet_v2_s(weights=weights)
        in_f    = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, 2)
        model.eval()
        print(f"  EfficientNetV2-S built, classifier in_features={in_f}")
        print("  PASS")
        return True
    except Exception as e:
        print(f"  FAIL -- {e}")
        return False


# -- Test 2: ImageAgent.predict_image() with loaded model ---------------------
def test_image_agent_predict():
    print("\n-- Test 2: ImageAgent.predict_image() returns valid prediction --")
    from api.agents.image_agent import ImageAgent

    model_path   = os.path.join(TRAINED, "1309624efe_image.pth")
    classes_path = os.path.join(TRAINED, "1309624efe_image_classes.json")

    if not os.path.exists(model_path):
        print("  SKIP -- model file not found")
        return True

    classes, num_classes = _load_classes(classes_path)
    agent = ImageAgent()
    agent.load_trained_model(model_path, classes, num_classes)

    if agent.trained_model is None:
        print("  FAIL -- model did not load")
        return False

    # Create a small test image
    try:
        import io, tempfile
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (64, 64), color=(128, 64, 200))
        tmp = tempfile.mktemp(suffix=".png")
        img.save(tmp)
        result = agent.predict_image(tmp)
        os.remove(tmp)
    except ImportError:
        print("  SKIP -- PIL not available")
        return True

    ok = (isinstance(result, dict)
          and result.get("label") not in (None, "error")
          and isinstance(result.get("confidence"), float)
          and result.get("confidence") >= 0.0)
    print(f"  arch used   : {agent.model_arch}")
    print(f"  prediction  : {result}")
    print("  PASS" if ok else "  FAIL -- unexpected result: " + repr(result))
    return ok


# -- Test 3: DynamicAgent routes "image" -> ImageAgent -----------------------
def test_routing_image():
    print("\n-- Test 3: DynamicAgent routes 'image' -> ImageAgent --")
    from api.agents.dynamic_agent import DynamicAgent
    from api.agents.image_agent   import ImageAgent

    da = DynamicAgent(
        agent_name="test_agent", class_name="TestAgent",
        problem="test", domain="image",
        model_type="resnet18", num_classes=2, classes=["a", "b"]
    )
    routed = da.route_to_agent("image")
    ok = isinstance(routed, ImageAgent)
    print(f"  routed to   : {routed.__class__.__name__}")
    print("  PASS" if ok else "  FAIL -- expected ImageAgent, got " + type(routed).__name__)
    return ok


# -- Test 4: DynamicAgent routes "medical" -> MedicalAgent -------------------
def test_routing_medical():
    print("\n-- Test 4: DynamicAgent routes 'medical' -> MedicalAgent --")
    from api.agents.dynamic_agent  import DynamicAgent
    from api.agents.medical_agent  import MedicalAgent

    da = DynamicAgent(
        agent_name="test_agent", class_name="TestAgent",
        problem="test", domain="image",
        model_type="resnet18", num_classes=2, classes=["a", "b"]
    )
    routed = da.route_to_agent("medical")
    ok = isinstance(routed, MedicalAgent)
    print(f"  routed to   : {routed.__class__.__name__}")
    print("  PASS" if ok else "  FAIL -- expected MedicalAgent, got " + type(routed).__name__)
    return ok


# -- Test 5: Fallback to ResNet18 when EfficientNetV2 weights don't match ----
def test_resnet_fallback():
    print("\n-- Test 5: ResNet18 fallback when EfficientNetV2-S load fails --")
    from api.agents.image_agent import ImageAgent

    model_path   = os.path.join(TRAINED, "1309624efe_image.pth")
    classes_path = os.path.join(TRAINED, "1309624efe_image_classes.json")

    if not os.path.exists(model_path):
        print("  SKIP -- model file not found")
        return True

    classes, num_classes = _load_classes(classes_path)
    agent = ImageAgent()
    agent.load_trained_model(model_path, classes, num_classes)

    # The existing model is ResNet18; EfficientNetV2-S should fail and fall back
    ok = (agent.trained_model is not None
          and agent.model_arch == "resnet18_fallback")
    print(f"  model_arch  : {agent.model_arch}")
    print(f"  model loaded: {agent.trained_model is not None}")
    if ok:
        print("  PASS -- EfficientNetV2-S failed as expected, ResNet18 loaded")
    else:
        print("  FAIL -- unexpected state")
    return ok


# -- Run all ------------------------------------------------------------------
if __name__ == "__main__":
    results = {
        "EfficientNetV2-S builds":     test_efficientnet_builds(),
        "ImageAgent predict_image":    test_image_agent_predict(),
        "Route image -> ImageAgent":   test_routing_image(),
        "Route medical -> MedAgent":   test_routing_medical(),
        "ResNet18 fallback":           test_resnet_fallback(),
    }

    print("\n==============================")
    print("  DAY 2 VERIFICATION SUMMARY")
    print("==============================")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print("  {:<30} {}".format(name, status))
        if not passed:
            all_pass = False

    print("==============================")
    sys.exit(0 if all_pass else 1)
