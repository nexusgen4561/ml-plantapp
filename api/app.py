import os, time, random, logging, threading
from typing import Optional, Tuple
from flask import Flask, render_template, jsonify, url_for, redirect, send_from_directory, abort, Response, request

logging.basicConfig(level=logging.INFO)

# Paths
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))  # project root
STATIC_DIR = os.path.join(ROOT_DIR, 'static')
TEMPLATE_DIR = os.path.join(ROOT_DIR, 'templates')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(MODEL_DIR, 'best_model_cnn.pth'))

# Flask
app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATE_DIR)

# Labels (edit to match your trained classes)
LABELS = ["basil", "coriander", "lettuce", "unrecognized"]

MODEL_ARCH = os.getenv("MODEL_ARCH")  # optional override: resnet18|resnet50

# ===== Model loading =====
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms as T, models
    from PIL import Image
    import cv2
    import numpy as np
except Exception as e:
    logging.error("Missing ML dependencies. Install: torch torchvision pillow opencv-python")
    raise

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _create_model(arch: str, num_classes: int):
    """
    Create a ResNet backbone matching the checkpoint (resnet18 or resnet50).
    """
    if arch == "resnet50":
        m = models.resnet50(weights=None)
    else:
        m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m

def _detect_resnet_arch_from_sd(sd: dict) -> str:
    """
    Heuristic: bottleneck ResNets (50/101/152) have 2048-d features at layer4,
    with 1x1 convs in bottleneck blocks. Basic (18/34) do not.
    """
    # Check BN in layer4 downsample (size 2048 => bottleneck)
    bn = sd.get("layer4.0.downsample.1.weight")
    if isinstance(bn, torch.Tensor) and bn.ndim == 1 and bn.shape[0] == 2048:
        return "resnet50"
    # Check conv1 kernel in layer4.0 (1x1 => bottleneck)
    c = sd.get("layer4.0.conv1.weight")
    if isinstance(c, torch.Tensor) and c.ndim == 4 and tuple(c.shape[2:]) == (1, 1):
        return "resnet50"
    return "resnet18"

def _clean_state_dict(sd: dict) -> dict:
    # unwrap {"state_dict": ...}
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    # strip common prefixes
    cleaned = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = v
    return cleaned

def load_model(model_path: str, num_classes: int):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    logging.info("Loading model from %s", model_path)

    # Try TorchScript first
    try:
        model = torch.jit.load(model_path, map_location=_device)
        model.eval().to(_device)
        logging.info("Loaded TorchScript model")
        return model
    except Exception:
        logging.info("Not a TorchScript model; loading state_dict...")

    # Load state_dict
    raw = torch.load(model_path, map_location=_device)
    sd = _clean_state_dict(raw)

    # Decide architecture
    arch = MODEL_ARCH or _detect_resnet_arch_from_sd(sd)
    logging.info("Detected/selected architecture: %s", arch)

    # Build model and load weights (ignore fc if shapes differ)
    model = _create_model(arch, num_classes)
    # Drop classifier weights from sd to avoid size mismatch
    for k in list(sd.keys()):
        if k.startswith("fc."):
            sd.pop(k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logging.info("load_state_dict strict=False -> missing=%s unexpected=%s", missing, unexpected)

    model.eval().to(_device)
    logging.info("Model ready on %s", _device)
    return model

_model = load_model(MODEL_PATH, num_classes=len(LABELS))

# Preprocessing (adjust to your training pipeline)
_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def infer_image_bgr(img_bgr: np.ndarray) -> Tuple[str, float]:
    """
    img_bgr: OpenCV frame in BGR
    returns (label, confidence)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    x = _transform(pil).unsqueeze(0).to(_device)
    with torch.inference_mode():
        logits = _model(x)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    idx = int(probs.argmax())
    return LABELS[idx], float(probs[idx])

# Replace previous camera prediction globals with upload-based state
_last_pred = {"plant": "Unknown", "confidence": 0.0, "health": "uncertain"}
image_data: Optional[bytes] = None
IMAGE_FILENAME = "capture.jpg"
IMAGE_PATH = os.path.join(STATIC_DIR, "images", IMAGE_FILENAME)

def _infer_bytes(jpeg_bytes: bytes):
    try:
        import numpy as np, cv2
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Decode failed")
        label, conf = infer_image_bgr(img_bgr)
        return label, conf
    except Exception as e:
        logging.exception("Inference on uploaded bytes failed: %s", e)
        return "Unknown", 0.0

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/image")
def get_image():
    global image_data
    if image_data:
        return Response(image_data, mimetype="image/jpeg")
    if os.path.exists(IMAGE_PATH):
        with open(IMAGE_PATH, "rb") as f:
            return Response(f.read(), mimetype="image/jpeg")
    return "No image", 404

@app.route("/upload", methods=["POST"])
def upload_image():
    """
    Receives raw JPEG bytes from ESP/Orange Pi.
    Performs inference immediately and stores prediction.
    """
    global image_data, _last_pred
    image_data = request.data
    if not image_data:
        return "Empty payload", 400
    try:
        os.makedirs(os.path.join(STATIC_DIR, "images"), exist_ok=True)
        with open(IMAGE_PATH, "wb") as f:
            f.write(image_data)
        label, conf = _infer_bytes(image_data)
        _last_pred = {
            "plant": label,
            "confidence": conf,
            "health": "healthy" if conf >= 0.7 else "uncertain"
        }
        logging.info("Upload received (%d bytes). Pred: %s %.1f%%", len(image_data), label, conf*100)
        return "OK", 200
    except Exception as e:
        logging.exception("Save/infer failed: %s", e)
        return "ERROR", 500

@app.route("/api/plant")
def plant():
    return jsonify(_last_pred)

# Remove /video_feed and camera endpoints; keep health + routes
@app.route("/health")
def health():
    return jsonify({"ok": True})

@app.route("/__routes")
def list_routes():
    rules = []
    for r in app.url_map.iter_rules():
        rules.append({
            "rule": str(r),
            "methods": sorted(m for m in r.methods if m not in {"HEAD","OPTIONS"}),
            "endpoint": r.endpoint
        })
    return jsonify({"routes": rules})

if __name__ == "__main__":
    for r in app.url_map.iter_rules():
        logging.info("Route: %-30s methods=%s endpoint=%s", r.rule, ",".join(sorted(r.methods)), r.endpoint)
    app.run(host="0.0.0.0", port=5000, debug=True)