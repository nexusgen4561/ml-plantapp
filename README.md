# Plant Monitoring / ResNet Inference App

Workspace layout:

```
api/app.py              Flask application (serves HTML + inference + upload)
templates/index.html    Front-end (auto refreshes /image every 2s)
static/images/          Stored last uploaded frame (capture.jpg) + sample folders
model/best_model_cnn.pth  Model checkpoint (TorchScript or state_dict)
vercel.json             Vercel routing config
requirements.txt        Base web dependencies (ML libs not listed)
```

Main application: [api/app.py](api/app.py)  
Front-end template: [templates/index.html](templates/index.html)  
Model file: [model/best_model_cnn.pth](model/best_model_cnn.pth)

## 1. Environment Setup

Create virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install web deps:

```bash
pip install -r requirements.txt
```

Install ML deps (not in requirements.txt to keep deployments lighter; needed locally):

```bash
pip install torch torchvision pillow opencv-python
```

(If Apple Silicon, you may need specific torch wheels.)

## 2. Running Locally

Option A (direct):

```bash
python api/app.py
```

Option B (Flask runner):

```bash
export FLASK_APP=api/app.py
flask run
```

Default listens on http://0.0.0.0:5000