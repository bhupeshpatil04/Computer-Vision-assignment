# Real-Time Scrap Classifier & Robotic Pick Simulation (Mini)

## What’s included
- `src/` : main scripts
  - `train_yolov8.py` : training stub (uses ultralytics YOLO)
  - `infer_realtime.py` : runs inference on a folder of images (simulated conveyor) or webcam
  - `streamlit_app.py` : simple Streamlit dashboard to display counts and frames
- `dataset/` : small synthetic sample dataset (5 images) in YOLO format
- `docs/writeup.pdf` : short write-up
- `requirements.txt` : Python dependencies

## Quick start (local)
1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate      # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. (Optional) Train a YOLOv8 model:
```bash
# requires ultralytics package and GPU for reasonable speed
python src/train_yolov8.py --data dataset/dataset.yaml --epochs 10 --imgsz 640
```
You can also start from `yolov8n.pt` pretrained weights (Ultralytics will auto-download).

3. Run real-time inference (folder mode):
```bash
python src/infer_realtime.py --source dataset/images/train --weights yolov8n.pt
```

4. Run Streamlit dashboard:
```bash
streamlit run src/streamlit_app.py
```

## Notes
- The repo includes a tiny synthetic dataset so you can run the inference script right away.
- `train_yolov8.py` is a convenience wrapper around Ultralytics' training API. If you don't want to train, set `--weights yolov8n.pt` in the inference step.
- See `docs/writeup.pdf` for the assignment write-up and explanations.