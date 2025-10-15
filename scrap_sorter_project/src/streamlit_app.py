import streamlit as st
import cv2, time, os
from ultralytics import YOLO
from PIL import Image

st.set_page_config(layout='wide', page_title='Scrap Sorter Dashboard')

st.title('Scrap Sorter — Live Dashboard (Simulated)')
col1, col2 = st.columns([2,1])

weights = st.sidebar.text_input('Weights path', 'yolov8n.pt')
source = st.sidebar.text_input('Image folder (for demo)', 'dataset/images/train')
conf = st.sidebar.slider('Confidence threshold', 0.0, 1.0, 0.25)

model = YOLO(weights)
model.conf = conf

if st.button('Run once'):
    imgs = sorted([os.path.join(source,f) for f in os.listdir(source) if f.lower().endswith('.jpg')])
    counts = {}
    for p in imgs:
        frame = cv2.imread(p)
        results = model(frame)
        for r in results:
            if r.boxes is None: continue
            for cls in r.boxes.cls.cpu().numpy():
                name = model.names[int(cls)]
                counts[name] = counts.get(name,0) + 1
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(img, caption=p, use_column_width=True)
    st.sidebar.write('Counts:')
    st.sidebar.json(counts)