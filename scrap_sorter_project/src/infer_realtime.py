#!/usr/bin/env python3
"""Run inference on webcam or folder of images. Draw bounding boxes and pick points."""
import argparse, time, os
import cv2
from ultralytics import YOLO

def draw_pickpoint(img, box, label):
    x1,y1,x2,y2 = map(int, box)
    cx = int((x1+x2)/2)
    cy = int((y1+y2)/2)
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.drawMarker(img, (cx,cy), (0,0,255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(img, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

def process_frame(model, frame):
    t0 = time.time()
    results = model(frame)
    t1 = time.time()
    latency_ms = (t1-t0)*1000
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        probs = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
        classes = r.boxes.cls.cpu().numpy() if r.boxes is not None else []
        for box,conf,cls in zip(boxes,probs,classes):
            label = f"{model.names[int(cls)]} {conf:.2f}"
            draw_pickpoint(frame, box, label)
    cv2.putText(frame, f'Latency: {latency_ms:.1f} ms', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='0 for webcam or path to folder/video')
    parser.add_argument('--weights', type=str, default='yolov8n.pt')
    args = parser.parse_args()

    model = YOLO(args.weights)
    # Set confidence threshold
    model.conf = 0.25

    # source handling
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        while True:
            ret, frame = cap.read()
            if not ret: break
            out = process_frame(model, frame)
            cv2.imshow('Scrap Sorter', out)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
    elif os.path.isdir(args.source):
        imgs = sorted([os.path.join(args.source,f) for f in os.listdir(args.source) if f.lower().endswith('.jpg')])
        for p in imgs:
            frame = cv2.imread(p)
            out = process_frame(model, frame)
            cv2.imshow('Scrap Sorter', out)
            if cv2.waitKey(1000) & 0xFF == 27:
                break
    else:
        # try video file
        cap = cv2.VideoCapture(args.source)
        while True:
            ret, frame = cap.read()
            if not ret: break
            out = process_frame(model, frame)
            cv2.imshow('Scrap Sorter', out)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()