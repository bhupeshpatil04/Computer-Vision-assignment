#!/usr/bin/env python3
"""Train YOLOv8 (wrapper)."""
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset/dataset.yaml')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()

    model = YOLO('yolov8n.pt')  # small model; ultralytics will download if missing
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)
    print('Training finished. Check runs/detect/train for weights.')

if __name__ == '__main__':
    main()