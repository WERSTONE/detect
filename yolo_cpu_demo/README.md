YOLOv8 CPU-only quick reference (detection + pose)

Prereqs
- Activate the existing venv: `.\.venv\Scripts\activate`
- CLIs available: `yolo`, `python`

Sample inputs and outputs
- Inputs: `yolo_cpu_demo/assets/bus.jpg`, `yolo_cpu_demo/assets/zidane.jpg`
- Outputs land under `yolo_cpu_demo/runs/`
- Local weights: `yolo_cpu_demo/models/yolov8n.pt`, `yolo_cpu_demo/models/yolov8n-pose.pt`

Run detection (CPU)
- Image: `yolo predict model=yolo_cpu_demo/models/yolov8n.pt source=yolo_cpu_demo/assets/bus.jpg device=cpu`
- Video: `yolo predict model=yolo_cpu_demo/models/yolov8n.pt source=path/to/video.mp4 device=cpu save=True`

Run pose (CPU)
- `yolo pose predict model=yolo_cpu_demo/models/yolov8n-pose.pt source=yolo_cpu_demo/assets/zidane.jpg device=cpu`

Custom video-to-video script (CPU)
- `python yolo_cpu_demo/video_detect.py --input path/to/video.mp4 --output yolo_cpu_demo/runs/detect/out.mp4 --model yolo_cpu_demo/models/yolov8n.pt --device cpu`

Notes
- Results are written to subfolders of `yolo_cpu_demo/runs` (e.g., `runs/detect/predict*`, `runs/pose/predict*`).
- You can point `source=` to any local image/video; CPU inference will be slower on larger models.
