import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict, deque

SOURCE_VIDEO_PATH = "vehicle_speed.mp4"
TARGET_VIDEO_PATH = "vehicle_speed-result.mp4"
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_NAME = "yolov8x.pt"
MODEL_RESOLUTION = 1280
SPEED_LIMIT_KMH = 150 

SOURCE = np.array([
    [1252, 787],
    [2298, 803],
    [5039, 2159],
    [-550, 2159]
])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
frame_iterator = iter(frame_generator)
frame = next(frame_iterator)

annotated_frame = frame.copy()
annotated_frame = sv.draw_polygon(scene=annotated_frame, polygon=SOURCE, color=sv.Color.RED, thickness=4)
sv.plot_image(annotated_frame)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
model = YOLO(MODEL_NAME)

video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
byte_track = sv.ByteTrack(frame_rate=video_info.fps)

# Dynamic thickness and text scale
thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

polygon_zone = sv.PolygonZone(polygon=SOURCE)
coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

class_names = model.names  # ✅ Use this to get class labels

with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        detections = detections[detections.class_id != 0]
        detections = detections[polygon_zone.trigger(detections)]
        detections = detections.with_nms(IOU_THRESHOLD)
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)

        # Track positions for speed
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)

        labels = []
        violation_flags = []

        for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
            class_name = class_names.get(class_id, "?")  # ✅ Avoid '?'
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"{class_name} #{tracker_id}")
                violation_flags.append(False)
            else:
                y_start = coordinates[tracker_id][-1]
                y_end = coordinates[tracker_id][0]
                distance = abs(y_start - y_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6  # Convert m/s to km/h
                is_violation = speed > SPEED_LIMIT_KMH
                label = f"⚠️ {class_name} {int(speed)} km/h" if is_violation else f"{class_name} {int(speed)} km/h"
                labels.append(label)
                violation_flags.append(is_violation)

        # Annotate using OpenCV
        annotated_frame = frame.copy()
        for box, label, is_violation in zip(detections.xyxy, labels, violation_flags):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 0, 255) if is_violation else (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        text_scale, color, thickness)

        sink.write_frame(annotated_frame)
