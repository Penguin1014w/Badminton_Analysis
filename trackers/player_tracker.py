from ultralytics import YOLO 
import cv2
import pickle
import sys
import constants
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.center_point = (constants.FRAME_WIDTH / 2, constants.FRAME_HEIGHT / 2)

    def choose_and_filter_players(self, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_players = self.choose_players(player_detections_first_frame)
    
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections

    def choose_players(self, player_dict):
    # 选择前两个玩家的 ID（假设 ID 有序）
        chosen_players = list(player_dict.keys())[:2]
        return chosen_players


    # def choose_and_filter_players(self, player_detections, frame_width, frame_height): ##failed will be fix in the future
    #     center_point = (frame_width / 2, frame_height / 2)
    #     player_detections_first_frame = player_detections[0]
    #     chosen_player = self.choose_players(center_point, player_detections_first_frame)

    #     filtered_player_detections = []
    #     for player_dict in player_detections:
    #         filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
    #         filtered_player_detections.append(filtered_player_dict)

    #     return filtered_player_detections


    # def choose_players(self, center_point, player_dict):
    #     distances = []

    #     for track_id, bbox in player_dict.items():
    #         player_center = get_center_of_bbox(bbox)
    #         distance = measure_distance(player_center, center_point)
    #         distances.append((track_id, distance))

    #     distances.sort(key=lambda x: x[1])
    #     chosen_players = [distances[0][0], distances[1][0]]

    #     return chosen_players


    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        player_detections = [self.detect_frame(frame) for frame in frames]
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names
        return {int(box.id.tolist()[0]): box.xyxy.tolist()[0] for box in results.boxes if id_name_dict[box.cls.tolist()[0]] == "person"}

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames
