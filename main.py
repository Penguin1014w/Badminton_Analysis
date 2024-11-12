from utils import (read_video, save_video)
from trackers import PlayerTracker, BallTracker
import constants
import cv2

def main(input_video_path="input_videos/input_video.mp4",
         output_video_path="output_videos/output_video.avi",
         player_model_path="yolov8x",
         ball_model_path="models/yolo5_best.pt",
         player_stub_path="tracker_stubs/player_detections.pkl",
         ball_stub_path="tracker_stubs/ball_detections.pkl"):

    video_frames = read_video(input_video_path)

    #tracker players and balls
    player_tracker = PlayerTracker(model_path=player_model_path)
    ball_tracker = BallTracker(model_path=ball_model_path)

    #detect players and balls in the first frame
    player_detections = player_tracker.detect_frames(
        video_frames, read_from_stub=True, stub_path=player_stub_path)

    ball_detections = ball_tracker.detect_frames(
        video_frames, read_from_stub=True, stub_path=ball_stub_path)
    
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)#the missing ball
    #return the players
    player_detections = player_tracker.choose_and_filter_players(player_detections)


    # draw the output
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    # save video
    save_video(output_video_frames, output_video_path)

if __name__ == "__main__":
    main()
