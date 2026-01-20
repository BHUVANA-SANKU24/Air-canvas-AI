import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


class HandTracker:
    def __init__(
        self,
        model_path="models/hand_landmarker.task",
        max_hands=1,
        detection_confidence=0.6,
        tracking_confidence=0.5,
        smooth_factor=0.6
    ):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None
        self.landmark_list = []
        self.tip_ids = [4, 8, 12, 16, 20]

        self.smooth_factor = smooth_factor
        self.prev_pos = None

        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

    def find_hands(self, frame, draw=True, timestamp_ms=0):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self.results = self.detector.detect_for_video(mp_image, timestamp_ms)

        if draw and self.results and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                self._draw_landmarks(frame, hand_landmarks)

        return frame

    def _draw_landmarks(self, frame, hand_landmarks):
        h, w, _ = frame.shape

        for lm in hand_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        for a, b in self.connections:
            ax, ay = int(hand_landmarks[a].x * w), int(hand_landmarks[a].y * h)
            bx, by = int(hand_landmarks[b].x * w), int(hand_landmarks[b].y * h)
            cv2.line(frame, (ax, ay), (bx, by), (255, 0, 0), 2)

    def get_position(self, frame, hand_no=0):
        self.landmark_list = []
        if self.results and self.results.hand_landmarks:
            if hand_no < len(self.results.hand_landmarks):
                hand = self.results.hand_landmarks[hand_no]
                h, w, _ = frame.shape
                for idx, lm in enumerate(hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.landmark_list.append([idx, cx, cy])
        return self.landmark_list

    def fingers_up(self):
        if len(self.landmark_list) == 0:
            return []

        fingers = []
        # thumb x axis
        if self.landmark_list[self.tip_ids[0]][1] > self.landmark_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # other fingers y axis
        for i in range(1, 5):
            if self.landmark_list[self.tip_ids[i]][2] < self.landmark_list[self.tip_ids[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def get_gesture(self):
        fingers = self.fingers_up()
        if len(fingers) == 0:
            return "none"

        # draw
        if fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]:
            return "draw"

        # submit fist
        if fingers[1:] == [0, 0, 0, 0]:
            return "submit"

        return "unknown"

    def get_fingertip_position(self, finger_id=8, smooth=True):
        if len(self.landmark_list) <= finger_id:
            self.prev_pos = None
            return None

        x, y = self.landmark_list[finger_id][1], self.landmark_list[finger_id][2]

        if not smooth:
            return x, y

        if self.prev_pos is None:
            self.prev_pos = (x, y)
            return x, y

        px, py = self.prev_pos
        sx = int(self.smooth_factor * x + (1 - self.smooth_factor) * px)
        sy = int(self.smooth_factor * y + (1 - self.smooth_factor) * py)
        self.prev_pos = (sx, sy)
        return sx, sy
