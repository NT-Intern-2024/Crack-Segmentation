import cv2
import mediapipe as mp
from .image_utils import *


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands

    def detect_hands(self, hand_image: cv2.typing.MatLike):
        with self.mp_hands.Hands(
                static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7
        ) as hands:
            return hands.process(cv2.flip(cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB), 1))

    def draw_detection(self, hand_image: cv2.typing.MatLike, results):
        mp_hands = self.mp_hands

        # Draw hand landmarks
        image_height, image_width, _ = hand_image.shape
        annotated_image = cv2.flip(hand_image.copy(), 1)
        for hand_landmarks in results.multi_hand_landmarks:
            print("Index finger tip coordinate:", end=" ")
            print(
                f"({hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, "
                f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})"
            )
            self.__draw_landmarks(annotated_image, hand_landmarks)
        plot_image(annotated_image)

    def __draw_landmarks(self, annotated_image: cv2.typing.MatLike, hand_landmarks: list):
        mp_hands = self.mp_hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
