import numpy as np
import cv2
import mediapipe as mp
from utility.constants import *
from utility.file_utils import *
from image.hand_detector import *

from skimage.morphology import skeletonize

def warp_image(image: cv2.typing.MatLike):
    pts_index = list(range(21))

    hand_detector = HandDetector()
    results = hand_detector.detect_hands(image)
    # hand_detector.draw_detection(image, results)
    image_height, image_width, _ = image.shape
    if results.multi_hand_landmarks is None:
        return WarpStatus.FAIL
    else:
        hand_landmarks = results.multi_hand_landmarks[0]
        print(f"hand_landmarks", hand_landmarks)
        # 2. Align images
        pts = np.float32(
            [
                [
                    hand_landmarks.landmark[i].x * image_width,
                    hand_landmarks.landmark[i].y * image_height,
                ]
                for i in pts_index
            ]
        )
        pts_target = get_target_point(image_width, image_height)
        target_matrix, mask = cv2.findHomography(pts, pts_target, cv2.RANSAC, 5.0)
        check_path_compatibility(PROCESSED_PATH)
        cv2.imwrite(f"{PROCESSED_PATH}/rect_2_M.jpg", target_matrix)
        cv2.imwrite(f"{PROCESSED_PATH}/rect_2_mask.jpg", mask)
        warped_image = cv2.warpPerspective(
            image, target_matrix, (image_width, image_height), borderMode=cv2.BORDER_REPLICATE
        )
        cv2.imwrite(f"{PROCESSED_PATH}/warp.jpg", warped_image)
        new_result = hand_detector.detect_hands(warped_image)
        hand_detector.draw_detection(warped_image, new_result)
        return WarpStatus.SUCCESS

def warp_image2(image: cv2.typing.MatLike):
    pts_index = list(range(21))

    hand_detector = HandDetector()
    results = hand_detector.detect_hands(image)
    # hand_detector.draw_detection(image, results)
    image_height, image_width, _ = image.shape
    if results.multi_hand_landmarks is None:
        return WarpStatus.FAIL
    else:
        hand_landmarks = results.multi_hand_landmarks[0]
        print(f"hand_landmarks", hand_landmarks)
        # 2. Align images
        pts = np.float32(
            [
                [
                    hand_landmarks.landmark[i].x * image_width,
                    hand_landmarks.landmark[i].y * image_height,
                ]
                for i in pts_index
            ]
        )
        pts_target = get_target_point(image_width, image_height)
        target_matrix, mask = cv2.findHomography(pts, pts_target, cv2.RANSAC, 5.0)

        warped_image = cv2.warpPerspective(
            image, target_matrix, (image_width, image_height), borderMode=cv2.BORDER_REPLICATE
        )

        cv2.imwrite(f"{PROCESSED_PATH}/warp.jpg", warped_image)
        new_result = hand_detector.detect_hands(warped_image)
        hand_detector.draw_detection(warped_image, new_result)
        return WarpStatus.SUCCESS

def get_target_point(image_width: int, image_height: int):
    return np.float32(
        [[x * image_width, y * image_height] for x, y in pts_target_normalized]
    )


def warp(image: cv2.typing.MatLike):
    warp_result = warp_image(image)
    return warp_result


def skeletonize_image(binary_image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    # gray_img = cv2.cvtColor(binary_image, cv2.COLOR_gra)
    
    skeleton = skeletonize(binary_image)
    # skeleton = skeletonize(gray_img)
    skel_img = skeleton.astype(np.uint8) * 255

    return skel_img