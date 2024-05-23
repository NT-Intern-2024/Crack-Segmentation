from utility.project import change_to_project_path
from utility.config import Config
from image.transform import *
from image.image_processor import *
from image.hand_detector import *
from image.image import *

def main():
    change_to_project_path(__file__)
    check_path_exists(CONFIG_FILE)
    user_config = Config()
    print(user_config)
    print(user_config.get('dataset', 'path'))

    # image_path = "../data/Palm/etc/hand-2.jpg"
    # image_path = "../data/Palm/PalmAll/IMG_FEMALE_0001.jpg"
    # image_path = "../test/female/2.jpg"
    image_path = "../test/hand-1.jpg"
    # resize_image(load_image(image_path))

    hand_detector = HandDetector()

    image = Image(image_path).image
    # resized_image = image_processor.resize_image(image)
    # image_processor.display_image(resized_image)

    results = hand_detector.detect_hands(image)
    hand_detector.draw_detection(image, results)

    # warp(image)
    image = Image(image_path)
    image.show()
    image.show()
    image.show()

if __name__ == "__main__":
    main()
