import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener


def heic_to_jpeg(heic_dir, jpeg_dir):
    register_heif_opener()
    image = Image.open(heic_dir)
    image.save(jpeg_dir, "JPEG")


def remove_background(jpeg_dir, path_to_clean_image):
    if jpeg_dir[-4:] in ["heic", "HEIC"]:
        heic_to_jpeg(jpeg_dir, jpeg_dir[:-4] + "jpg")
        jpeg_dir = jpeg_dir[:-4] + "jpg"

    img = cv2.imread(jpeg_dir)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 80], dtype="uint8")
    upper = np.array([50, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    b, g, r = cv2.split(result)
    filter = g.copy()
    ret, mask = cv2.threshold(filter, 10, 255, 1)
    img[mask == 255] = 255
    cv2.imwrite(path_to_clean_image, img)


def resize(
    path_to_warped_image,
    path_to_warped_image_clean,
    path_to_warped_image_mini,
    path_to_warped_image_clean_mini,
    resize_value,
):
    pil_img = Image.open(path_to_warped_image)
    pil_img_clean = Image.open(path_to_warped_image_clean)
    pil_img.resize((resize_value, resize_value), resample=Image.NEAREST).save(
        path_to_warped_image_mini
    )
    pil_img_clean.resize((resize_value, resize_value), resample=Image.NEAREST).save(
        path_to_warped_image_clean_mini
    )


def save_result(im, contents, resize_value, path_to_result):
    if im is None:
        print_error()
    else:
        (
            heart_content_1,
            heart_content_2,
            head_content_1,
            head_content_2,
            life_content_1,
            life_content_2,
        ) = contents
        image_height, image_width = im.size
        fontsize = 12

        plt.tick_params(
            axis="both",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            left=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False,
        )

        note_1 = "* Note: This program is just for fun! Please take the result with a light heart."
        note_2 = "   If you want to check out more about palmistry, we recommend https://www.allure.com/story/palm-reading-guide-hand-lines"

        plt.title(" Check your palmistry result!", fontsize=14, y=1.01)

        plt.text(image_width + 15, 15, "<Heart line>", color="r", fontsize=fontsize)
        plt.text(image_width + 15, 35, heart_content_1, fontsize=fontsize)
        plt.text(image_width + 15, 55, heart_content_2, fontsize=fontsize)
        plt.text(image_width + 15, 80, "<Head line>", color="g", fontsize=fontsize)
        plt.text(image_width + 15, 100, head_content_1, fontsize=fontsize)
        plt.text(image_width + 15, 120, head_content_2, fontsize=fontsize)
        plt.text(image_width + 15, 145, "<Life line>", color="b", fontsize=fontsize)
        plt.text(image_width + 15, 165, life_content_1, fontsize=fontsize)
        plt.text(image_width + 15, 185, life_content_2, fontsize=fontsize)

        plt.text(image_width + 15, 230, note_1, fontsize=fontsize - 1, color="gray")
        plt.text(image_width + 15, 250, note_2, fontsize=fontsize - 1, color="gray")

        plt.imshow(im)
        plt.savefig(path_to_result, bbox_inches="tight")
        plt.close()


def print_error():
    print("Palm lines not properly detected! Please use another palm image.")


def load_image(image_path: str):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def my_plot(
    original: cv2.typing.MatLike,
    warped: cv2.typing.MatLike,
    palmline: cv2.typing.MatLike,
    result: cv2.typing.MatLike,
    line_length: list,
    result_contents: list,
):
    axes = plt.figure(layout="constrained").subplot_mosaic(
        """
        ABCD
        EFFF
        """
    )

    axes["A"].imshow(original)
    axes["A"].axis("off")
    axes["A"].set_title("Original")

    axes["B"].imshow(warped)
    axes["B"].axis("off")
    axes["B"].set_title("Warped Palm")

    axes["C"].imshow(palmline)
    axes["C"].axis("off")
    axes["C"].set_title("Detected Line")

    axes["D"].imshow(result)
    axes["D"].axis("off")
    axes["D"].set_title("Result")

    # Add text with pattern in the last subplot
    length_info = f"""
    Measure the Length of each line 
    1. Heart line = {line_length[0]}
    2. Head line  = {line_length[1]}
    3. Life line  = {line_length[2]}
    """

    axes["E"].text(0.1, 0.5, length_info, fontsize=12, verticalalignment="center")
    axes["E"].axis("off")

    create_predicted_text(axes["F"], result_contents)

    plt.savefig("results/compare.jpg")
    plt.show()


def create_predicted_text(axes, contents: list):
    (
        heart_content_1,
        heart_content_2,
        head_content_1,
        head_content_2,
        life_content_1,
        life_content_2,
    ) = contents

    fontsize = 12
    axes.axis("off")

    pos_x = 0
    pos_y = 0

    heart_line_text = f"""
    <Heart line>
    {heart_content_1}
    {heart_content_2}
    """

    head_line_text = f"""
    <Head line>
    {head_content_1}
    {head_content_2}
    """

    life_line_text = f"""
    <Life line>
    {life_content_1}
    {life_content_2}
    """

    axes.text(pos_x, pos_y, heart_line_text, fontsize=fontsize, wrap=True, color="r")
    axes.text(
        pos_x, pos_y + 0.3, head_line_text, fontsize=fontsize, wrap=True, color="g"
    )
    axes.text(
        pos_x, pos_y + 0.6, life_line_text, fontsize=fontsize, wrap=True, color="b"
    )
