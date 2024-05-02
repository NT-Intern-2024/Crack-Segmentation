import numpy as np
import cv2
import my_masking
import image_masking_builder as imb
import my_utils
import contour_plotter as cp

# image_path = my_utils.path_best
# image_path = my_utils.path
# image_path = "../data/Palm/After1/IMG_0016.JPG"
# image_path = "..\\data\\Palm\\After1\\IMG_0016.JPG"
# image_path = "palm1.JPG"
image_path = "../output/IMG_0001.JPG"

# Old using
# image1 = my_masking.load_image(image_path)
# image1_process1 = my_masking.equalize_histogram(image=image1)
# image1_process2 = my_masking.adaptive_mean(image=image1_process1)
#
# image2 = my_masking.load_image(image_path)
# image2_process1 = my_masking.adaptive_mean(image=image2)
# image2_process2 = my_masking.equalize_histogram(image=image2_process1)

image8 = imb.ImageMaskingBuilder(image_path)

datas = [
    # (image1, image1_process1, image1_process2, ),
    # (image2, image2_process1, image2_process2, ),
    # (image8.image, image8.do_adaptive_mean().image, image8.do_denoise_morphology_combined().image,),
]


def add_data(image_data: list[np.ndarray]):
    datas.append(
        {"Original": image_data[0],
         "Processed 1": image_data[1],
         "Processed 2": image_data[2]
         }
    )


# add_data(
#     [
#         image8.image,
#         image8.do_adaptive_mean().image,
#         image8.do_denoise_morphology_combined().image,
#     ]
# )
#
# image8.reset_image()
# add_data(
#     [
#         image8.image,
#         image8.do_adaptive_mean().image,
#         image8.do_denoise_morphology_combined().image,
#     ]
# )

# TODO: อยากลอง plot กราฟ ใช้บรรทัดนี้
# my_masking.plot_datas(datas)
# my_utils.plot_datas(datas)

# TODO: export ภาพ ใช้บรรทัดนี้
# my_masking.export_masking_dataset()


# TODO: สร้างขอบมือ
# image_contour = cp.ContourPlotter(image8)

# แสดงผล contour
# image_contour.show()

# เซฟรุูปใน export > plot.png
# image_contour.export_image()

# image8.show()

# Read the input image
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding or segmentation to extract the hand region
_, binary_hand = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

# Find contours in the binary hand image
contours, _ = cv2.findContours(binary_hand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Select the largest contour (assuming it corresponds to the hand)
hand_contour = max(contours, key=cv2.contourArea)

# Compute the convex hull of the hand contour
convex_hull = cv2.convexHull(hand_contour)

# Draw the hand contour and convex hull on a blank canvas
canvas = np.zeros_like(image)
hand_image_with_contour = cv2.drawContours(canvas.copy(), [hand_contour], -1, (0, 255, 0), 2)
hand_image_with_hull = cv2.drawContours(canvas.copy(), [convex_hull], -1, (0, 0, 255), 2)

# Display the original image, hand contour, and convex hull
my_utils.show_image(window_name="Original", image=image)
my_utils.show_image(window_name="Hand Contour", image=hand_image_with_contour)
my_utils.show_image(window_name="Convex Hull", image=hand_image_with_hull)
# cv2.imshow('Original Image', image)
# cv2.imshow('Hand Contour', hand_image_with_contour)
# cv2.imshow('Convex Hull', hand_image_with_hull)
cv2.waitKey(0)

# Approximate the palm region using the convex hull
palm_rect = cv2.boundingRect(convex_hull)

# Extract the palm area from the original image
palm_area = image[palm_rect[1]:palm_rect[1] + palm_rect[3], palm_rect[0]:palm_rect[0] + palm_rect[2]]

# Resize the palm area for display
resized_palm = cv2.resize(palm_area, (300, 300))  # Adjust the size as needed

# Display the resized palm area
cv2.imshow('Resized Palm Area', resized_palm)
cv2.waitKey(0)
cv2.destroyAllWindows()
