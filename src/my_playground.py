import numpy as np
import cv2
import my_masking
import image_masking_builder as imb
import my_utils
import contour_plotter as cp

# image_path = my_utils.path_best
# image_path = my_utils.path

image_path = "../data/Palm/After1/IMG_0016.JPG"
# image_path = "..\\data\\Palm\\After1\\IMG_0016.JPG"
# image_path = "palm1.JPG"
# image_path = "../output/IMG_0001.JPG"

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
        {
            "Original": image_data[0],
            "Processed 1": image_data[1],
            "Processed 2": image_data[2],
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

# Run
image8.show("Original")

image8.do_adaptive_mean()
image8.show("Adaptive Mean")

cv2.waitKey()
