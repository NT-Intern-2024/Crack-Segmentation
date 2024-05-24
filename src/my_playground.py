from utility.project import *
from image.image_utils import *
from image.masking import *
from archive.masking_builder import *

image_path = "../data/Palm/After1/IMG_0016.JPG"
# image_path = "..\\data\\Palm\\After1\\IMG_0016.JPG"
# image_path = "palm1.JPG"
# image_path = "../output/IMG_0001.JPG"

# Old using
change_to_project_path(__file__)
image8 = MaskingBuilder(image_path)

# Run
# image8.show("Original")

# image8.do_equalize_histogram().do_adaptive_mean().do_denoise_morphology_open()
# image8.show("Process")

# cv2.waitKey()


datas = [
    # (
    #     image1,
    #     image1_process1,
    #     image1_process2,
    # ),
    # (
    #     image2,
    #     image2_process1,
    #     image2_process2,
    # ),
    # (
    #     image8.image,
    #     image8.do_adaptive_mean().image,
    #     image8.do_denoise_morphology_combined().image,
    # ),
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
# my_masking.export_masking(image_path="../data/CrackLS315/image", output_path="../data/output/CrackLS315/mask")
# my_masking.export_masking(image_path="../data/Palm/etc", output_path="../output/etc/")

# TODO: สร้างขอบมือ
# image_contour = cp.ContourPlotter(image8)

# แสดงผล contour
# image_contour.show()

# เซฟรุูปใน export > plot.png
# image_contour.export_image()
