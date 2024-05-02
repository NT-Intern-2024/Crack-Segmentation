import my_masking
import my_masking_2 as mk2
import my_utils

# image_path = my_utils.path_best
# image_path = my_utils.path
# image_path = "../data/Palm/After1/IMG_0016.JPG"
# image_path = "..\\data\\Palm\\After1\\IMG_0016.JPG"
image_path = "palm1.JPG"

image1 = my_masking.load_image(image_path)
image1_process1 = my_masking.equalize_histogram(image=image1)
image1_process2 = my_masking.adaptive_mean(image=image1_process1)

image2 = my_masking.load_image(image_path)
image2_process1 = my_masking.adaptive_mean(image=image2)
image2_process2 = my_masking.equalize_histogram(image=image2_process1)

image3 = my_masking.load_image(image_path)
image3_process1 = my_masking.adaptive_gaussian(image=image3)
image3_process2 = my_masking.equalize_histogram(image=image3_process1)

image4 = my_masking.load_image(image_path)
image4_process1 = my_masking.adaptive_gaussian(image=image4)
image4_process2 = my_masking.adaptive_mean(image=image4_process1)

image5 = my_masking.load_image(image_path)
image5_process1 = my_masking.adaptive_mean_fix1(image=image5)
image5_process2 = my_masking.denoise_morphology(image=image5_process1)

image6 = my_masking.load_image(image_path)
image6_process1 = my_masking.adaptive_mean_fix1(image=image6)
image6_process2 = my_masking.denoise_morphology(image=image6_process1)

image7 = my_masking.load_image(image_path)
image7_process1 = my_masking.adaptive_mean(image=image7)
image7_process2 = my_masking.denoise_morphology(image=image7_process1)

image8 = mk2.ImageMaskingBuilder(image_path)

datas = [
    # (image1, image1_process1, image1_process2, ),
    # (image2, image2_process1, image2_process2, ),
    # (image3, image3_process1, image3_process2,),
    # (image4, image4_process1, image4_process2,),
    # (image5, image5_process1, image5_process2,),
    # (image6, image6_process1, image6_process2,),
    # (image7, image7_process1, image7_process2,),
    # (image8.image, image8.do_adaptive_mean().image, image8.do_denoise_morphology_combined().image,),
]


def add_data(image_data: list):
    datas.append(
        {"Original": image_data[0],
         "Processed 1": image_data[1],
         "Processed 2": image_data[2]
         }
    )


add_data(
    [
        image8.image,
        image8.do_adaptive_mean().image_file,
        image8.do_denoise_morphology_combined().image_file,
    ]
)

image8.reset_image()
add_data(
    [
        image8.image,
        image8.do_adaptive_mean().image_file,
        image8.do_denoise_morphology_combined().image_file,
    ]
)

# TODO: อยากลอง plot กราฟ ใช้บรรทัดนี้
# my_masking.plot_datas(datas)
my_utils.plot_datas(datas)

# TODO: export ภาพ ใช้บรรทัดนี้
# my_masking.export_masking_dataset()
