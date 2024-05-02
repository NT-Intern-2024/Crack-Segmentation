import my_masking_2 as mk2
import numpy as np
import matplotlib.pyplot as plt
import my_utils


class ContourPlotter:
    def __init__(self, image_file: mk2.ImageMaskingBuilder):
        self.contours = None
        self.image_file: mk2.ImageMaskingBuilder = image_file
        self.__my_plot = None

    def __plot(self):
        contours_with_points = self.__create_contour()
        # Plot each contour with points
        for contour_points, color in contours_with_points:
            # Extract x and y coordinates from contour points
            x = [point[0] for point in contour_points]
            y = [point[1] for point in contour_points]

            # Plot points with the specified color
            plt.scatter(x, y, color=color)
        self.__my_plot = plt
        return self.__my_plot

    def show(self):
        self.__create_plot_object()
        print("---------- plotting -------------")
        self.__my_plot.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        self.__my_plot.show()

    def __is_plot_created(self):
        return self.__my_plot is None

    def __create_plot_object(self):
        if self.__is_plot_created():
            self.__plot()

    def __create_contour(self):
        # Iterate over contours
        self.contours, _ = self.image_file.get_contours()
        print(f"contour size: {len(self.contours)}")

        contours_with_points = []
        distinct_colors = self.__random_point_color()

        for i, contour in enumerate(self.contours):
            # Initialize a list to store points in the current contour
            contour_points = []

            # Get color for the current contour
            color = distinct_colors[i]
            print(f"\t contour: {i}")

            # Iterate over points in contour
            for point in contour:
                x, y = point[0]  # Extract x, y coordinates
                contour_points.append((x, y))  # Add point to contour_points list

            # Add the contour with its points and color to the list
            contours_with_points.append((contour_points, color))

        return contours_with_points

    def __random_point_color(self):
        # Generate distinct colors for contours
        num_contours = len(self.contours)
        return [np.random.rand(3, ) for _ in range(num_contours)]

    def export_image(self, file_name: str = "plot.png"):
        self.__create_plot_object()
        self.__my_plot.gca().invert_yaxis()  # Invert y-axis to match image coordinates

        export_path = f"../export/"
        my_utils.check_path_compatibility(export_path)
        self.__my_plot.savefig(f"{export_path}{file_name}", dpi=300, bbox_inches='tight')
        print(f"Export success: {file_name}")
