from utility.project import *
from image.transform import *
from image.image_utils import *
from image.line import *

import numpy as np
import cv2
from skimage.morphology import skeletonize
from collections import defaultdict

def load_and_skeletonize(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Ensure the image is binary
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # Skeletonize the image
    skeleton = skeletonize(binary_image // 255).astype(np.uint8) * 255
    return skeleton

def get_neighbors(y, x, shape):
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
                neighbors.append((ny, nx))
    return neighbors

def detect_endpoints_and_branchpoints(skeleton):
    endpoints = set()
    branchpoints = set()
    
    for y in range(skeleton.shape[0]):
        for x in range(skeleton.shape[1]):
            if skeleton[y, x] == 255:
                neighbors = get_neighbors(y, x, skeleton.shape)
                count = sum(skeleton[ny, nx] == 255 for ny, nx in neighbors)
                if count == 1:
                    endpoints.add((y, x))
                elif count > 2:
                    branchpoints.add((y, x))
    
    return endpoints, branchpoints

def extract_segments(skeleton):
    # Get the coordinates of all white pixels
    white_pixels = np.argwhere(skeleton == 255)
    
    # Detect branch points and endpoints
    endpoints, branchpoints = detect_endpoints_and_branchpoints(skeleton)
    
    # Create a graph where each white pixel is a node
    graph = defaultdict(list)
    for y, x in white_pixels:
        for ny, nx in get_neighbors(y, x, skeleton.shape):
            if skeleton[ny, nx] == 255:
                graph[(y, x)].append((ny, nx))
    
    # Function to perform BFS or DFS and extract a line segment
    def extract_line_segment(start, graph, visited, branchpoints):
        stack = [start]
        segment = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            segment.append(node)
            if node in branchpoints:
                continue
            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
        return segment

    # Separate lines using BFS or DFS
    visited = set()
    lines = []
    for pixel in white_pixels:
        pixel_tuple = tuple(pixel)
        if pixel_tuple not in visited and pixel_tuple not in branchpoints:
            line_segment = extract_line_segment(pixel_tuple, graph, visited, branchpoints)
            if line_segment:
                lines.append(line_segment)
    
    # Handle branch points by treating each branch as a new segment
    for branch in branchpoints:
        if branch not in visited:
            for neighbor in graph[branch]:
                if neighbor not in visited:
                    line_segment = extract_line_segment(neighbor, graph, visited, branchpoints)
                    if line_segment:
                        lines.append(line_segment)
    
    return lines

def merge_segments(lines, merge_distance_threshold=10):
    # Convert lines to list of tuples for easier manipulation
    lines = [list(map(tuple, line)) for line in lines]
    
    # Merge segments based on proximity of endpoints
    merged_lines = []
    merged = set()  # To keep track of merged segments
    
    for i in range(len(lines)):
        if i in merged:
            continue
        merged_segment = lines[i][:]
        start = merged_segment[0]
        end = merged_segment[-1]
        
        for j in range(i + 1, len(lines)):
            if j in merged:
                continue
            other_segment = lines[j]
            other_start = other_segment[0]
            other_end = other_segment[-1]
            
            # Check proximity of endpoints
            if np.linalg.norm(np.array(start) - np.array(other_end)) <= merge_distance_threshold:
                merged_segment = other_segment[::-1] + merged_segment
                merged.add(j)
            elif np.linalg.norm(np.array(end) - np.array(other_start)) <= merge_distance_threshold:
                merged_segment = merged_segment + other_segment[1:]
                merged.add(j)
        
        merged_lines.append(merged_segment)
    
    return merged_lines

def format_segments(segments):
    return [[[y, x] for y, x in segment] for segment in segments]

def separate_lines(image_path):
    skeleton = load_and_skeletonize(image_path)
    segments = extract_segments(skeleton)
    merged_segments = merge_segments(segments)
    return format_segments(merged_segments)



# Example usage
# Assume `skeleton_image` is the input binary image

change_to_main_root()
# TestImage: Perfect
# image_path = "./sample/Results-stwcrack-Alex0310-Mask.png"

# TestImage: Small braching
# image_path = "./sample/Results-stwcrack-Alex0373-Mask.png"

# TestImage:
image_path = "./sample/Results-stwcrack-Alex0342-Mask.png"
# image_path = "./sample/Alex0324.JPG"


my_image = load_image(image_path)

img_width, img_height = my_image.shape[0], my_image.shape[1]
check_loaded_image(my_image)
# show_image(my_image, "My Image")

skel_image = skeletonize_image(my_image)
# show_image(skel_image, "Skel")

# skeleton_image = cv2.imread('path_to_skeleton_image.png', cv2.IMREAD_GRAYSCALE)
output_path = "./output/grouping-3"
check_path_compatibility(output_path)
remove_all_files(output_path)

export_image(my_image, "00-image.png", output_path)
export_image(skel_image, "01-skel.png", output_path)
lines = separate_lines(f"{output_path}/01-skel.png")

print(lines)


export_image_from_lines(img_width, img_height, lines, "test-grouping", output_path)
# Print the extracted lines
# for i, line in enumerate(lines):
#     print(f"Line {i+1}: {line}")
