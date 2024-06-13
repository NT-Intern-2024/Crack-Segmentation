from utility.logger import *
from image.image_utils import *
from utility.project import *
from graph.cluster import *
from utility.printer import *
from utility.matrix import *

import numpy as np
import os
import cv2
import glob
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
import mediapipe as mp

#########################################################################################
# Sketch of idea                                                                        #
# - Build a graph : nodes = end pt. + intersection pt.                                  #
#                  edges = line segments (save pixel coordinates with next direction)   #
# - Find all possible paths(lines) from graph : graph backtracking                      #
# - Find three lines nearest to each cluster centers in feature space                   #
#   Find cluster centers using k-means clustering (k=3)                                 #
#########################################################################################

# code reference: https://google.github.io/mediapipe/solutions/hands.html
# data reference: https://drive.google.com/file/d/1B4uj-b4RuUNkC_oeCsRkuih4rjuQ6tHH/view

### 0. Load Data ###


# PLSU folder should exist in advance
# rectify images in PLSU folder
# find homography matrix using original image
# then apply homography matrix to detected line image
# input is the number 'idx' from image{idx}.jpg(.png)
def rectify(idx):
    # TODO: Good sample path (dataset)
    # img_path = "./PLSU/PLSU/"
    # img_path = "./data/PLSU/" 
    img_path = "./data/MySample2/"
    image = cv2.imread(img_path + "img/image" + str(idx) + ".jpg")
    print(f"load image: image{idx}")
    check_loaded_image(image)
    image_mask = cv2.imread(
        img_path + "Mask/image" + str(idx) + ".png", cv2.IMREAD_GRAYSCALE
    )
    print(f"load mask: mask{idx}")
    check_loaded_image(image_mask)
    mp_hands = mp.solutions.hands

    # 7 landmark points (normalized)
    pts_index = list(range(21))
    pts_target_normalized = np.float32(
        [
            [1 - 0.48203104734420776, 0.9063420295715332],
            [1 - 0.6043621301651001, 0.8119394183158875],
            [1 - 0.6763232946395874, 0.6790258884429932],
            [1 - 0.7340714335441589, 0.5716733932495117],
            [1 - 0.7896472215652466, 0.5098430514335632],
            [1 - 0.5655680298805237, 0.5117031931877136],
            [1 - 0.5979393720626831, 0.36575648188591003],
            [1 - 0.6135331392288208, 0.2713503837585449],
            [1 - 0.6196483373641968, 0.19251111149787903],
            [1 - 0.4928809702396393, 0.4982593059539795],
            [1 - 0.4899863600730896, 0.3213786780834198],
            [1 - 0.4894656836986542, 0.21283167600631714],
            [1 - 0.48334982991218567, 0.12900274991989136],
            [1 - 0.4258815348148346, 0.5180916786193848],
            [1 - 0.4033462107181549, 0.3581996262073517],
            [1 - 0.3938145041465759, 0.2616880536079407],
            [1 - 0.38608720898628235, 0.1775170862674713],
            [1 - 0.36368662118911743, 0.5642163157463074],
            [1 - 0.33553171157836914, 0.44737303256988525],
            [1 - 0.3209102153778076, 0.3749568462371826],
            [1 - 0.31213682889938354, 0.3026996850967407],
        ]
    )

    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    ) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks == None:
            return np.zeros_like(image)
        image_height, image_width, _ = image.shape
        hand_landmarks = results.multi_hand_landmarks[0]
        pts = np.float32(
            [
                [
                    hand_landmarks.landmark[i].x * image_width,
                    hand_landmarks.landmark[i].y * image_height,
                ]
                for i in pts_index
            ]
        )
        pts_target = np.float32(
            [[x * image_width, y * image_height] for x, y in pts_target_normalized]
        )
        M, mask = cv2.findHomography(pts, pts_target, cv2.RANSAC, 5.0)
        rectified_image = cv2.warpPerspective(
            image_mask, M, (image_width, image_height)
        )
        pil_img = Image.fromarray(rectified_image)
        # TODO: Convert to binary image
        pil_img = pil_img.point(lambda pixel: pixel > 128 and 255)
        rectified_image = np.asarray(
            pil_img.resize((1024, 1024), resample=Image.NEAREST)
        )
        

        return rectified_image


# load rectified data from PLSU to new folder
def load_data(num_data):
    # make directory
    data_path = "./line_sample"
    result_path = "./line_sample_result"
    path_list = [data_path, result_path]
    for path in path_list:
        os.makedirs(path, exist_ok=True)

    # count the current number of data
    cur_num_data = len(os.listdir(data_path))

    # load rectified data from PLSU data
    offset = 50
    img_path_list = [img_path for img_path in glob.glob("./PLSU/PLSU/Img/*")][
        offset + cur_num_data : offset + num_data
    ]
    for i, img_path in enumerate(img_path_list):
        idx = img_path.split("image")[1].split(".")[0]
        rectified_image = rectify(idx)
        if np.sum(rectified_image) == 0:
            continue
        cv2.imwrite(
            "line_sample/image" + str(i + cur_num_data) + ".png", rectified_image
        )


### 1. Find possible lines ###

# connect seperated lines by gradient
# https://stackoverflow.com/questions/63727525/how-to-connect-broken-lines-that-cannot-be-connected-by-erosion-and-dilation
# https://stackoverflow.com/questions/43859750/how-to-connect-broken-lines-in-a-binary-image-using-python-opencv


# find all possible lines by graph backtracking
# lines_node : list of lines represented by nodes // ex. [[node0, ..., node3], ..., [node2, ..., node4]]
# temp : list of nodes up to now
# graph : node -> {adj. node -> line between two node} (dictionary type)
# visited_node : visited nodes up to now
# finished_node : visited nodes at least once
# node : current node
def backtrack(lines_node, temp, graph, visited_node, finished_node, node):
    # logger_backtrack.info(f"\t\t Enter {backtrack.__name__}")
    end_pt = True
    for next_node in graph[node].keys():
        # logger_backtrack.info(f"\t\t\t visted[]: \t{visited_node}")
        # logger_backtrack.info(f"\t\t\t finished[]: \t{finished_node}")

        if not visited_node[next_node]:
            end_pt = False
            temp.append(next_node)
            visited_node[next_node] = True
            finished_node[next_node] = True
            backtrack(lines_node, temp, graph, visited_node, finished_node, next_node)
            del temp[-1]
            visited_node[next_node] = False
    # if there is no way to preceed, current node is the end node
    # add current line to the list
    if end_pt:
        line_node = []
        line_node.extend(temp)
        lines_node.append(line_node)


# find possible lines
# (1) build a graph
# (2) find all possible lines by graph backtracking
# (3) filter lines with length, direction criteria
def group(img):
    # (1) build a graph

    # (1)-1 find all nodes
    count = np.zeros(img.shape)
    image_size = list(img.shape[:2])

    nodes = []
    # print(f"------ group():----------")
    for j in range(1, img.shape[0] - 1):
        for i in range(1, img.shape[1] - 1):
            if img[j, i] == 0:
                continue
            count[j, i] = np.count_nonzero(img[j - 1 : j + 2, i - 1 : i + 2]) - 1
            if count[j, i] == 1 or count[j, i] >= 3:
                nodes.append((j, i))

    # print_matrix(count, "count: (After check nonzero)")
    export_image_from_line(image_size[0], image_size[1], nodes, "01-node")

    # sort nodes to traverse from upper-left to lower-right
    logger_classify.info(f"\t nodes \t: {nodes}")

    nodes.sort(key=lambda x: x[0] + x[1])
    # print("sorted nodes")
    # print(nodes)

    logger_classify.info(f"\t nodes (sorted): {nodes}")

    # (1)-2 save all connections
    graph = dict()
    for node in nodes:
        graph[node] = dict()

    not_visited = np.ones(img.shape)

    n_node = 1
    
    for node in nodes:
        y, x = node
        # print(f"node - y:{y}, x:{x}")
        not_visited[y, x] = 0
        around = np.multiply(
            count[y - 1 : y + 2, x - 1 : x + 2],
            not_visited[y - 1 : y + 2, x - 1 : x + 2],
        )

        # print_matrix(not_visited, "nv: 0")
        # print_matrix(around, "around: ")

        next_pos = np.transpose(np.nonzero(around))
        if next_pos.shape[0] == 0:
            # n_node += 1
            continue

        # print(np.nonzero(around))
        # print_matrix(next_pos, "next_pos: ")

        for dy, dx in next_pos:
            # print(f"next_pos({dx}, {dy}) - #count {around[dy, dx]}")
            y, x = node
            # print_matrix(count[y - 1 : y + 2, x - 1 : x + 2], 'check count again')
            next_y = y + dy - 1
            next_x = x + dx - 1
            if dx == 0 or (dy == 0 and dx == 1):
                dy, dx = 2 - dy, 2 - dx
            temp_line = [[y, x, 0, 0], [next_y, next_x, dy - 1, dx - 1]]

            # print(f"temp_line")
            # print(temp_line, end="\n\n")

            if count[next_y, next_x] == 1 or count[next_y, next_x] >= 3:
                not_visited[next_y, next_x] = 1
                # MyDebug
                # print_matrix(not_visited, "nv: (1st if)")
                graph[tuple(temp_line[0][:2])][tuple(temp_line[-1][:2])] = temp_line
                temp_line_rev = list(reversed(temp_line))
                graph[tuple(temp_line[-1][:2])][tuple(temp_line[0][:2])] = temp_line_rev

                # export_image_from_line(image_size[0], image_size[1], temp_line, f"02-temp-node{n_node}-x{x}-y{y}")
                # n_node += 1
                continue
        
            while True:
                y, x = temp_line[-1][:2]
                not_visited[y, x] = 0
                around = np.multiply(
                    count[y - 1 : y + 2, x - 1 : x + 2],
                    not_visited[y - 1 : y + 2, x - 1 : x + 2],
                )

                next_pos = np.transpose(np.nonzero(around))
                if next_pos.shape[0] == 0:
                    break

                # update line
                next_y = y + next_pos[0][0] - 1
                next_x = x + next_pos[0][1] - 1
                dy, dx = next_y - y, next_x - x
                if dx == -1 or (dy == -1 and dx == 0):
                    dy, dx = -dy, -dx
                temp_line.append([next_y, next_x, dy, dx])
                not_visited[next_y, next_x] = 0

                # check end condition
                if count[next_y, next_x] == 1 or count[next_y, next_x] >= 3:
                    # if len(temp_line) > 10:
                    graph[tuple(temp_line[0][:2])][tuple(temp_line[-1][:2])] = temp_line
                    temp_line_rev = list(reversed(temp_line))
                    graph[tuple(temp_line[-1][:2])][
                        tuple(temp_line[0][:2])
                    ] = temp_line_rev
                    not_visited[next_y, next_x] = 1
                    break
            
            # export_image_from_line(image_size[0], image_size[1], temp_line, f"02-temp-node{n_node}-x{x}-y{y}")
            n_node += 1        
        not_visited[node[0], node[1]] = 1

    # (2) find all possible lines by graph backtracking
    lines_node = []
    visited_node = dict()
    finished_node = dict()
    for node in nodes:
        visited_node[node] = False
        finished_node[node] = False
    
    # logger_backtrack.info("---------------- START LOGGING -------------------")
    # logger_backtrack.info(f"lines_node: {lines_node}")
    # logger_backtrack.info(f"graph: {graph}")

    for node in nodes:
        # logger_backtrack.info(f"Focus at node: {node}")
        # logger_backtrack.info(f"\t [recall] #all node: {len(visited_node)}")
        if not finished_node[node]:
            temp = [node]
            visited_node[node] = True
            finished_node[node] = True
            # logger_backtrack.info(f"\t Go to {backtrack.__name__}()")
            
            # logger_backtrack.info(f"\t\t graph[node]: \t{graph[node]}")
            backtrack(lines_node, temp, graph, visited_node, finished_node, node)
    # logger_backtrack.info(f"lines_node: {lines_node}")
    # logger_backtrack.info(f"#lines_node: {len(lines_node)}")

    # TODO: Check after backtrack
    connected_node = get_path_points(graph, lines_node)
    export_image_from_lines(image_size[0], image_size[1], connected_node, "03-after-backtrack")

    # (3) filter lines with length, direction criteria
    lines = []
    lines_before_filter = []
    for line_node in lines_node:
        num_node = len(line_node)
        if num_node == 1:
            continue
        wrong = False
        line = []
        prev, cur = None, line_node[0]
        for i in range(1, num_node):
            nxt = line_node[i]
            # if the inner product of two connected line segments vectors is <0, discard it
            if (
                i > 1
                and (cur[0] - prev[0]) * (nxt[0] - cur[0])
                + (cur[1] - prev[1]) * (nxt[1] - cur[1])
                < 0
            ):
                wrong = True
                break
            line.extend(graph[cur][nxt])
            prev, cur = cur, nxt
        # if the length is <10, discard it
        lines_before_filter.append(line)
        if wrong or len(line) < 10:
            continue
        lines.append(line)
    export_image_from_lines(image_size[0], image_size[1], lines_before_filter, "04-before-filter")
    return lines


### 2. Choose three lines ###


# classify lines using l2 distance with centers in feature space
# remain at most 3 lines
def classify_lines(centers, lines, image_height, image_width):
    # TODO: edit number of cluster
    classified_lines = [None, None, None]
    line_idx = [None, None, None]
    nearest = [1e9, 1e9, 1e9]

    if n_cluster == 4:
        classified_lines.append(None)
        line_idx.append(None)
        nearest.append(1e9)

    feature_list = np.empty((0, 24))
    for line in lines:
        feature = extract_feature(line, image_height, image_width)
        feature_list = np.vstack((feature_list, feature))

    num_lines = len(lines)
    # for i in range(3):
    for i in range(n_cluster):
        center = centers[i]
        for j in range(num_lines):
            chosen = False
            for k in range(i - 1):
                if line_idx[k] == j:
                    chosen = True
                    break
            if chosen:
                continue
            feature = feature_list[j]
            dist = np.linalg.norm(feature - center)
            if dist < nearest[i]:
                nearest[i] = dist
                classified_lines[i] = lines[j]
                line_idx[i] = j

    return classified_lines


### 3. Color each line ###


# color lines with BGR
def color(skel_img, lines):
    color_list = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # [B,G,R]

    num_lines = len(lines)
    colored_img = cv2.cvtColor(skel_img, cv2.COLOR_GRAY2RGB)
    for i in range(num_lines):
        line = lines[i]
        for y, x, _, _ in line:
            colored_img[y, x] = color_list[i]

    return colored_img


### Others ###


# extract feature from a line
def extract_feature(line, image_height, image_width):
    # feature = [min_y, min_x, max_y, max_x] + mean of direction info(dy,dx) * N intervals
    # => (2N+4)-dim
    image_size = np.array([image_height, image_width], dtype=np.float32)
    feature = np.append(
        np.min(line, axis=0)[:2] / image_size, np.max(line, axis=0)[:2] / image_size
    )
    # print(f"feature: {feature}")
    feature *= 10
    # print(f"feature*10: {feature}")
    N = 10
    step = len(line) // N
    for i in range(N):
        l = line[i * step : (i + 1) * step]
        # print(f"l{i+1}: {l}")
        feature = np.append(feature, np.mean(l, axis=0)[2:])
        # print(f"np.mean: {np.mean(l, axis=0)[2:]}", end="\n\n")
    
    # print(f"feature (after): {feature}")
    # print_matrix(feature, f"feature (after)")
    return feature


def extract_feature_2(line: list, image_height: int, image_width: int):
    image_size = np.array([image_height, image_width], dtype=np.float32)

    min_coords = np.min(line, axis=0)[:2] / image_size
    max_coords = np.max(line, axis=0)[:2] / image_size

    feature = np.concatenate((min_coords, max_coords))

    N = 10
    step = len(line) // N

    for i in range(N):
        l = line[i * step : (i + 1) * step]
        
        if len(l) > 0:
            mean_direction = np.mean(np.diff(l, axis=0), axis=0)
            feature = np.append(feature, mean_direction)
        else:
            feature = np.append(feature, [0, 0])
    return feature

# find 3 cluster centers in feature space
# we can use pre-trained centers for testing
def get_cluster_centers(new_centers=False):
    if new_centers:
        # prepare good samples
        good = [
            12,
            104,
            193,
            212,
            220,
            249,
            256, 
            295,
            304,
            396,
            402,
            487,
            698,
            908,
            992,
        ]

        # TODO: Add cutsom output_path 
        my_output_path = f"./output/good_sample"
        check_path_compatibility(my_output_path)
        remove_all_files(my_output_path)
        remove_all_files(f"./output/good_sample_skel")

        # for idx in good:
        # for idx in my_good_3:
        for idx in my_good_4:
            rectified = rectify(idx)

            # TODO: Use cutsom output_path
            # cv2.imwrite("good_sample/image" + str(idx) + ".png", rectified)
            cv2.imwrite(f"{my_output_path}/image" + str(idx) + ".png", rectified)

        # put all data in feature space
        data = np.empty((0, 24))
        # data = np.empty((0, 44))

        # for img_path in glob.glob("good_sample/*.png"):
        for img_path in glob.glob(f"{my_output_path}/*.png"):
            img = cv2.imread(img_path)

            # MyDebug
            check_loaded_image(img)

            # TODO: Fix skeletonize input format
            # skel_img = cv2.cvtColor(skeletonize(img), cv2.COLOR_BGR2GRAY)

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            skeleton = skeletonize(gray_img)
            skel_img = skeleton.astype(np.uint8) * 255
            
            image_name = get_filename_without_extension(img_path)
            export_image(skel_img, f"{image_name}-skel.png", f"./output/good_sample_skel")

            # TODO: Warning use original group function
            # lines = group(skel_img)

            # MyDebug:
            print(f"Enter group_original(): {image_name}")
            lines = group_original(skel_img)
            
            for line in lines:
                # TODO: Add cutsom image size?
                # feature = extract_feature(line, 1024, 1024)
                feature = extract_feature(line, 1024, 1024)
                # print(f"feature: {feature}")
                data = np.vstack((data, feature))

        # k-means clustering (k=3)
        # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

        # ret, label, centers = cv2.kmeans(
        #     # TODO: Change k=4
        #     # data.astype(np.float32), 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        #     data.astype(np.float32), n_cluster, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        # )

        print_matrix(data, "data: ")
        plot_dbscan(data)
        # plot_silhouette_scores(data, 10)
        # plot_cluster_pca(data, n_cluster)

        print(f"Center: {type(centers)}")
        print(centers)

        # sort centers according to max_y
        centers = list(centers)
        centers.sort(key=lambda x: x[3])
    else:
        centers = [
            np.array(
                [
                    5.232849,
                    4.881592,
                    6.3223267,
                    6.64093,
                    0.8113839,
                    0.655735,
                    0.82874316,
                    0.74796075,
                    0.7993417,
                    0.8345605,
                    0.68143266,
                    0.90320605,
                    0.5769709,
                    0.9721149,
                    0.53258324,
                    0.98307294,
                    0.4804058,
                    0.9829783,
                    0.36796156,
                    0.99141085,
                    0.24345541,
                    0.99082345,
                    0.30017138,
                    0.9736235,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    5.645419,
                    4.169626,
                    7.126243,
                    6.0026045,
                    0.3532842,
                    0.928315,
                    0.4692493,
                    0.9680717,
                    0.578683,
                    0.9680221,
                    0.7227269,
                    0.9454175,
                    0.7741767,
                    0.9495983,
                    0.7802345,
                    0.89685285,
                    0.8743354,
                    0.8478447,
                    0.85625464,
                    0.82669544,
                    0.88459945,
                    0.8000444,
                    0.8956431,
                    0.74734426,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    5.755994,
                    3.8910964,
                    8.680631,
                    5.3926454,
                    0.4247846,
                    0.93111324,
                    0.6940754,
                    0.9203782,
                    0.8567455,
                    0.767301,
                    0.9177662,
                    0.6054738,
                    0.9801044,
                    0.47111732,
                    0.9812451,
                    0.34593108,
                    0.97122467,
                    0.28715244,
                    0.9036454,
                    0.26124895,
                    0.8069528,
                    0.25324377,
                    0.59989274,
                    0.32016128,
                ],
                dtype=np.float32,
            ),
        ] 
    return centers


def group_original(img):
    # (1) build a graph

    # (1)-1 find all nodes
    count = np.zeros(img.shape)
    nodes = []

    for j in range(1, img.shape[0] - 1):
        for i in range(1, img.shape[1] - 1):
            if img[j, i] == 0:
                continue
            count[j, i] = np.count_nonzero(img[j - 1 : j + 2, i - 1 : i + 2]) - 1
            if count[j, i] == 1 or count[j, i] >= 3:
                nodes.append((j, i))

    # sort nodes to traverse from upper-left to lower-right
    logger_classify.info(f"\t nodes \t: {nodes}")
    nodes.sort(key=lambda x: x[0] + x[1])
    logger_classify.info(f"\t nodes (sorted): {nodes}")

    # (1)-2 save all connections
    graph = dict()
    for node in nodes:
        graph[node] = dict()

    not_visited = np.ones(img.shape)

    for node in nodes:
        y, x = node
        not_visited[y, x] = 0
        around = np.multiply(
            count[y - 1 : y + 2, x - 1 : x + 2],
            not_visited[y - 1 : y + 2, x - 1 : x + 2],
        )
        next_pos = np.transpose(np.nonzero(around))
        if next_pos.shape[0] == 0:
            continue

        for dy, dx in next_pos:
            y, x = node
            next_y = y + dy - 1
            next_x = x + dx - 1
            if dx == 0 or (dy == 0 and dx == 1):
                dy, dx = 2 - dy, 2 - dx
            temp_line = [[y, x, 0, 0], [next_y, next_x, dy - 1, dx - 1]]
            if count[next_y, next_x] == 1 or count[next_y, next_x] >= 3:
                not_visited[next_y, next_x] = 1
                graph[tuple(temp_line[0][:2])][tuple(temp_line[-1][:2])] = temp_line
                temp_line_rev = list(reversed(temp_line))
                graph[tuple(temp_line[-1][:2])][tuple(temp_line[0][:2])] = temp_line_rev
                continue
        
            while True:
                y, x = temp_line[-1][:2]
                not_visited[y, x] = 0
                around = np.multiply(
                    count[y - 1 : y + 2, x - 1 : x + 2],
                    not_visited[y - 1 : y + 2, x - 1 : x + 2],
                )
                next_pos = np.transpose(np.nonzero(around))
                if next_pos.shape[0] == 0:
                    break

                # update line
                next_y = y + next_pos[0][0] - 1
                next_x = x + next_pos[0][1] - 1
                dy, dx = next_y - y, next_x - x
                if dx == -1 or (dy == -1 and dx == 0):
                    dy, dx = -dy, -dx
                temp_line.append([next_y, next_x, dy, dx])
                not_visited[next_y, next_x] = 0

                # check end condition
                if count[next_y, next_x] == 1 or count[next_y, next_x] >= 3:
                    # if len(temp_line) > 10:
                    graph[tuple(temp_line[0][:2])][tuple(temp_line[-1][:2])] = temp_line
                    temp_line_rev = list(reversed(temp_line))
                    graph[tuple(temp_line[-1][:2])][
                        tuple(temp_line[0][:2])
                    ] = temp_line_rev
                    not_visited[next_y, next_x] = 1

                    # MyDebug
                    # print(f"end loop => count:{count[next_y, next_x]}")
                    break

        not_visited[node[0], node[1]] = 1

    # (2) find all possible lines by graph backtracking
    lines_node = []
    visited_node = dict()
    finished_node = dict()
    for node in nodes:
        visited_node[node] = False
        finished_node[node] = False

    for node in nodes:
        if not finished_node[node]:
            temp = [node]
            visited_node[node] = True
            finished_node[node] = True
            backtrack(lines_node, temp, graph, visited_node, finished_node, node)

    # (3) filter lines with length, direction criteria
    lines = []
    for line_node in lines_node:
        num_node = len(line_node)
        if num_node == 1:
            continue
        wrong = False
        line = []
        prev, cur = None, line_node[0]
        for i in range(1, num_node):
            nxt = line_node[i]
            # if the inner product of two connected line segments vectors is <0, discard it
            if (
                i > 1
                and (cur[0] - prev[0]) * (nxt[0] - cur[0])
                + (cur[1] - prev[1]) * (nxt[1] - cur[1])
                < 0
            ):
                wrong = True
                break
            line.extend(graph[cur][nxt])
            prev, cur = cur, nxt
        # if the length is <10, discard it
        if wrong or len(line) < 10:
            continue
        lines.append(line)

    return lines

def classify(path_to_palmline_image):
    # load (rectified) test data
    # num_data = 10
    # load_data(num_data)

    # get cluster centers
    # centers = get_cluster_centers()
    if n_cluster == 4:
        centers = get_cluster_centers(True)
    else:
        centers = get_cluster_centers(True)
    print(f"#centers: {len(centers)}")   

    palmline_img = cv2.imread(path_to_palmline_image)
    # show_image(palmline_img, "Palm Line")

    # kernel = np.ones((3, 3), np.uint8)
    # dilated = cv2.dilate(palmline_img, kernel, iterations=3)
    # eroded = cv2.erode(dilated, kernel, iterations=3)
    
    # TODO: Fix
    gray_img = cv2.cvtColor(palmline_img, cv2.COLOR_BGR2GRAY)

    skeleton = skeletonize(gray_img)
    skel_img = skeleton.astype(np.uint8) * 255

    # TODO: comment code
    # skel_img = cv2.cvtColor(skeletonize(palmline_img), cv2.COLOR_BGR2GRAY)

    # cv2.imwrite('results/skel.jpg',skel_img)
    # cv2.imwrite(f'{output_path}/00-skel.png',skel_img)
    export_image(skel_img, "00-skel.png", output_path)

    lines = group(skel_img)  # get candidate lines
    print(f"#group lines: {len(lines)}")
    # print(lines)
    logger.info(f"\t number of lines (group by backtrack): {len(lines)}")

    image_size = list(gray_img.shape[:2])
    export_image_from_lines(image_size[0], image_size[1], lines, "05-after-group")

    lines = classify_lines(
        centers, lines, palmline_img.shape[0], palmline_img.shape[1]
    )  # choose 3 lines from candidates
    # colored_img = color(skel_img, classified_lines) # color 3 lines (RGB)
    # show_image(skel_img, "Skel")
    return lines


def export_image_from_lines(width: int, height: int, lines: list, output_pattern_name: str = "line"):
    line_count = 1

    check_path_compatibility(output_path)

    logger.info(f"Save image")
    for line in lines:
        binary_image = np.zeros((width, height), dtype=np.uint8)
        for point in line:
            binary_image[point[0], point[1]] = 255
        image_path = f"{output_path}/{output_pattern_name}-{line_count}.png"
        logger.info(f"\t save image: {image_path}")
        cv2.imwrite(image_path, binary_image)

        line_count += 1

def export_image_from_line(width: int, height: int, line: list, output_pattern_name: str = "line"):
    check_path_compatibility(output_path)

    binary_image = np.zeros((width, height), dtype=np.uint8)
    for point in line:
        binary_image[point[0], point[1]] = 255
    image_path = f"{output_path}/{output_pattern_name}.png"
    cv2.imwrite(image_path, binary_image)

    logger.info(f"Save image: {image_path}")
    
def get_path_points(graph: dict, lines_node: list):
    all_path_points = []
    for line in lines_node:
        if len(line) > 1:
            path_points = []
            for i in range(len(line)-1):
                start_node = line[i]
                end_node = line[i+1]
                if start_node in graph and end_node in graph[start_node]:
                    path_points.extend(graph[start_node][end_node])
            all_path_points.append(path_points)
    return all_path_points


# mask_path = "./sample/line-cross-100x100.png"
# mask_path = "./sample/line-complex-100x100.png"
mask_path = "./sample/Results-stwcrack-Alex0310-Mask.png"


# mask_path = "./sample/test-mask.png"

# mask_path = "./sample/line-cross-50x50.png"
# mask_path = "./sample/simple-line-cross-15x15.png"
# mask_path = "./sample/simple-line-10x10.png"
# mask_path = "./sample/line-100x100.png"

change_to_main_root()
print(f"current path: {os.getcwd()}")

image_mask = cv2.imread(mask_path)
check_loaded_image(image_mask)
image_size = list(image_mask.shape[:2])
image_name = get_filename_without_extension(mask_path)
n_cluster = 4
# n_cluster = 3


my_good = [
    13,  
    97,
    106,
    112,
    352,
    356,
    382,
    401,
    744,
    756,
    762,
    1083,
]

my_good_2 = [
    5,  
    13,
    28,
    106,
    108,
    112,
    140,
    # 142,
    # 477,
    # 498,
    # 513,
    # 596,
    # 616,
    # 622,
    # 648,
    # 652,
    # 661,
    # 671,
    # 673,
    # 685,
    # 691,
    # 762,
    # 744,
    # 750,
    # 756,
    # 762,
    # 763,
    # 817,
    # 818,
    # 819,
]

my_good_3 = [
    5,
    28,
    106,
    108,
    142,
    477,
    498,
    596,
    616,
    622,
    648,
    652,
    661,
    671,
    673,
    685,
    691,
    762,
    750,
    818,
    819,

]

centers_new1 = [
            np.array(
                [
                    5.4170465, 3.8961558, 6.3814063, 5.6454954, 0.3250605, 0.8100688,
                    0.34097046, 0.9526658, 0.40841228, 0.8891923, 0.44644922, 0.82036567,
                    0.5481257, 0.9231054, 0.6076798, 0.8870097, 0.6225118, 0.81023467,
                    0.68190926, 0.61726826, 0.5856578, 0.70883733, 0.5123793, 0.67894644,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    5.2557163, 3.7675252, 7.0953894, 5.6750093, 0.34175143, 0.9627604,
                    0.4353329, 0.98086256, 0.5270465, 0.8735952, 0.56606346, 0.70149636,
                    0.77529895, 0.78744084, 0.87813234, 0.65993667, 0.9252021, 0.52762437,
                    0.94468206, 0.47014883, 0.91631913, 0.5546346, 0.66775674, 0.47080833,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    5.601473, 3.4987862, 7.395289, 6.690849, 0.23009704, 0.9008044,
                    0.45073014, 0.8196588, 0.8776976, 0.63294667, 0.927213, 0.5897087,
                    0.7820175, 0.41643777, 0.95956403, 0.14166228, 0.8708497, 0.27273476,
                    0.6948253, 0.5874401, 0.3956642, 0.99854386, 0.1832288, 0.9599472,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    5.438477, 3.5410795, 8.200345, 5.5458174, 0.15321459, 0.8810476,
                    0.39036936, 0.90064764, 0.7141441, 0.7426404, 0.695159, 0.61015314,
                    0.8494364, 0.4329585, 0.89095706, 0.28980485, 0.93531924, 0.2557217,
                    0.94224215, 0.21574447, 0.8467629, 0.33156824, 0.83431816, 0.4950585,
                ],
                dtype=np.float32,
            ),
        ]

my_good_4 = [
    12,
    104,
    249,
    256,
    396,
    402,
    487,
    698,
    908,
    992
]


output_path = "output/process-lines"
remove_all_files(output_path)

lines = classify(mask_path)
print(len(lines))
export_image_from_lines(image_size[0], image_size[1], lines, "06-after-classify")
# cv2.imshow("Skel", cv2.imread("./output/process-lines/skel.png"))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# skel_img = load_image("./output/process-lines/skel.png")
# show_image(skel_img)

print(f"#lines: {len(lines)}")
