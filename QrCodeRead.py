import cv2
import numpy as np
from math import atan2, pi


class QrCodeRead:
    threshold = None
    binary_image = None
    finder_pattern_centers = []
    x_medians = []
    y_medians = []
    angle = 1
    module_width = 0
    module_height = 0
    qr_code_version = -1

    def __init__(self, qr_code_image):
        try:
            self.raw_image = cv2.imread(qr_code_image)
        except FileNotFoundError:
            raise FileNotFoundError(f'The file {qr_code_image} does not exist or is not valid.')

    def read_qr_code(self):
        self.correct_image_rotation()
        self.threshold = (self.raw_image.max() - self.raw_image.min()) // 2  # light / dark threshold value
        self.binary_image = np.where(np.max(self.raw_image, axis=2) > self.threshold, 0, 1)  # spaces=0, pixels=1
        self.x_medians, self.module_width = self.get_medians(self.binary_image)
        self.y_medians, self.module_height = self.get_medians(np.transpose(self.binary_image))
        self.finder_pattern_centers = self.intersect_medians(self.x_medians, self.y_medians)
        self.get_image_and_module_widths()

        a = 1

    def get_image_and_module_widths(self):
        upper_left_pattern = [p for p in self.finder_pattern_centers if p[0] < self.binary_image.shape[0] // 2 and
                              p[1] < self.binary_image.shape[1] // 2][0]
        upper_right_pattern = [p for p in self.finder_pattern_centers if p[0] > self.binary_image.shape[0] // 2 and
                               p[1] < self.binary_image.shape[1] // 2][0]
        lower_left_pattern = [p for p in self.finder_pattern_centers if p[0] < self.binary_image.shape[0] // 2 and
                              p[1] > self.binary_image.shape[1] // 2][0]

        upper_left_width = [y[2] - y[1] for y in self.x_medians if y[0] == upper_left_pattern[1]][0]
        upper_right_width = [y[2] - y[1] for y in self.x_medians if y[0] == upper_right_pattern[1]][0]
        x_module_width = round((upper_left_width + upper_right_width) / 14)
        inter_module_distance = upper_right_pattern[0] - upper_left_pattern[0]
        self.qr_code_version = round(((inter_module_distance / x_module_width) - 10) / 4)

        upper_left_height = [x[2] - x[1] for x in self.y_medians if x[0] == upper_left_pattern[0]][0]
        lower_left_height = [x[2] - x[1] for x in self.y_medians if x[0] == lower_left_pattern[0]][0]
        y_module_height = round((upper_left_height + lower_left_height) / 14)

        a = 1

    def correct_image_rotation(self):
        half_a_degree = 0.5 * pi / 180
        while abs(self.angle) > half_a_degree:
            self.threshold = (self.raw_image.max() - self.raw_image.min()) // 2  # light / dark threshold value
            self.binary_image = np.where(np.max(self.raw_image, axis=2) > self.threshold, 0, 1)  # spaces=0, pixels=1
            self.x_medians, self.module_width = self.get_medians(self.binary_image)
            self.y_medians, self.module_height = self.get_medians(np.transpose(self.binary_image))
            self.finder_pattern_centers = self.intersect_medians(self.x_medians, self.y_medians)
            self.angle = self.determine_angle_of_image(self.finder_pattern_centers)
            self.raw_image = self.rotate_image(self.raw_image, self.angle)

    @staticmethod
    def rotate_image(image, angle):
        degrees = angle * 180 / pi
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, degrees, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        return result

    @staticmethod
    def determine_angle_of_image(mid_points):
        min_x = min(m[0] for m in mid_points)
        min_y = min(m[1] for m in mid_points)
        max_x = max(m[0] for m in mid_points)
        min_x_min_y = [m for m in mid_points if m[0] == min_x and m[1] == min_y]    # only has entry for 0 rotation
        if len(min_x_min_y) == 0:
            # We have a non-zero rotation (or a distortion)
            rightmost = [m for m in mid_points if m[0] == max_x]
            leftmost = sorted([m for m in mid_points if m[0] != max_x], key=lambda x: x[1])
            y_diff = rightmost[0][1] - leftmost[0][1]
            x_diff = rightmost[0][0] - leftmost[0][0]
            angle = atan2(y_diff, x_diff)   # +ve values ==> clockwise (positive) rotation
        else:
            angle = 0

        return angle

    @staticmethod
    def intersect_medians(x_medians, y_medians):
        finder_pattern_centers = []
        for x in x_medians:
            mid_point = [x[0]]
            for y in y_medians:
                if y[1] <= x[0] <= y[2] and x[1] <= y[0] <= x[2]:
                    mid_point.insert(0, y[0])
                    finder_pattern_centers.append(mid_point)
                    break
        return finder_pattern_centers

    def get_medians(self, binary_array):
        candidates, pixels_per_module = self.get_candidate_finder_patterns(binary_array)
        sorted_candidates = sorted(candidates, key=lambda x: (x[1], x[2], x[0]))
        clusters = self.cluster_groups(sorted_candidates)
        return self.find_cluster_medians(clusters), pixels_per_module

    @staticmethod
    def get_candidate_finder_patterns(binary_array):
        """ Look for the finder patterns using the pixel ratios 1:1:3:1:1, with a tolerance of 0.5 on each. """
        candidates = []
        pixels_per_module = -1
        groupings = [np.diff(np.flatnonzero(np.concatenate(([True], a[1:] != a[:-1], [True])))) for a in binary_array]
        for row_index, row in enumerate(groupings):
            for col5 in range(len(row) - 5):
                sub_array = row[col5: col5 + 5]     # sliding window of 5 entries in the current row
                start_position_in_row = np.sum(groupings[row_index][:col5])   # pixel index of start of sliding window
                norm = np.divide(sub_array, min(sub_array))     # normalised sub-array (lowest number will be 1)

                # Valid pattern must start with a dark pixel
                is_pixel = binary_array[row_index][start_position_in_row] == 1

                if is_pixel and \
                        0.5 <= norm[0] <= 1.5 and \
                        0.5 <= norm[1] <= 1.5 and \
                        2.5 <= norm[2] <= 3.5 and \
                        0.5 <= norm[3] <= 1.5 and \
                        0.5 <= norm[4] <= 1.5:
                    pixels_per_module = sub_array[np.where(norm == 1)][0]
                    end_position_in_row = np.sum(groupings[row_index][:col5 + 5])  # pixel index: end of sliding window
                    candidates.append((row_index, start_position_in_row, end_position_in_row))

        return candidates, pixels_per_module

    @staticmethod
    def cluster_groups(list_to_cluster):
        """ Clusters a list of x & y values into neighbouring groups using Euclidean distance """
        # ASSUMPTION: The nearest neighbour to any given pixel will never be more than 1.9 pixels away.
        #             Mathematically, the upper bound is sqrt(2.0), but we use 1.9 to be certain.
        max_distance = 1.9 * 1.9
        clusters = [[]]
        for entry_to_cluster in list_to_cluster:
            append_list = []    # Used to determine if an entry belongs to more than 1 cluster.  If so, join clusters.
            for cluster_index, curr_cluster in enumerate(clusters):
                if len(curr_cluster) == 0:    # Only happens when creating a new cluster
                    curr_cluster.append(entry_to_cluster)
                    append_list.append(cluster_index)
                    continue
                for cluster_element in curr_cluster:
                    x_diff = cluster_element[1] - entry_to_cluster[1]
                    y_diff = cluster_element[0] - entry_to_cluster[0]
                    if x_diff * x_diff + y_diff * y_diff <= max_distance:
                        if len(append_list) == 0:
                            curr_cluster.append(entry_to_cluster)
                            append_list.append(cluster_index)
                        else:
                            # need to append current list to list with index of append_list[0]
                            clusters[append_list[0]] += [x for x in curr_cluster]
                            clusters[cluster_index] = []
                        break
            if len(append_list) == 0:
                clusters.append([entry_to_cluster])

        return clusters

    @staticmethod
    def find_cluster_medians(clusters, number_of_clusters=3):
        # ASSUMPTION:
        # Finder patterns are designed to create the densest clusters of 1:1:3:1:1 pixel patterns in the QR Code
        # and masking on the rest of the code helps to ensure this.
        # Therefore, return the 3 sublist with the most entries
        longest_clusters = sorted(clusters, key=len, reverse=True)[:number_of_clusters]
        medians = []
        for cluster in longest_clusters:
            sorted_cluster = sorted(cluster)    # sort current cluster in ascending order
            median = sorted_cluster[len(sorted_cluster) // 2]   # take the element at the mid-point
            medians.append(median)
        return medians


if __name__ == '__main__':
    qrr = QrCodeRead('Untitled4-3.png')
    qrr.read_qr_code()
