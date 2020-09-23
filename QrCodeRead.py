import cv2
import numpy as np
from math import atan2, pi
from QrCode import QrCode


class QrCodeRead:
    threshold = None
    binary_image = None
    finder_pattern_centers = []
    sampling_grid = None
    x_medians = []
    y_medians = []
    lower_left_pattern = []
    upper_left_pattern = []
    upper_right_pattern = []
    angle = 1
    module_width = 0
    module_height = 0
    qr_code_version = -1
    width_in_modules = 0
    height_in_modules = 0

    def __init__(self, qr_code_image):
        try:
            self.raw_image = cv2.imread(qr_code_image)
        except FileNotFoundError:
            raise FileNotFoundError(f'The file {qr_code_image} does not exist or is not valid.')

    def read_qr_code(self):
        space_value = 0
        pixel_value = 1
        self.correct_image_rotation()
        self.get_finder_pattern_centers(space_value, pixel_value)
        if len(self.finder_pattern_centers) < 3:
            # Need to try inverted light / dark modules
            self.correct_image_rotation(1, 0)
            space_value = 1
            pixel_value = 0

        self.get_image_and_module_dimensions()
        if self.qr_code_version == 1:
            self.reset_module_dimensions_from_timing_patterns()
        else:
            self.get_alignment_pattern_positions(space_value, pixel_value)

        self.populate_sampling_grid(space_value, pixel_value)
        a = 1

    def populate_sampling_grid(self, space_value=0, pixel_value=1):
        pass

    def get_alignment_pattern_positions(self, space_value=0, pixel_value=1):
        alignment_positions = QrCode.alignment_pattern_locations[self.qr_code_version]
        self.width_in_modules = alignment_positions[-1] + 7
        self.height_in_modules = alignment_positions[-1] + 7
        self.initialise_sampling_grid()
        pattern_limit = int(round((len(alignment_positions) - 1) / 2))
        self.resolve_upper_left_alignment_patterns(alignment_positions, pattern_limit, space_value, pixel_value)
        self.resolve_upper_right_alignment_patterns(alignment_positions, pattern_limit, space_value, pixel_value)
        self.resolve_lower_left_alignment_patterns(alignment_positions, pattern_limit, space_value, pixel_value)

    def resolve_upper_left_alignment_patterns(self, alignment_patterns, pattern_limit, space_value=0, pixel_value=1):
        top_row_positions = [[self.upper_left_pattern[0] + 3 * self.module_width,
                              self.upper_left_pattern[1] + 3 * self.module_height]]
        module_diff_in_pixels = (alignment_patterns[1] - alignment_patterns[0]) * self.module_width
        for x in range(1, pattern_limit + 1):   # ignore first element along top row, since it will be a finder pattern
            candidate = [module_diff_in_pixels + top_row_positions[x - 1][0], top_row_positions[x - 1][1]]
            candidate = self.tweak_alignment_pattern_position(candidate, space_value, pixel_value)
            top_row_positions.append(candidate)
            self.sampling_grid[alignment_patterns[0]][alignment_patterns[x]] = candidate

        left_row_positions = [[self.upper_left_pattern[0] + 3 * self.module_width,
                              self.upper_left_pattern[1] + 3 * self.module_height]]
        module_diff_in_pixels = (alignment_patterns[1] - alignment_patterns[0]) * self.module_height
        for y in range(1, pattern_limit + 1):   # ignore first element along top row, since it will be a finder pattern
            candidate = [left_row_positions[y - 1][0], module_diff_in_pixels + left_row_positions[y - 1][1]]
            candidate = self.tweak_alignment_pattern_position(candidate, space_value, pixel_value)
            left_row_positions.append(candidate)
            self.sampling_grid[alignment_patterns[y]][alignment_patterns[0]] = candidate
            for x in range(1, pattern_limit + 1):
                candidate = [top_row_positions[x][0], left_row_positions[y][1]]
                candidate = self.tweak_alignment_pattern_position(candidate, space_value, pixel_value)
                self.sampling_grid[alignment_patterns[y]][alignment_patterns[x]] = candidate

    def resolve_upper_right_alignment_patterns(self, alignment_patterns, pattern_limit, space_value=0, pixel_value=1):
        top_row_positions = [[self.upper_right_pattern[0] - 3 * self.module_width,
                              self.upper_right_pattern[1] + 3 * self.module_height]]
        module_diff_in_pixels = (alignment_patterns[1] - alignment_patterns[0]) * self.module_width
        for x in range(len(alignment_patterns) - 2, pattern_limit, -1):   # ignore last element for same reason
            candidate = [top_row_positions[0][0] - module_diff_in_pixels, top_row_positions[0][1]]
            candidate = self.tweak_alignment_pattern_position(candidate, space_value, pixel_value)
            top_row_positions.insert(0, candidate)
            self.sampling_grid[alignment_patterns[0]][alignment_patterns[x]] = candidate

        right_row_positions = [[self.upper_right_pattern[0] - 3 * self.module_width,
                                self.upper_right_pattern[1] + 3 * self.module_height]]
        module_diff_in_pixels = (alignment_patterns[1] - alignment_patterns[0]) * self.module_height
        for y in range(1, pattern_limit + 1):   # ignore first element along top row, since it will be a finder pattern
            candidate = [right_row_positions[y - 1][0], module_diff_in_pixels + right_row_positions[y - 1][1]]
            candidate = self.tweak_alignment_pattern_position(candidate, space_value, pixel_value)
            right_row_positions.append(candidate)
            self.sampling_grid[alignment_patterns[y]][alignment_patterns[-1]] = candidate
            for x in range(1, pattern_limit):
                candidate = [top_row_positions[-1 - x][0], right_row_positions[y][1]]
                candidate = self.tweak_alignment_pattern_position(candidate, space_value, pixel_value)
                self.sampling_grid[alignment_patterns[y]][alignment_patterns[-1 - x]] = candidate

    def resolve_lower_left_alignment_patterns(self, alignment_patterns, pattern_limit, space_value=0, pixel_value=1):
        bottom_row_positions = [[self.lower_left_pattern[0] + 3 * self.module_width,
                                 self.lower_left_pattern[1] - 3 * self.module_height]]
        module_diff_in_pixels = (alignment_patterns[1] - alignment_patterns[0]) * self.module_width
        for x in range(1, pattern_limit + 1):   # ignore first element along top row, since it will be a finder pattern
            candidate = [module_diff_in_pixels + bottom_row_positions[x - 1][0], bottom_row_positions[x - 1][1]]
            candidate = self.tweak_alignment_pattern_position(candidate, space_value, pixel_value)
            bottom_row_positions.append(candidate)
            self.sampling_grid[alignment_patterns[-1]][alignment_patterns[x]] = candidate

        left_row_positions = [[self.lower_left_pattern[0] + 3 * self.module_width,
                               self.lower_left_pattern[1] - 3 * self.module_height]]
        module_diff_in_pixels = (alignment_patterns[1] - alignment_patterns[0]) * self.module_height
        for y in range(1, pattern_limit + 1):   # ignore first element along top row, since it will be a finder pattern
            candidate = [left_row_positions[y - 1][0], left_row_positions[y - 1][1] - module_diff_in_pixels]
            candidate = self.tweak_alignment_pattern_position(candidate, space_value, pixel_value)
            left_row_positions.append(candidate)
            self.sampling_grid[alignment_patterns[-1 - y]][alignment_patterns[0]] = candidate
            for x in range(1, pattern_limit + 1):
                candidate = [bottom_row_positions[x][0], left_row_positions[y][1]]
                candidate = self.tweak_alignment_pattern_position(candidate, space_value, pixel_value)
                self.sampling_grid[alignment_patterns[-1 - y]][alignment_patterns[x]] = candidate

    def tweak_alignment_pattern_position(self, candidate, space_value, pixel_value):
        if self.binary_image[candidate[1], candidate[0]] == space_value:
            candidate = self.find_nearest_pixel(candidate, pixel_value)
        self.center_between_spaces_horizontally(candidate, space_value, pixel_value)
        self.center_between_spaces_vertically(candidate, space_value, pixel_value)
        return candidate

    def center_between_spaces_vertically(self, test_pixel, space_value=0, pixel_value=1):
        up_dist = 0
        down_dist = 0
        for i in range(1, 3 * self.module_height):
            if up_dist == 0 and self.binary_image[test_pixel[1] - i, test_pixel[0]] == space_value and \
                    self.binary_image[test_pixel[1] - i - 1, test_pixel[0]] == pixel_value:
                up_dist = i
            if down_dist == 0 and self.binary_image[test_pixel[1] + i, test_pixel[0]] == space_value and \
                    self.binary_image[test_pixel[1] + i + 1, test_pixel[0]] == pixel_value:
                down_dist = i
            if up_dist != 0 and down_dist != 0:
                break
        test_pixel[1] += (down_dist - up_dist) // 2

    def center_between_spaces_horizontally(self, test_pixel, space_value=0, pixel_value=1):
        left_dist = 0
        right_dist = 0
        for i in range(1, 3 * self.module_width):
            if left_dist == 0 and self.binary_image[test_pixel[1], test_pixel[0] - i] == space_value and \
                    self.binary_image[test_pixel[1], test_pixel[0] - i - 1] == pixel_value:
                left_dist = i
            if right_dist == 0 and self.binary_image[test_pixel[1], test_pixel[0] + i] == space_value and \
                    self.binary_image[test_pixel[1], test_pixel[0] + i + 1] == pixel_value:
                right_dist = i
            if left_dist != 0 and right_dist != 0:
                break
        test_pixel[0] += (right_dist - left_dist) // 2

    def find_nearest_pixel(self, test_pixel, pixel_value=1):
        indices_of_pixels = []
        for radius in range(1, self.module_width):
            temp = self.binary_image[test_pixel[1] - radius: test_pixel[1] + radius + 1,
                                     test_pixel[0] - radius: test_pixel[0] + radius + 1]
            if np.any((temp == pixel_value)):
                indices_of_pixels = [[x - radius, y - radius] for y, x in list(zip(*np.where(temp == pixel_value)))]
                indices_of_pixels.sort(key=lambda x: x[0] * x[0] + x[1] * x[1])
                indices_of_pixels = [test_pixel[0] + indices_of_pixels[0][0], test_pixel[1] + indices_of_pixels[0][1]]
                break
        return indices_of_pixels

    # def get_alignment_pattern_centers(self, space_value=0, pixel_value=1):
    #     w = 4 * self.module_width
    #     h = 4 * self.module_height
    #     total_elements = 25 * self.module_width * self.module_height
    #     template = self.get_alignment_pattern_template(space_value, pixel_value)
    #
    #     for ap_idx, alignment_pattern in enumerate(self.alignment_pattern_centers):
    #         px = alignment_pattern[0]
    #         py = alignment_pattern[1]
    #         array_to_scan = self.binary_image[py - h: py + h, px - w: px + w]
    #
    #         max_x = -1
    #         max_y = -1
    #         max_percent = 0
    #         for y in range(len(array_to_scan) - len(template)):
    #             for x in range(len(array_to_scan[0]) - len(template[0])):
    #                 temp = array_to_scan[y: y + 5 * self.module_height, x: x + 5 * self.module_width]
    #
    #                 number_of_equal_elements = np.sum(temp == template)
    #                 percentage = number_of_equal_elements / total_elements
    #                 if percentage > max_percent:
    #                     max_percent = percentage
    #                     max_x = x
    #                     max_y = y
    #
    #         self.alignment_pattern_centers[ap_idx] = [px - int(1.5 * self.module_width) + max_x,
    #                                                   py - int(1.5 * self.module_height) + max_y]
    #
    # def get_alignment_pattern_template(self, space_value=0, pixel_value=1):
    #     template1 = [[pixel_value for _ in range(5 * self.module_width)] for _ in range(self.module_height)]
    #     template2 = [[pixel_value for _ in range(self.module_width)] +
    #                  [space_value for _ in range(3 * self.module_width)] +
    #                  [pixel_value for _ in range(self.module_width)] for _ in range(self.module_height)]
    #     template3 = [[pixel_value for _ in range(self.module_width)] + [space_value for _ in range(self.module_width)] +
    #                  [pixel_value for _ in range(self.module_width)] + [space_value for _ in range(self.module_width)] +
    #                  [pixel_value for _ in range(self.module_width)] for _ in range(self.module_height)]
    #     template = np.array(template1 + template2 + template3 + template2 + template1)
    #     return template

    def reset_module_dimensions_from_timing_patterns(self):
        col_in_pixels_start = self.upper_left_pattern[0] + self.module_width * 3
        col_in_pixels_end = self.upper_right_pattern[0] - self.module_width * 3
        row_in_pixels_start = self.upper_left_pattern[1] + self.module_height * 3
        row_in_pixels_end = self.lower_left_pattern[1] - self.module_height * 3
        half_col = int(self.module_width * 0.5)
        half_row = int(self.module_height * 0.5)
        image_row = self.binary_image[row_in_pixels_start, col_in_pixels_start - half_col: col_in_pixels_end + half_col]
        groupings = np.diff(np.flatnonzero(np.concatenate(([True], image_row[1:] != image_row[:-1], [True]))))
        self.width_in_modules = len(groupings) + 12
        self.module_width = round(np.average(groupings))
        image_col = self.binary_image[row_in_pixels_start - half_row: row_in_pixels_end + half_row, col_in_pixels_start]
        groupings = np.diff(np.flatnonzero(np.concatenate(([True], image_col[1:] != image_col[:-1], [True]))))
        self.module_height = round(np.average(groupings))
        self.height_in_modules = len(groupings) + 12
        self.initialise_sampling_grid()

    def initialise_sampling_grid(self):
        self.sampling_grid = [[[] for _ in range(self.width_in_modules)] for _ in range(self.height_in_modules)]
        self.sampling_grid[3][3] = self.upper_left_pattern
        self.sampling_grid[3][-4] = self.upper_right_pattern
        self.sampling_grid[-4][3] = self.lower_left_pattern

    def get_image_and_module_dimensions(self):
        self.upper_left_pattern = [p for p in self.finder_pattern_centers if p[0] < self.binary_image.shape[0] // 2 and
                                   p[1] < self.binary_image.shape[1] // 2][0]
        self.upper_right_pattern = [p for p in self.finder_pattern_centers if p[0] > self.binary_image.shape[0] // 2 and
                                    p[1] < self.binary_image.shape[1] // 2][0]
        self.lower_left_pattern = [p for p in self.finder_pattern_centers if p[0] < self.binary_image.shape[0] // 2 and
                                   p[1] > self.binary_image.shape[1] // 2][0]

        upper_left_width = [y[2] - y[1] for y in self.x_medians if y[0] == self.upper_left_pattern[1]][0]
        upper_right_width = [y[2] - y[1] for y in self.x_medians if y[0] == self.upper_right_pattern[1]][0]
        module_x_pixels = round((upper_left_width + upper_right_width) / 14)

        inter_module_distance = self.upper_right_pattern[0] - self.upper_left_pattern[0]
        self.qr_code_version = round(((inter_module_distance / module_x_pixels) - 10) / 4)

        if self.qr_code_version > 6:
            self.module_width = round(upper_right_width / 7)
            lower_left_height = [x[2] - x[1] for x in self.y_medians if x[0] == self.lower_left_pattern[0]][0]
            self.module_height = round(lower_left_height / 7)
            self.qr_code_version = self.decode_upper_right_version_information(self.upper_right_pattern)

            if self.qr_code_version == -1:
                self.qr_code_version = self.decode_lower_left_version_information(self.lower_left_pattern)

            if self.qr_code_version == -1:
                raise ValueError('Cannot find version information in QR Code.')

    def decode_upper_right_version_information(self, upper_right_center):
        x_pix = upper_right_center[0] - 5 * self.module_width
        y_pix = upper_right_center[1] + 2 * self.module_height
        version_bytes = []
        for _ in range(18):
            version_bytes.append(self.read_value_from_binary_image_in_pixels(x_pix, y_pix))
            x_pix -= self.module_width
            if len(version_bytes) % 3 == 0:
                x_pix += 3 * self.module_width
                y_pix -= self.module_height

        version_string = ''.join(str(s) for s in version_bytes)
        if version_string in QrCode.version_information:
            qr_code_version = QrCode.version_information.index(version_string) + 7
        else:
            qr_code_version = self.get_version_from_bit_string(version_string)

        return qr_code_version

    def decode_lower_left_version_information(self, lower_left_center):
        x_pix = lower_left_center[0] + 2 * self.module_width
        y_pix = lower_left_center[1] - 5 * self.module_height
        version_bytes = []
        for _ in range(18):
            version_bytes.append(self.read_value_from_binary_image_in_pixels(x_pix, y_pix))
            y_pix -= self.module_height
            if len(version_bytes) % 3 == 0:
                x_pix -= self.module_width
                y_pix += 3 * self.module_height

        version_string = ''.join(str(s) for s in version_bytes)
        if version_string in QrCode.version_information:
            qr_code_version = QrCode.version_information.index(version_string) + 7
        else:
            qr_code_version = self.get_version_from_bit_string(version_string)

        return qr_code_version

    @staticmethod
    def get_version_from_bit_string(string_to_find):
        # Version information id encoded with a maximum Hamming distance of 8.  This allows up to 3 bits to differ
        # when performing a lookup.  The aim is to find the bit string corresponding to the smallest Hamming number.
        min_dist = 100
        min_dist_index = -1
        for v, version in enumerate(QrCode.version_information):
            dist = sum(c1 != c2 for c1, c2 in zip(string_to_find, version))     # get Hamming distance
            if dist < min_dist:
                min_dist = dist
                min_dist_index = v
        if min_dist > 3:
            return -1

        return min_dist_index + 7   # Only versions 7 and above are stored as bit strings

    def read_value_from_binary_image_in_pixels(self, x_pixel, y_pixel):
        return self.binary_image[y_pixel, x_pixel]

    def read_value_from_binary_image_in_modules(self, x_module, y_module):
        x_pixel = (x_module + 0.5) * self.module_width
        y_pixel = (y_module + 0.5) * self.module_height
        return self.read_value_from_binary_image_in_pixels(x_pixel, y_pixel)

    def correct_image_rotation(self, space_value=0, pixel_value=1):
        half_a_degree = 0.5 * pi / 180
        while abs(self.angle) > half_a_degree:
            self.get_finder_pattern_centers(space_value, pixel_value)
            if len(self.finder_pattern_centers) < 3:
                return
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

    def get_finder_pattern_centers(self, space_value, pixel_value):
        self.threshold = (self.raw_image.max() - self.raw_image.min()) // 2  # light / dark threshold value
        # Reduce raw image to a binary array of 1s and 0s, determined by the threshold value.
        # By default, spaces are represented by 0s and pixels by 1s, although these can be reversed by parameters.
        self.binary_image = np.where(np.max(self.raw_image, axis=2) > self.threshold, space_value, pixel_value)
        self.x_medians, self.module_width = self.get_medians(self.binary_image)
        self.y_medians, self.module_height = self.get_medians(np.transpose(self.binary_image))
        self.finder_pattern_centers = self.intersect_medians(self.x_medians, self.y_medians)

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

                is_finder_pattern = 0.5 <= norm[0] <= 1.5 and 0.5 <= norm[1] <= 1.5 and 2.5 <= norm[2] <= 3.5 and \
                    0.5 <= norm[3] <= 1.5 and 0.5 <= norm[4] <= 1.5

                # Valid pattern must start with a dark pixel
                is_pixel = binary_array[row_index][start_position_in_row] == 1

                if is_pixel and is_finder_pattern:
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
    qrr = QrCodeRead('Untitled_v21.png')
    qrr.read_qr_code()
