import cv2
import numpy as np


def main():
    t = cv2.imread('Untitled4-2.png')
    threshold = (t.max() - t.min()) // 2
    # t_max = np.max(t, axis=2)
    t_forced = np.where(np.max(t, axis=2) > threshold, 0, 1)     # spaces = 0, pixels = 1

    x_candidates = get_candidate_finder_patterns(t_forced)
    x_sorted = sorted(x_candidates, key=lambda x: (x[1], x[2], x[0]))
    x_mid_lines = cluster_groups(x_sorted)

    t_forced_transposed = np.transpose(t_forced)
    y_candidates = get_candidate_finder_patterns(t_forced_transposed)
    y_sorted = sorted(y_candidates, key=lambda y: (y[1], y[2], y[0]))
    y_mid_lines = cluster_groups(y_sorted)

    mid_points = []
    for x in x_mid_lines:
        mid_point = [x[0]]
        for y in y_mid_lines:
            if y[1] <= x[0] <= y[2] and x[1] <= y[0] <= x[2]:
                mid_point.insert(0, y[0])
        mid_points.append(mid_point)

    first_pixel_data = np.where(t < threshold)
    first_pixel_row = first_pixel_data[0][0]
    first_pixel_column = first_pixel_data[1][0]

    last_pixel_row = np.where(t < threshold)[0][-1]
    qr_height_in_pixels = last_pixel_row - first_pixel_row + 1

    last_pixel_column = np.where(t[first_pixel_row] < threshold)[0][-1]
    qr_width_in_pixels = last_pixel_column - first_pixel_column + 1

    qr_array = t[first_pixel_row: last_pixel_row + 1, first_pixel_column: last_pixel_column + 1]

    first_white_column = np.where(qr_array[0] > threshold)[0][0]
    scaling = first_white_column // 7   # first white pixel appears after dark, light, 3 * dark, light, dark
    r = cv2.resize(qr_array, (qr_width_in_pixels // scaling, qr_height_in_pixels // scaling), interpolation=cv2.INTER_AREA)

    cv2.imshow("orig", t)
    cv2.imshow("trunc", r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cluster_groups(list_to_cluster):
    groups = [[]]
    for coord in list_to_cluster:
        append_list = []
        for group_idx, curr_group in enumerate(groups):
            if len(curr_group) == 0:
                curr_group.append(coord)
                append_list.append(group_idx)
                continue
            for curr_elem in curr_group:
                x_diff = curr_elem[1] - coord[1]
                y_diff = curr_elem[0] - coord[0]
                if x_diff * x_diff + y_diff * y_diff <= 1.9 * 1.9:
                    if len(append_list) == 0:
                        curr_group.append(coord)
                        append_list.append(group_idx)
                    else:
                        # need to append current list to list with index of append_list[0]
                        groups[append_list[0]] += [x for x in curr_group]
                        groups[group_idx] = []
                    break
        if len(append_list) == 0:
            groups.append([coord])

    # return the longest 3 sublists (one for each finder pattern), with each sublist sorted
    # return sorted((sorted(group) for group in groups), key=len, reverse=True)[:3]

    finder_patterns_clusters = sorted(groups, key=len, reverse=True)[:3]
    mid_lines = []
    for cluster in finder_patterns_clusters:
        sorted_cluster = sorted(cluster)
        mid_line = sorted_cluster[len(sorted_cluster) // 2]
        mid_lines.append(mid_line)
    return mid_lines


def get_candidate_finder_patterns(t_forced):
    candidates = []
    groupings = [np.diff(np.flatnonzero(np.concatenate(([True], a[1:] != a[:-1], [True])))) for a in t_forced]
    for row_position, row in enumerate(groupings):
        for col5 in range(len(row) - 5):
            sub_array = row[col5: col5 + 5]
            col_position = np.sum(groupings[row_position][:col5])
            norm = np.divide(sub_array, min(sub_array))
            is_pixel = t_forced[row_position][col_position] == 1  # Valid pattern must start with a dark pixel
            if is_pixel and \
                    0.5 <= norm[0] <= 1.5 and \
                    0.5 <= norm[1] <= 1.5 and \
                    2.5 <= norm[2] <= 3.5 and \
                    0.5 <= norm[3] <= 1.5 and \
                    0.5 <= norm[4] <= 1.5:
                col_end_position = np.sum(groupings[row_position][:col5 + 5])
                candidates.append((row_position, col_position, col_end_position))
    return candidates


if __name__ == '__main__':
    main()