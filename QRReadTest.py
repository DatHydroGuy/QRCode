import cv2
import numpy as np


def main():
    t = cv2.imread('Untitled4.png')
    threshold = (t.max() - t.min()) // 2
    # t_max = np.max(t, axis=2)
    t_forced = np.where(np.max(t, axis=2) > threshold, 0, 1)     # spaces = 0, pixels = 1
    groupings = [np.diff(np.flatnonzero(np.concatenate(([True], a[1:] != a[:-1], [True])))) for a in t_forced]
    i = 0
    for row in groupings:
        temp = np.divide(row, min(row))
        for col5 in range(len(temp) - 5):
            test = temp[col5: col5 + 5]
            is_pixel = t_forced[i][np.sum(groupings[i][:col5])] == 1
            if is_pixel and 0.5 <= test[0] <= 1.5 and\
                0.5 <= test[1] <= 1.5 and\
                2.5 <= test[2] <= 3.5 and\
                0.5 <= test[3] <= 1.5 and\
                0.5 <= test[4] <= 1.5:
                    print(i, col5)
        i += 1

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


if __name__ == '__main__':
    main()
