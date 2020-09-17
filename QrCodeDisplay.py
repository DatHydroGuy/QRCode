# import pygame
import cv2
import numpy as np
from copy import deepcopy
from QrCode import QrCode
from Polynomials import Polynomials


class QrCodeDraw:
    masked_format_info = {'00000': '101010000010010',
                          '00001': '101000100100101',
                          '00010': '101111001111100',
                          '00011': '101101101001011',
                          '00100': '100010111111001',
                          '00101': '100000011001110',
                          '00110': '100111110010111',
                          '00111': '100101010100000',
                          '01000': '111011111000100',
                          '01001': '111001011110011',
                          '01010': '111110110101010',
                          '01011': '111100010011101',
                          '01100': '110011000101111',
                          '01101': '110001100011000',
                          '01110': '110110001000001',
                          '01111': '110100101110110',
                          '10000': '001011010001001',
                          '10001': '001001110111110',
                          '10010': '001110011100111',
                          '10011': '001100111010000',
                          '10100': '000011101100010',
                          '10101': '000001001010101',
                          '10110': '000110100001100',
                          '10111': '000100000111011',
                          '11000': '011010101011111',
                          '11001': '011000001101000',
                          '11010': '011111100110001',
                          '11011': '011101000000110',
                          '11100': '010010010110100',
                          '11101': '010000110000011',
                          '11110': '010111011011010',
                          '11111': '010101111101101'
                          }
    error_correction_indicators = ['01', '00', '11', '10']
    polynomial_manager = Polynomials()

    def __init__(self, qr_code, mask_number=-1, width=1000, height=1000):
        self.qr_code = qr_code
        self.qr_code.mask_pattern = mask_number
        self.width_in_pixels = width
        self.height_in_pixels = height
        self.width_in_modules = 21 + 4 * (qr.minimum_version - 1)
        self.height_in_modules = 21 + 4 * (qr.minimum_version - 1)
        self.matrix = [[-1 for _ in range(self.width_in_modules)] for _ in range(self.height_in_modules)]
        self.matrix_copy = []

    def draw(self):
        # pygame.init()
        # screen = pygame.display.set_mode((self.width_in_pixels, self.height_in_pixels))
        # clock = pygame.time.Clock()

        self.create_qr_code()
        ecc_lookup = {0: 'L', 1: 'M', 2: 'Q', 3: 'H'}

        test = np.ones([self.height_in_modules + 8, self.width_in_modules + 8]) - np.array(self.matrix)
        r = cv2.resize(test, (self.width_in_pixels, self.height_in_pixels), interpolation=cv2.INTER_AREA)
        cv2.imshow(f"QR Code Generator, ver {self.qr_code.minimum_version},"
                   f" ecc {ecc_lookup[self.qr_code.error_correction_level]},"
                   f" mask {self.qr_code.mask_pattern}", r)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # pygame.display.set_caption(f"QR Code Generator, ver {self.qr_code.minimum_version},"
        #                            f" ecc {ecc_lookup[self.qr_code.error_correction_level]},"
        #                            f" mask {self.qr_code.mask_pattern}")

        # running = True
        #
        # while running:
        #     clock.tick(60)
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             running = False
        #
        #         if event.type == pygame.KEYDOWN:  # Change game speed with number keys
        #             if event.key == pygame.K_ESCAPE:
        #                 running = False
        #     screen.fill((255, 255, 255))  # draw background
        #
        #     self.draw_qr_code(screen)
        #     pygame.display.update()
        #
        # pygame.quit()
        # quit()

    def create_qr_code(self):
        self.create_finder_patterns()
        self.create_separators()
        self.create_alignment_patterns()
        self.create_timing_patterns()
        self.create_reserved_areas()
        self.place_data_bits()

        if self.qr_code.mask_pattern == -1:
            best_mask = self.evaluate_masks()
            self.qr_code.mask_pattern = best_mask
        else:
            best_mask = self.qr_code.mask_pattern

        self.mask_bits(best_mask)
        self.create_dark_module()
        self.write_format_string(best_mask)
        self.write_version_information()

        self.create_quiet_zone()

    def create_quiet_zone(self):
        final_width = self.width_in_modules + 8
        for i in range(4):
            self.matrix.insert(0, [0] * final_width)
        for i in range(self.height_in_modules):
            self.matrix[i + 4] = [0] * 4 + self.matrix[i + 4] + [0] * 4
        for i in range(4):
            self.matrix.append([0] * final_width)

    def write_version_information(self):
        if self.qr_code.minimum_version < 7:
            return
        version_info = self.qr_code.version_information[self.qr_code.minimum_version - 7][::-1]
        column = self.matrix[0].index(4)
        row = [a[0] for a in self.matrix].index(4)
        for i in range(6):
            self.matrix[row][i] = int(version_info[i * 3])
            self.matrix[row + 1][i] = int(version_info[i * 3 + 1])
            self.matrix[row + 2][i] = int(version_info[i * 3 + 2])
            self.matrix[i][column] = int(version_info[i * 3])
            self.matrix[i][column + 1] = int(version_info[i * 3 + 1])
            self.matrix[i][column + 2] = int(version_info[i * 3 + 2])

    def write_format_string(self, mask_number):
        format_info = self.error_correction_indicators[self.qr_code.error_correction_level]
        format_info += f'{mask_number:0>3b}'
        format_info = self.masked_format_info[format_info]

        column = self.matrix[0].index(4)
        row = [a[0] for a in self.matrix].index(4)

        idx = 0
        for c in range(len(self.matrix[0])):
            if self.matrix[row][c] == 4:
                if idx == 7 and c <= column:
                    pass
                else:
                    self.matrix[row][c] = int(format_info[idx])
                    idx += 1
        idx = 14
        for r in range(len(self.matrix)):
            if self.matrix[r][column] == 4:
                self.matrix[r][column] = int(format_info[idx])
                idx -= 1

    def mask_bits(self, mask_number):
        if mask_number == 0:
            self.matrix = self.evaluate_mask_0(self.matrix, True)
        elif mask_number == 1:
            self.matrix = self.evaluate_mask_1(self.matrix, True)
        elif mask_number == 2:
            self.matrix = self.evaluate_mask_2(self.matrix, True)
        elif mask_number == 3:
            self.matrix = self.evaluate_mask_3(self.matrix, True)
        elif mask_number == 4:
            self.matrix = self.evaluate_mask_4(self.matrix, True)
        elif mask_number == 5:
            self.matrix = self.evaluate_mask_5(self.matrix, True)
        elif mask_number == 6:
            self.matrix = self.evaluate_mask_6(self.matrix, True)
        else:
            self.matrix = self.evaluate_mask_7(self.matrix, True)

    def evaluate_masks(self):
        scores = []
        matrix_copy = deepcopy(self.matrix)
        scores.append(self.evaluate_mask_0(matrix_copy))
        matrix_copy = deepcopy(self.matrix)
        scores.append(self.evaluate_mask_1(matrix_copy))
        matrix_copy = deepcopy(self.matrix)
        scores.append(self.evaluate_mask_2(matrix_copy))
        matrix_copy = deepcopy(self.matrix)
        scores.append(self.evaluate_mask_3(matrix_copy))
        matrix_copy = deepcopy(self.matrix)
        scores.append(self.evaluate_mask_4(matrix_copy))
        matrix_copy = deepcopy(self.matrix)
        scores.append(self.evaluate_mask_5(matrix_copy))
        matrix_copy = deepcopy(self.matrix)
        scores.append(self.evaluate_mask_6(matrix_copy))
        matrix_copy = deepcopy(self.matrix)
        scores.append(self.evaluate_mask_7(matrix_copy))
        lowest = min(scores)
        best_mask = scores.index(lowest)
        return best_mask

    def evaluate_condition_1(self, matrix):
        penalties = self.count_condition1_row_penalties(matrix)

        transposed = list(map(list, zip(*matrix)))
        penalties += self.count_condition1_row_penalties(transposed)

        return penalties

    @staticmethod
    def count_condition1_row_penalties(matrix):
        penalties = 0
        for row in matrix:
            row_penalties = 0
            curr_val = row[0]
            val_count = 1
            for col in range(1, len(matrix[0])):
                if row[col] == curr_val:
                    val_count += 1
                else:
                    if val_count >= 5:
                        row_penalties += val_count - 2
                    curr_val = row[col]
                    val_count = 1
            if val_count >= 5:
                row_penalties += val_count - 2
            penalties += row_penalties
        return penalties

    @staticmethod
    def evaluate_condition_2(matrix):
        penalties = 0
        for row in range(1, len(matrix)):
            for col in range(1, len(matrix[0])):
                if matrix[row][col] == matrix[row - 1][col - 1] and\
                   matrix[row][col] == matrix[row - 1][col] and\
                   matrix[row][col] == matrix[row][col - 1]:
                    penalties += 3
        return penalties

    def evaluate_condition_3(self, matrix):
        penalties = self.count_condition3_row_penalties(matrix)

        transposed = list(map(list, zip(*matrix)))
        penalties += self.count_condition3_row_penalties(transposed)

        return penalties

    @staticmethod
    def count_condition3_row_penalties(matrix):
        penalties = 0
        for row in range(len(matrix)):
            for col in range(10, len(matrix[0])):
                if matrix[row][col - 10: col + 1] == [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0] or \
                        matrix[row][col - 10: col + 1] == [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1]:
                    penalties += 40
        return penalties

    def evaluate_condition_4(self, matrix):
        num_dark_modules = sum(row.count(1) for row in matrix)
        total_modules = self.width_in_modules * self.height_in_modules
        percent_dark = 100 * num_dark_modules / total_modules
        val1 = (percent_dark // 5) * 5
        val2 = val1 + 5
        val1 = abs(val1 - 50)
        val2 = abs(val2 - 50)
        penalties = 10 * min(val1 / 5, val2 / 5)

        return int(penalties)

    def evaluate_conditions(self, matrix, apply_mask=False):
        matrix_copy = deepcopy(matrix)
        for r, row in enumerate(matrix_copy):
            for c in range(len(row)):
                matrix_copy[r][c] = 1 if matrix_copy[r][c] == 2 else matrix_copy[r][c]
                matrix_copy[r][c] = 0 if matrix_copy[r][c] == 3 else matrix_copy[r][c]
                if apply_mask is False:
                    matrix_copy[r][c] = 0 if matrix_copy[r][c] == 4 else matrix_copy[r][c]

        if apply_mask:
            return matrix_copy
        else:
            c1 = self.evaluate_condition_1(matrix_copy)
            c2 = self.evaluate_condition_2(matrix_copy)
            c3 = self.evaluate_condition_3(matrix_copy)
            c4 = self.evaluate_condition_4(matrix_copy)
            return c1 + c2 + c3 + c4

    def evaluate_mask_0(self, matrix, apply_mask=False):
        for y in range(self.height_in_modules):
            for x in range(self.width_in_modules):
                if matrix[y][x] not in [2, 3, 4]:
                    if (x + y) % 2 == 0:
                        matrix[y][x] = 1 - matrix[y][x]

        return self.evaluate_conditions(matrix, apply_mask)

    def evaluate_mask_1(self, matrix, apply_mask=False):
        for y in range(self.height_in_modules):
            for x in range(self.width_in_modules):
                if matrix[y][x] not in [2, 3, 4]:
                    if y % 2 == 0:
                        matrix[y][x] = 1 - matrix[y][x]

        return self.evaluate_conditions(matrix, apply_mask)

    def evaluate_mask_2(self, matrix, apply_mask=False):
        for y in range(self.height_in_modules):
            for x in range(self.width_in_modules):
                if matrix[y][x] not in [2, 3, 4]:
                    if x % 3 == 0:
                        matrix[y][x] = 1 - matrix[y][x]

        return self.evaluate_conditions(matrix, apply_mask)

    def evaluate_mask_3(self, matrix, apply_mask=False):
        for y in range(self.height_in_modules):
            for x in range(self.width_in_modules):
                if matrix[y][x] not in [2, 3, 4]:
                    if (x + y) % 3 == 0:
                        matrix[y][x] = 1 - matrix[y][x]

        return self.evaluate_conditions(matrix, apply_mask)

    def evaluate_mask_4(self, matrix, apply_mask=False):
        for y in range(self.height_in_modules):
            for x in range(self.width_in_modules):
                if matrix[y][x] not in [2, 3, 4]:
                    if ((x // 3) + (y // 2)) % 2 == 0:
                        matrix[y][x] = 1 - matrix[y][x]

        return self.evaluate_conditions(matrix, apply_mask)

    def evaluate_mask_5(self, matrix, apply_mask=False):
        for y in range(self.height_in_modules):
            for x in range(self.width_in_modules):
                if matrix[y][x] not in [2, 3, 4]:
                    if ((x * y) % 2) + ((x * y) % 3) == 0:
                        matrix[y][x] = 1 - matrix[y][x]

        return self.evaluate_conditions(matrix, apply_mask)

    def evaluate_mask_6(self, matrix, apply_mask=False):
        for y in range(self.height_in_modules):
            for x in range(self.width_in_modules):
                if matrix[y][x] not in [2, 3, 4]:
                    if (((x * y) % 2) + ((x * y) % 3)) % 2 == 0:
                        matrix[y][x] = 1 - matrix[y][x]

        return self.evaluate_conditions(matrix, apply_mask)

    def evaluate_mask_7(self, matrix, apply_mask=False):
        for y in range(self.height_in_modules):
            for x in range(self.width_in_modules):
                if matrix[y][x] not in [2, 3, 4]:
                    if (((x + y) % 2) + ((x * y) % 3)) % 2 == 0:
                        matrix[y][x] = 1 - matrix[y][x]

        return self.evaluate_conditions(matrix, apply_mask)

    def place_data_bits(self):
        x = self.width_in_modules - 1
        y = self.height_in_modules - 1
        idx = 0
        while x >= 0:
            while y >= 0:
                idx, x = self.update_matrix_cell_pair(idx, x, y)
                y -= 1
                idx += 2
            x -= 2
            if x == 6:
                x -= 1
            y = 0
            if x < 0:
                break
            while y < self.height_in_modules:
                idx, x = self.update_matrix_cell_pair(idx, x, y)
                y += 1
                idx += 2
            x -= 2
            if x == 6:
                x -= 1
            y = self.height_in_modules - 1

    def update_matrix_cell_pair(self, idx, x, y):
        if self.matrix[y][x] == -1:
            self.matrix[y][x] = int(self.qr_code.bit_string[idx])
        else:
            idx -= 1
        x -= 1
        if self.matrix[y][x] == -1:
            self.matrix[y][x] = int(self.qr_code.bit_string[idx + 1])
        else:
            idx -= 1
        x += 1
        return idx, x

    def create_reserved_areas(self):
        self.create_dark_module()
        self.create_format_information_areas()
        self.create_version_information_areas()

    def create_version_information_areas(self):
        if self.qr_code.minimum_version < 7:
            return

        min_value = -11
        max_value = -8
        for i in range(min_value, max_value):
            for j in range(6):
                self.matrix[j][i] = 4
                self.matrix[i][j] = 4

    def create_format_information_areas(self):
        self.create_format_information_area(8, 8)
        self.create_format_information_area(8, -8)
        self.create_format_information_area(-8, 8)

    def create_format_information_area(self, x, y):
        min_x = 0 if x > 0 else x
        max_x = x if x > 0 else 0
        min_y = 0 if y > 0 else y
        max_y = y if y > 0 else 0
        if x > 0:
            for yy in range(min_y, max_y + 1):
                self.matrix[yy][x] = 4 if self.matrix[yy][x] == -1 else self.matrix[yy][x]
        else:
            for xx in range(min_x, max_x + 1):
                self.matrix[y][xx] = 4 if self.matrix[y][xx] == -1 else self.matrix[y][xx]
        if y > 0:
            for xx in range(min_x, max_x):
                self.matrix[y][xx] = 4 if self.matrix[y][xx] == -1 else self.matrix[y][xx]
        else:
            for yy in range(min_y, max_y):
                self.matrix[yy][x] = 4 if self.matrix[yy][x] == -1 else self.matrix[yy][x]

    def create_dark_module(self):
        self.matrix[(4 * self.qr_code.minimum_version) + 9][8] = 1

    def create_timing_patterns(self):
        for x in range(6, self.width_in_modules - 8, 2):
            self.matrix[6][x: x + 2] = [2, 3]
        y_val = 2
        for y in range(6, self.height_in_modules - 8):
            self.matrix[y][6] = y_val
            y_val = 5 - y_val

    def create_alignment_patterns(self):
        for y in self.qr_code.alignment_pattern_locations[self.qr_code.minimum_version]:
            for x in self.qr_code.alignment_pattern_locations[self.qr_code.minimum_version]:
                self.create_alignment_pattern(x, y)

    def create_alignment_pattern(self, x, y):
        if any([any([c != -1 for c in b[x - 2: x + 3]]) for b in self.matrix[y - 2: y + 3]]):
            return
        pattern1 = [2, 2, 2, 2, 2]
        pattern2 = [2, 3, 3, 3, 2]
        pattern3 = [2, 3, 2, 3, 2]
        self.matrix[y - 2][x - 2: x + 3] = pattern1
        self.matrix[y - 1][x - 2: x + 3] = pattern2
        self.matrix[y][x - 2: x + 3] = pattern3
        self.matrix[y + 1][x - 2: x + 3] = pattern2
        self.matrix[y + 2][x - 2: x + 3] = pattern1

    def create_separators(self):
        self.create_separator(7, 7)
        self.create_separator(7, -8)
        self.create_separator(-8, 7)

    def create_separator(self, x, y):
        min_x = 0 if x > 0 else x
        max_x = x if x > 0 else 0
        min_y = 0 if y > 0 else y
        max_y = y if y > 0 else 0
        if x > 0:
            for yy in range(min_y, max_y + 1):
                self.matrix[yy][x] = 3
        else:
            for yy in range(min_y, max_y + 1):
                self.matrix[yy][x] = 3
        if y > 0:
            for xx in range(min_x, max_x):
                self.matrix[y][xx] = 3
        else:
            for xx in range(min_x, max_x):
                self.matrix[y][xx] = 3

    def create_finder_patterns(self):
        self.create_finder_pattern(3, 3)
        self.create_finder_pattern(3, -4)
        self.create_finder_pattern(-4, 3)

    def create_finder_pattern(self, x, y):
        pattern1 = [2, 2, 2, 2, 2, 2, 2]
        pattern2 = [2, 3, 3, 3, 3, 3, 2]
        pattern3 = [2, 3, 2, 2, 2, 3, 2]
        if x > 0:
            self.matrix[y - 3][:7] = pattern1
            self.matrix[y - 2][:7] = pattern2
            self.matrix[y - 1][:7] = pattern3
            self.matrix[y][:7] = pattern3
            self.matrix[y + 1][:7] = pattern3
            self.matrix[y + 2][:7] = pattern2
            self.matrix[y + 3][:7] = pattern1
        else:
            self.matrix[y - 3][-7:] = pattern1
            self.matrix[y - 2][-7:] = pattern2
            self.matrix[y - 1][-7:] = pattern3
            self.matrix[y][-7:] = pattern3
            self.matrix[y + 1][-7:] = pattern3
            self.matrix[y + 2][-7:] = pattern2
            self.matrix[y + 3][-7:] = pattern1

    # def draw_qr_code(self, screen):
    #     width_in_modules = self.width_in_modules + 8
    #     height_in_modules = self.height_in_modules + 8
    #     pixels_per_module = min(self.width_in_pixels // width_in_modules,
    #                             self.height_in_pixels // height_in_modules)
    #
    #     for y in range(height_in_modules):
    #         for x in range(width_in_modules):
    #             col = (0, 0, 0) if self.matrix[y][x] == 1 else (255, 255, 255) if self.matrix[y][x] == 0 else\
    #                 (0, 0, 255) if self.matrix[y][x] == 4 else (255, 0, 0) if self.matrix[y][x] == 3 else\
    #                 (0, 255, 0) if self.matrix[y][x] == 2 else (128, 128, 128)
    #             pygame.draw.rect(screen, col, (x * pixels_per_module, y * pixels_per_module,
    #                                            (x + 1) * pixels_per_module, (y + 1) * pixels_per_module))
    #     pygame.draw.rect(screen, (255, 255, 255), (width_in_modules * pixels_per_module, 0,
    #                                                self.width_in_pixels, self.height_in_pixels))
    #     pygame.draw.rect(screen, (255, 255, 255), (0, height_in_modules * pixels_per_module,
    #                                                self.width_in_pixels, self.height_in_pixels))


if __name__ == '__main__':
    qr = QrCode(error_correction_level=3, minimum_version=1)
    qr.generate('Ratio of circle circumference to diameter (C/D) π = 3.141592653589793238462643383279....')
    qr_draw = QrCodeDraw(qr)
    qr_draw.draw()
