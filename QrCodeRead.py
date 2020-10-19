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
    error_correction_level = -1
    mask_pattern = -1
    qr = None
    bit_stream = None
    byte_stream = None
    p = 0
    ckr = []
    format_decoding_information = {'101010000010010': ['00000', '0000000000'],
                                   '101000100100101': ['00001', '0100110111'],
                                   '101111001111100': ['00010', '1001101110'],
                                   '101101101001011': ['00011', '1101011001'],
                                   '100010111111001': ['00100', '0111101011'],
                                   '100000011001110': ['00101', '0011011100'],
                                   '100111110010111': ['00110', '1110000101'],
                                   '100101010100000': ['00111', '1010110010'],
                                   '111011111000100': ['01000', '1111010110'],
                                   '111001011110011': ['01001', '1011100001'],
                                   '111110110101010': ['01010', '0110111000'],
                                   '111100010011101': ['01011', '0010001111'],
                                   '110011000101111': ['01100', '1000111101'],
                                   '110001100011000': ['01101', '1100001010'],
                                   '110110001000001': ['01110', '0001010011'],
                                   '110100101110110': ['01111', '0101100100'],
                                   '001011010001001': ['10000', '1010011011'],
                                   '001001110111110': ['10001', '1110101100'],
                                   '001110011100111': ['10010', '0011110101'],
                                   '001100111010000': ['10011', '0111000010'],
                                   '000011101100010': ['10100', '1101110000'],
                                   '000001001010101': ['10101', '1001000111'],
                                   '000110100001100': ['10110', '0100011110'],
                                   '000100000111011': ['10111', '0000101001'],
                                   '011010101011111': ['11000', '0101001101'],
                                   '011000001101000': ['11001', '0001111010'],
                                   '011111100110001': ['11010', '1100100011'],
                                   '011101000000110': ['11011', '1000010100'],
                                   '010010010110100': ['11100', '0010100110'],
                                   '010000110000011': ['11101', '0110010001'],
                                   '010111011011010': ['11110', '1011001000'],
                                   '010101111101101': ['11111', '1111111111']}
    ckr_values = [
        [],  # There is no version 0!
        [[[26, 19, 2]], [[26, 16, 4]], [[26, 13, 6]], [[26, 9, 8]]],
        [[[44, 34, 4]], [[44, 28, 8]], [[44, 22, 11]], [[44, 16, 14]]],
        [[[70, 55, 7]], [[70, 44, 13]], [[35, 17, 9]], [[35, 13, 11]]],
        [[[100, 80, 10]], [[50, 32, 9]], [[50, 24, 13]], [[25, 9, 8]]],
        [[134, 108, 13], [[67, 43, 12]], [[33, 15, 9], [34, 16, 9]], [[33, 11, 11], [34, 12, 11]]],
        [[[86, 68, 9]], [[43, 27, 8]], [[43, 19, 12]], [[43, 15, 14]]],
        [[[98, 78, 10]], [[49, 31, 9]], [[32, 14, 9], [33, 15, 9]], [[39, 13, 13], [40, 14, 13]]],
        [[[121, 97, 12]], [[60, 38, 11], [61, 39, 11]], [[40, 18, 11], [41, 19, 11]], [[40, 14, 13], [41, 15, 13]]],
        [[[146, 116, 15]], [[58, 36, 11], [59, 37, 11]], [[36, 16, 10], [37, 17, 10]], [[36, 12, 12], [37, 13, 12]]],
        [[[86, 68, 9], [87, 69, 9]], [[69, 43, 13], [70, 44, 13]], [[43, 19, 12], [44, 20, 12]],
         [[43, 15, 14], [44, 16, 14]]],
        [[[101, 81, 10]], [[80, 50, 15], [81, 51, 15]], [[50, 22, 14], [51, 23, 14]], [[36, 12, 12], [37, 13, 12]]],
        [[[116, 92, 12], [117, 93, 12]], [[58, 36, 11], [59, 37, 11]], [[46, 20, 13], [47, 21, 13]],
         [[42, 14, 14], [43, 15, 14]]],
        [[[133, 107, 13]], [[59, 37, 11], [60, 38, 11]], [[44, 20, 12], [45, 21, 12]], [[33, 11, 11], [34, 12, 11]]],
        [[[145, 115, 15], [146, 116, 15]], [[64, 40, 12], [65, 41, 12]], [[36, 16, 10], [37, 17, 10]],
         [[36, 12, 12], [37, 13, 12]]],
        [[[109, 87, 11], [110, 88, 11]], [[65, 41, 12], [66, 42, 12]], [[54, 24, 15], [55, 25, 15]],
         [[36, 12, 12], [37, 13, 12]]],
        [[[122, 98, 12], [123, 99, 12]], [[73, 45, 14], [74, 46, 14]], [[43, 19, 12], [44, 20, 12]],
         [[45, 15, 15], [46, 16, 15]]],
        [[[135, 107, 14], [136, 108, 14]], [[74, 46, 14], [75, 47, 14]], [[50, 22, 14], [51, 23, 14]],
         [[42, 14, 14], [43, 15, 14]]],
        [[[150, 120, 15], [151, 121, 15]], [[69, 43, 13], [70, 44, 13]], [[50, 22, 14], [51, 23, 14]],
         [[42, 14, 14], [43, 15, 14]]],
        [[[141, 113, 14], [142, 114, 14]], [[70, 44, 13], [71, 45, 13]], [[47, 21, 13], [48, 22, 13]],
         [[39, 13, 13], [40, 14, 13]]],
        [[[135, 107, 14], [136, 108, 14]], [[67, 41, 13, ], [68, 42, 13]], [[54, 24, 15], [55, 25, 15]],
         [[43, 15, 14], [44, 16, 14]]],
        [[[144, 116, 14], [145, 117, 14]], [[68, 42, 13]], [[50, 22, 14], [51, 23, 14]], [[46, 16, 15], [47, 17, 15]]],
        [[[139, 111, 14], [140, 112, 14]], [[74, 46, 14]], [[54, 24, 15], [55, 25, 15]], [[37, 13, 12]]],
        [[[151, 121, 15], [152, 122, 15]], [[75, 47, 14], [76, 48, 14]], [[54, 24, 15], [55, 25, 15]],
         [[45, 15, 15], [46, 16, 15]]],
        [[[147, 117, 15], [148, 118, 15]], [[73, 45, 14], [74, 46, 14]], [[54, 24, 15], [55, 25, 15]],
         [[46, 16, 15], [47, 17, 15]]],
        [[[132, 106, 13], [133, 107, 13]], [[75, 47, 14], [76, 48, 14]], [[54, 24, 15], [55, 25, 15]],
         [[45, 15, 15], [46, 16, 15]]],
        [[[142, 114, 14], [143, 115, 14]], [[74, 46, 14], [75, 47, 14]], [[50, 22, 14], [51, 23, 14]],
         [[46, 16, 15], [47, 17, 15]]],
        [[[152, 122, 15], [153, 123, 15]], [[73, 45, 14], [74, 46, 14]], [[53, 23, 15], [54, 24, 15]],
         [[45, 15, 15], [46, 16, 15]]],
        [[[147, 117, 15], [148, 118, 15]], [[73, 45, 14], [74, 46, 14]], [[54, 24, 15], [55, 25, 15]],
         [[45, 15, 15], [46, 16, 15]]],
        [[[146, 116, 15], [147, 117, 15]], [[73, 45, 14], [74, 46, 14]], [[53, 23, 15], [54, 24, 15]],
         [[45, 15, 15], [46, 16, 15]]],
        [[[145, 115, 15], [146, 116, 15]], [[75, 47, 14], [76, 48, 14]], [[54, 24, 15], [55, 25, 15]],
         [[45, 15, 15], [46, 16, 15]]],
        [[[145, 115, 15], [146, 116, 15]], [[74, 46, 14], [75, 47, 14]], [[54, 24, 15], [55, 25, 15]],
         [[45, 15, 15], [46, 16, 15]]],
        [[[145, 115, 15]], [[74, 46, 14], [75, 47, 14]], [[54, 24, 15], [55, 25, 15]], [[45, 15, 15], [46, 16, 15]]],
        [[[145, 115, 15], [146, 116, 15]], [[74, 46, 14], [75, 47, 14]], [[54, 24, 15], [55, 25, 15]],
         [[45, 15, 15], [46, 16, 15]]],
        [[[145, 115, 15], [146, 116, 15]], [[74, 46, 14], [75, 47, 14]], [[54, 24, 15], [55, 25, 15]],
         [[46, 16, 15], [47, 17, 15]]],
        [[[151, 121, 15], [152, 122, 15]], [[75, 47, 14], [76, 48, 14]], [[54, 24, 15], [55, 25, 15]],
         [[45, 15, 15], [46, 16, 15]]],
        [[[151, 121, 15], [152, 122, 15]], [[75, 47, 14], [76, 48, 14]], [[54, 24, 15], [55, 25, 15]],
         [[45, 15, 15], [46, 16, 15]]],
        [[[152, 122, 15], [153, 123, 15]], [[74, 46, 14], [75, 47, 14]], [[54, 24, 15], [55, 25, 15]],
         [[45, 15, 15], [46, 16, 15]]],
        [[[152, 122, 15], [153, 123, 15]], [[74, 46, 14], [75, 47, 14]], [[54, 24, 15], [55, 25, 15]],
         [[45, 15, 15], [46, 16, 15]]],
        [[[147, 117, 15], [148, 118, 15]], [[75, 47, 14], [76, 48, 14]], [[54, 24, 15], [55, 25, 15]],
         [[45, 15, 15], [46, 16, 15]]],
        [[[148, 118, 15], [149, 119, 15]], [[75, 47, 14], [76, 48, 14]], [[54, 24, 15], [55, 25, 15]],
         [[45, 15, 15], [46, 16, 15]]]
    ]

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
            alignment_positions = QrCode.alignment_pattern_locations[self.qr_code_version]
            self.get_alignment_pattern_positions(alignment_positions, space_value, pixel_value)
            self.populate_sampling_grid(alignment_positions)

        self.sample_grid(space_value, pixel_value)
        self.error_correction_level, self.mask_pattern = self.decode_format_information(space_value)
        self.qr = QrCode(self.error_correction_level, self.qr_code_version, self.mask_pattern)
        self.calculate_value_of_p()
        self.mask_out_functional_areas()
        self.sampling_grid = self.qr.mask_bits(self.sampling_grid, self.mask_pattern, False)
        self.read_grid()
        a = 1

    def read_grid(self):
        self.get_data_bits()
        self.translate_bits_to_byte_values()
        self.reverse_interleave()

    def reverse_interleave(self):
        ecc_info = QrCode.ecc_word_and_block_info[self.error_correction_level][self.qr_code_version]
        ecc_block_length = ecc_info[1]
        block1_rows = ecc_info[2]
        block1_length = ecc_info[3]
        block2_rows = ecc_info[4]
        block2_length = ecc_info[5]
        idx = 0
        block1, block2, idx = self.process_interleaved_blocks(block1_length, block1_rows,
                                                              block2_length, block2_rows, idx)
        for i in range(block1_length, block2_length):
            for j in range(block2_rows):
                block2[j][i] = self.byte_stream[idx]
                idx += 1

        ecc_block1, ecc_block2, idx = self.process_interleaved_blocks(ecc_block_length, block1_rows,
                                                                      ecc_block_length, block2_rows, idx)

        # TODO: Something with rebuilding bytes using the error correction codes goes here
        ckr_temp = self.ckr_values[self.qr_code_version][self.error_correction_level]
        temp = [block1_rows, block2_rows]
        curr_blocks = [block1, block2]
        for i, vals in enumerate(ckr_temp):
            for _ in range(temp[i]):
                self.ckr.append(vals)

            curr_block = curr_blocks[i]

            n = vals[0] - vals[1] - vals[2]
            syndromes = []
            for j in range(n):
                syndromes[j] = 0
                for k in range(vals[0]):
                    syndromes[j] += 1

        self.byte_stream = []
        for row in block1:
            self.byte_stream.extend(row)
        for row in block2:
            self.byte_stream.extend(row)
        for row in ecc_block1:
            self.byte_stream.extend(row)
        for row in ecc_block2:
            self.byte_stream.extend(row)

    def process_interleaved_blocks(self, block1_length, block1_rows, block2_length, block2_rows, idx):
        block1 = [[0 for _ in range(block1_length)] for _ in range(block1_rows)]
        block2 = [[0 for _ in range(block2_length)] for _ in range(block2_rows)]
        for i in range(block1_length):
            for j in range(block1_rows):
                block1[j][i] = self.byte_stream[idx]
                idx += 1
            for j in range(block2_rows):
                block2[j][i] = self.byte_stream[idx]
                idx += 1
        return block1, block2, idx

    def translate_bits_to_byte_values(self):
        self.byte_stream = []
        for idx in range(0, len(self.bit_stream), 8):
            temp = int(''.join([str(x) for x in self.bit_stream[idx: idx + 8]]), 2)
            self.byte_stream.append(temp)

    def calculate_value_of_p(self):
        if self.qr_code_version < 4:
            if self.qr_code_version == 3 and self.error_correction_level == 0:
                self.p = 1
            elif self.qr_code_version == 2 and self.error_correction_level == 0:
                self.p = 2
            else:
                if self.error_correction_level == 0:
                    self.p = 3
                elif self.error_correction_level == 1:
                    self.p = 2
                else:
                    self.p = 1

    def mask_out_functional_areas(self):
        self.mask_out_finder_patterns()
        self.mask_out_alignment_patterns()
        self.mask_out_timing_patterns()
        self.mask_out_reserved_areas()

    def mask_out_finder_patterns(self):
        template = np.array([[2, 2, 2, 2, 2, 2, 2, 3],
                             [2, 3, 3, 3, 3, 3, 2, 3],
                             [2, 3, 2, 2, 2, 3, 2, 3],
                             [2, 3, 2, 2, 2, 3, 2, 3],
                             [2, 3, 2, 2, 2, 3, 2, 3],
                             [2, 3, 3, 3, 3, 3, 2, 3],
                             [2, 2, 2, 2, 2, 2, 2, 3],
                             [3, 3, 3, 3, 3, 3, 3, 3]])
        self.sampling_grid[:8, :8] = template
        self.sampling_grid[:8, -8:] = np.flip(template, axis=1)
        self.sampling_grid[-8:, :8] = np.flip(template, axis=0)

    def mask_out_alignment_patterns(self):
        template = np.array([[2, 2, 2, 2, 2],
                             [2, 3, 3, 3, 2],
                             [2, 3, 2, 3, 2],
                             [2, 3, 3, 3, 2],
                             [2, 2, 2, 2, 2]])
        for y in self.qr.alignment_pattern_locations[self.qr.minimum_version]:
            for x in self.qr.alignment_pattern_locations[self.qr.minimum_version]:
                if any([any([c == 2 for c in b[x - 2: x + 3]]) for b in self.sampling_grid[y - 2: y + 3]]):
                    continue
                self.sampling_grid[y - 2: y + 3, x - 2: x + 3] = template

    def mask_out_timing_patterns(self):
        timing_value = 2
        for i in range(6, self.height_in_modules - 8):
            self.sampling_grid[6, i] = timing_value
            self.sampling_grid[i, 6] = timing_value
            timing_value = 5 - timing_value

    def mask_out_reserved_areas(self):
        self.mask_out_dark_module()
        self.mask_out_format_information_areas()
        self.mask_out_version_information_areas()

    def mask_out_version_information_areas(self):
        if self.qr.minimum_version < 7:
            return

        self.sampling_grid[:6, -11: -8] = 4
        self.sampling_grid[-11: -8, :6] = 4

    def mask_out_format_information_areas(self):
        self.sampling_grid[8, :9] = 4
        self.sampling_grid[:9, 8] = 4
        self.sampling_grid[8, -8:] = 4
        self.sampling_grid[-8:, 8] = 4

    def mask_out_dark_module(self):
        self.sampling_grid[(4 * self.qr.minimum_version) + 9, 8] = 4

    def get_data_bits(self):
        self.bit_stream = []
        x = self.qr.width_in_modules - 1
        y = self.qr.height_in_modules - 1
        while x >= 0:
            while y >= 0:
                x = self.update_matrix_cell_pair(x, y)
                y -= 1
            x -= 2
            if x == 6:
                x -= 1
            y = 0
            if x < 0:
                break
            while y < self.height_in_modules:
                x = self.update_matrix_cell_pair(x, y)
                y += 1
            x -= 2
            if x == 6:
                x -= 1
            y = self.height_in_modules - 1

    def update_matrix_cell_pair(self, x, y):
        if self.sampling_grid[y][x] not in [2, 3, 4]:
            self.bit_stream.append(self.sampling_grid[y][x])
        x -= 1
        if self.sampling_grid[y][x] not in [2, 3, 4]:
            self.bit_stream.append(self.sampling_grid[y][x])
        x += 1
        return x

    def decode_format_information(self, space_value=0):
        info_part1 = self.sampling_grid[:6, 8]
        info_part2 = self.sampling_grid[7:9, 8]
        info_part3 = self.sampling_grid[8, 7]
        info_part4 = self.sampling_grid[8, :6][::-1]
        format_info = np.concatenate((info_part1, info_part2, np.array([info_part3]), info_part4))
        format_info_string = ''.join([str(x - space_value) for x in format_info])[::-1]
        old_string = format_info_string
        data_bits, error_correction_bits = self.get_format_information_from_bit_string(format_info_string)
        if len(data_bits) == 0:
            info_part1 = self.sampling_grid[8, -8:][::-1]
            info_part2 = self.sampling_grid[-7:, 8]
            format_info = np.concatenate((info_part1, info_part2))
            format_info_string = ''.join([str(x - space_value) for x in format_info])[::-1]
            data_bits, error_correction_bits = self.get_format_information_from_bit_string(format_info_string)
            if len(data_bits) == 0:
                # TODO: Try reading mirror images of the QR Code and decoding format info from those
                raise ValueError(f'Cannot decode format information from bit strings {old_string} or '
                                 f'{format_info_string}')
        error_correction_level = QrCode.error_correction_indicators.index(data_bits[:2])
        mask_pattern = int(data_bits[2:], 2)
        return error_correction_level, mask_pattern

    def get_format_information_from_bit_string(self, string_to_find):
        # Format information id encoded with a maximum Hamming distance of 7.  This allows up to 3 bits to differ
        # when performing a lookup.  The aim is to find the bit string corresponding to the smallest Hamming number.
        min_dist = 100
        min_dist_string = -1
        for bit_string in self.format_decoding_information.keys():
            dist = sum(c1 != c2 for c1, c2 in zip(string_to_find, bit_string))     # get Hamming distance
            if dist < min_dist:
                min_dist = dist
                min_dist_string = bit_string
        if min_dist > 3:
            return '', ''

        return self.format_decoding_information[min_dist_string]

    def sample_grid(self, space_value=0, pixel_value=1):
        for y in range(self.height_in_modules):
            for x in range(self.width_in_modules):
                pixel = self.sampling_grid[y, x]
                sample = self.binary_image[pixel[1] - 1: pixel[1] + 2, pixel[0] - 1: pixel[0] + 2]
                self.sampling_grid[y, x] = pixel_value if len(sample[sample == 1]) > 4 else space_value
        self.sampling_grid = self.sampling_grid[:, :, 0]

    def populate_sampling_grid(self, alignment_positions):
        resolutions = self.get_resolutions_for_alignment_patterns(alignment_positions)
        self.use_resolutions_to_populate_sampling_grid(alignment_positions, resolutions)

    def use_resolutions_to_populate_sampling_grid(self, alignment_positions, resolutions):
        alignment_boundary = (alignment_positions[1] - alignment_positions[0]) // 2
        for yi, y in enumerate(alignment_positions):
            min_y = -7 if yi == 0 else -alignment_boundary
            max_y = 6 if y == alignment_positions[-1] else alignment_boundary
            for yy in range(min_y + 1, max_y + 1):
                curr_y = alignment_positions[yi] + yy
                for xi, x in enumerate(alignment_positions):
                    min_x = -7 if xi == 0 else -alignment_boundary
                    max_x = 6 if x == alignment_positions[-1] else alignment_boundary
                    curr_alignment_pos = self.sampling_grid[alignment_positions[yi], alignment_positions[xi]]
                    for xx in range(min_x + 1, max_x + 1):
                        curr_x = alignment_positions[xi] + xx
                        self.sampling_grid[curr_y, curr_x] = np.add(curr_alignment_pos,
                                                                    [int(xx * resolutions[yi][xi][0]),
                                                                     int(yy * resolutions[yi][xi][1])])

    def get_resolutions_for_alignment_patterns(self, alignment_positions):
        resolutions = []
        for y in range(1, len(alignment_positions)):
            row = []
            for x in range(1, len(alignment_positions)):
                res_x = (self.sampling_grid[alignment_positions[y], alignment_positions[x], 0] -
                         self.sampling_grid[alignment_positions[y], alignment_positions[x - 1], 0]) / \
                        (alignment_positions[x] - alignment_positions[x - 1])
                res_y = (self.sampling_grid[alignment_positions[y], alignment_positions[x], 1] -
                         self.sampling_grid[alignment_positions[y - 1], alignment_positions[x], 1]) / \
                        (alignment_positions[y] - alignment_positions[y - 1])
                row.append([res_x, res_y])
            row.insert(0, row[0])  # extend x-resolutions for the first column of alignment patterns
            # row.append(row[-1])     # extend x-resolutions for the last column of alignment patterns
            resolutions.append(row)
        resolutions.insert(0, resolutions[0])  # extend y-resolutions for the first row of alignment patterns
        # resolutions.append(resolutions[-1])     # extend y-resolutions for the last row of alignment patterns
        return resolutions

    def get_alignment_pattern_positions(self, alignment_positions, space_value=0, pixel_value=1):
        self.width_in_modules = alignment_positions[-1] + 7
        self.height_in_modules = alignment_positions[-1] + 7
        self.initialise_sampling_grid()
        pattern_limit = int(round((len(alignment_positions) - 1) / 2))
        self.resolve_upper_left_alignment_patterns(alignment_positions, pattern_limit, space_value, pixel_value)
        self.resolve_upper_right_alignment_patterns(alignment_positions, pattern_limit, space_value, pixel_value)
        self.resolve_lower_left_alignment_patterns(alignment_positions, pattern_limit, space_value, pixel_value)
        self.resolve_lower_right_alignment_patterns(alignment_positions, pattern_limit)

    def resolve_upper_left_alignment_patterns(self, alignment_patterns, pattern_limit, space_value=0, pixel_value=1):
        top_row_positions = [[self.upper_left_pattern[0] + 3 * self.module_width,
                              self.upper_left_pattern[1] + 3 * self.module_height]]
        self.sampling_grid[6, 6] = top_row_positions[0]
        module_diff_in_pixels = (alignment_patterns[1] - alignment_patterns[0]) * self.module_width
        for x in range(1, pattern_limit + 1):   # ignore first element along top row, since it will be a finder pattern
            candidate = [module_diff_in_pixels + top_row_positions[x - 1][0], top_row_positions[x - 1][1]]
            candidate = self.tweak_alignment_pattern_position(candidate, space_value, pixel_value)
            top_row_positions.append(candidate)
            self.sampling_grid[alignment_patterns[0], alignment_patterns[x]] = candidate

        left_row_positions = [[self.upper_left_pattern[0] + 3 * self.module_width,
                              self.upper_left_pattern[1] + 3 * self.module_height]]
        module_diff_in_pixels = (alignment_patterns[1] - alignment_patterns[0]) * self.module_height
        for y in range(1, pattern_limit + 1):   # ignore first element along top row, since it will be a finder pattern
            candidate = [left_row_positions[y - 1][0], module_diff_in_pixels + left_row_positions[y - 1][1]]
            candidate = self.tweak_alignment_pattern_position(candidate, space_value, pixel_value)
            left_row_positions.append(candidate)
            self.sampling_grid[alignment_patterns[y], alignment_patterns[0]] = candidate
            for x in range(1, pattern_limit + 1):
                candidate = [top_row_positions[x][0], left_row_positions[y][1]]
                candidate = self.tweak_alignment_pattern_position(candidate, space_value, pixel_value)
                self.sampling_grid[alignment_patterns[y], alignment_patterns[x]] = candidate

    def resolve_upper_right_alignment_patterns(self, alignment_patterns, pattern_limit, space_value=0, pixel_value=1):
        top_row_positions = [[self.upper_right_pattern[0] - 3 * self.module_width,
                              self.upper_right_pattern[1] + 3 * self.module_height]]
        self.sampling_grid[6, -7] = top_row_positions[0]
        module_diff_in_pixels = (alignment_patterns[1] - alignment_patterns[0]) * self.module_width
        for x in range(len(alignment_patterns) - 2, pattern_limit, -1):   # ignore last element for same reason
            candidate = [top_row_positions[0][0] - module_diff_in_pixels, top_row_positions[0][1]]
            candidate = self.tweak_alignment_pattern_position(candidate, space_value, pixel_value)
            top_row_positions.insert(0, candidate)
            self.sampling_grid[alignment_patterns[0], alignment_patterns[x]] = candidate

        right_row_positions = [[self.upper_right_pattern[0] - 3 * self.module_width,
                                self.upper_right_pattern[1] + 3 * self.module_height]]
        module_diff_in_pixels = (alignment_patterns[1] - alignment_patterns[0]) * self.module_height
        for y in range(1, pattern_limit + 1):   # ignore first element along top row, since it will be a finder pattern
            candidate = [right_row_positions[y - 1][0], module_diff_in_pixels + right_row_positions[y - 1][1]]
            candidate = self.tweak_alignment_pattern_position(candidate, space_value, pixel_value)
            right_row_positions.append(candidate)
            self.sampling_grid[alignment_patterns[y], alignment_patterns[-1]] = candidate
            for x in range(1, pattern_limit):
                candidate = [top_row_positions[-1 - x][0], right_row_positions[y][1]]
                candidate = self.tweak_alignment_pattern_position(candidate, space_value, pixel_value)
                self.sampling_grid[alignment_patterns[y], alignment_patterns[-1 - x]] = candidate

    def resolve_lower_left_alignment_patterns(self, alignment_patterns, pattern_limit, space_value=0, pixel_value=1):
        bottom_row_positions = [[self.lower_left_pattern[0] + 3 * self.module_width,
                                 self.lower_left_pattern[1] - 3 * self.module_height]]
        self.sampling_grid[-7, 6] = bottom_row_positions[0]
        module_diff_in_pixels = (alignment_patterns[1] - alignment_patterns[0]) * self.module_width
        for x in range(1, pattern_limit + 1):   # ignore first element along top row, since it will be a finder pattern
            candidate = [module_diff_in_pixels + bottom_row_positions[x - 1][0], bottom_row_positions[x - 1][1]]
            candidate = self.tweak_alignment_pattern_position(candidate, space_value, pixel_value)
            bottom_row_positions.append(candidate)
            self.sampling_grid[alignment_patterns[-1], alignment_patterns[x]] = candidate

        left_row_positions = [[self.lower_left_pattern[0] + 3 * self.module_width,
                               self.lower_left_pattern[1] - 3 * self.module_height]]
        module_diff_in_pixels = (alignment_patterns[1] - alignment_patterns[0]) * self.module_height
        for y in range(1, pattern_limit + 1):   # ignore first element along top row, since it will be a finder pattern
            candidate = [left_row_positions[y - 1][0], left_row_positions[y - 1][1] - module_diff_in_pixels]
            candidate = self.tweak_alignment_pattern_position(candidate, space_value, pixel_value)
            left_row_positions.append(candidate)
            self.sampling_grid[alignment_patterns[-1 - y], alignment_patterns[0]] = candidate
            for x in range(1, pattern_limit + 1):
                candidate = [bottom_row_positions[x][0], left_row_positions[y][1]]
                candidate = self.tweak_alignment_pattern_position(candidate, space_value, pixel_value)
                self.sampling_grid[alignment_patterns[-1 - y], alignment_patterns[x]] = candidate

    def resolve_lower_right_alignment_patterns(self, alignment_patterns, pattern_limit):
        for y in range(pattern_limit + 1, len(alignment_patterns)):
            for x in range(pattern_limit + 1, len(alignment_patterns)):
                self.sampling_grid[alignment_patterns[y], alignment_patterns[x]] = [
                    self.sampling_grid[alignment_patterns[y - 1], alignment_patterns[x], 0],
                    self.sampling_grid[alignment_patterns[y], alignment_patterns[x - 1], 1]
                    ]

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
    #     template3 = [[pixel_value for _ in range(self.module_width)] +
    #                  [space_value for _ in range(self.module_width)] +
    #                  [pixel_value for _ in range(self.module_width)] +
    #                  [space_value for _ in range(self.module_width)] +
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
        for y in range(self.height_in_modules):
            curr_y = self.upper_left_pattern[1] + (y - 3) * self.module_height
            for x in range(self.width_in_modules):
                self.sampling_grid[y, x] = [self.upper_left_pattern[0] + (x - 3) * self.module_width, curr_y]

    def initialise_sampling_grid(self):
        self.sampling_grid = np.array([[[None, None] for _ in range(self.width_in_modules)]
                                       for _ in range(self.height_in_modules)])
        self.sampling_grid[3, 3] = self.upper_left_pattern
        self.sampling_grid[3, -4] = self.upper_right_pattern
        self.sampling_grid[-4, 3] = self.lower_left_pattern

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
        # Version information id encoded with a maximum Hamming distance of 7.  This allows up to 3 bits to differ
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
    qrr = QrCodeRead('qr_a_to_q_low_v1.png')  # Pi_ecc_3_min_v1.png')
    qrr.read_qr_code()
