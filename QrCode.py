from Polynomials import Polynomials


class QrCode:
    encoding = {'numeric': '0001', 'alphanumeric': '0010', 'byte': '0100', 'kanji': '1000', 'ECI': '0111'}
    encoder = ''
    error_correction_level = 0
    minimum_version = 1
    mask_pattern = -1
    ecc_info = [0, 0, 0, 0, 0, 0, 0]
    total_bits_required = 0
    alphanumeric_lookup = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                           'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                           ' ', '$', '%', '*', '+', '-', '.', '/', ':']
    ecc_word_and_block_info = [
        [
            [0, 0, 0, 0, 0, 0, 0], [19, 7, 1, 19, 0, 0, 19], [34, 10, 1, 34, 0, 0, 34], [55, 15, 1, 55, 0, 0, 55],
            [80, 20, 1, 80, 0, 0, 80], [108, 26, 1, 108, 0, 0, 108], [136, 18, 2, 68, 0, 0, 136],
            [156, 20, 2, 78, 0, 0, 156], [194, 24, 2, 97, 0, 0, 194], [232, 30, 2, 116, 0, 0, 232],
            [274, 18, 2, 68, 2, 69, 274], [324, 20, 4, 81, 0, 0, 324], [370, 24, 2, 92, 2, 93, 370],
            [428, 26, 4, 107, 0, 0, 428], [461, 30, 3, 115, 1, 116, 461], [523, 22, 5, 87, 1, 88, 523],
            [589, 24, 5, 98, 1, 99, 589], [647, 28, 1, 107, 5, 108, 647], [721, 30, 5, 120, 1, 121, 721],
            [795, 28, 3, 113, 4, 114, 795], [861, 28, 3, 107, 5, 108, 861], [932, 28, 4, 116, 4, 117, 932],
            [1006, 28, 2, 111, 7, 112, 1006], [1094, 30, 4, 121, 5, 122, 1094], [1174, 30, 6, 117, 4, 118, 1174],
            [1276, 26, 8, 106, 4, 107, 1276], [1370, 28, 10, 114, 2, 115, 1370], [1468, 30, 8, 122, 4, 123, 1468],
            [1531, 30, 3, 117, 10, 118, 1531], [1631, 30, 7, 116, 7, 117, 1631], [1735, 30, 5, 115, 10, 116, 1735],
            [1843, 30, 13, 115, 3, 116, 1843], [1955, 30, 17, 115, 0, 0, 1955], [2071, 30, 17, 115, 1, 116, 2071],
            [2191, 30, 13, 115, 6, 116, 2191], [2306, 30, 12, 121, 7, 122, 2306], [2434, 30, 6, 121, 14, 122, 2434],
            [2566, 30, 17, 122, 4, 123, 2566], [2702, 30, 4, 122, 18, 123, 2702], [2812, 30, 20, 117, 4, 118, 2812],
            [2956, 30, 19, 118, 6, 119, 2956]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0], [16, 10, 1, 16, 0, 0, 16], [28, 16, 1, 28, 0, 0, 28], [44, 26, 1, 44, 0, 0, 44],
            [64, 18, 2, 32, 0, 0, 64], [86, 24, 2, 43, 0, 0, 86], [108, 16, 4, 27, 0, 0, 108],
            [124, 18, 4, 31, 0, 0, 124], [154, 22, 2, 38, 2, 39, 154], [182, 22, 3, 36, 2, 37, 182],
            [216, 26, 4, 43, 1, 44, 216], [254, 30, 1, 50, 4, 51, 254], [290, 22, 6, 36, 2, 37, 290],
            [334, 22, 8, 37, 1, 38, 334], [365, 24, 4, 40, 5, 41, 365], [415, 24, 5, 41, 5, 42, 415],
            [453, 28, 7, 45, 3, 46, 453], [507, 28, 10, 46, 1, 47, 507], [563, 26, 9, 43, 4, 44, 563],
            [627, 26, 3, 44, 11, 45, 627], [669, 26, 3, 41, 13, 42, 669], [714, 26, 17, 42, 0, 0, 714],
            [782, 28, 17, 46, 0, 0, 782], [860, 28, 4, 47, 14, 48, 860], [914, 28, 6, 45, 14, 46, 914],
            [1000, 28, 8, 47, 13, 48, 1000], [1062, 28, 19, 46, 4, 47, 1062], [1128, 28, 22, 45, 3, 46, 1128],
            [1193, 28, 3, 45, 23, 46, 1193], [1267, 28, 21, 45, 7, 46, 1267], [1373, 28, 19, 47, 10, 48, 1373],
            [1455, 28, 2, 46, 29, 47, 1455], [1541, 28, 10, 46, 23, 47, 1541], [1631, 28, 14, 46, 21, 47, 1631],
            [1725, 28, 14, 46, 23, 47, 1725], [1812, 28, 12, 47, 26, 48, 1812], [1914, 28, 6, 47, 34, 48, 1914],
            [1992, 28, 29, 46, 14, 47, 1992], [2102, 28, 13, 46, 32, 47, 2102], [2216, 28, 40, 47, 7, 48, 2216],
            [2334, 28, 18, 47, 31, 48, 2334]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0], [13, 13, 1, 13, 0, 0, 13], [22, 22, 1, 22, 0, 0, 22], [34, 18, 2, 17, 0, 0, 34],
            [48, 26, 2, 24, 0, 0, 48], [62, 18, 2, 15, 2, 16, 62], [76, 24, 4, 19, 0, 0, 76],
            [88, 18, 2, 14, 4, 15, 88], [110, 22, 4, 18, 2, 19, 110], [132, 20, 4, 16, 4, 17, 132],
            [154, 24, 6, 19, 2, 20, 154], [180, 28, 4, 22, 4, 23, 180], [206, 26, 4, 20, 6, 21, 206],
            [244, 24, 8, 20, 4, 21, 244], [261, 20, 11, 16, 5, 17, 261], [295, 30, 5, 24, 7, 25, 295],
            [325, 24, 15, 19, 2, 20, 325], [367, 28, 1, 22, 15, 23, 367], [397, 28, 17, 22, 1, 23, 397],
            [445, 26, 17, 21, 4, 22, 445], [485, 30, 15, 24, 5, 25, 485], [512, 28, 17, 22, 6, 23, 512],
            [568, 30, 7, 24, 16, 25, 568], [614, 30, 11, 24, 14, 25, 614], [664, 30, 11, 24, 16, 25, 664],
            [718, 30, 7, 24, 22, 25, 718], [754, 28, 28, 22, 6, 23, 754], [808, 30, 8, 23, 26, 24, 808],
            [871, 30, 4, 24, 31, 25, 871], [911, 30, 1, 23, 37, 24, 911], [985, 30, 15, 24, 25, 25, 985],
            [1033, 30, 42, 24, 1, 25, 1033], [1115, 30, 10, 24, 35, 25, 1115], [1171, 30, 29, 24, 19, 25, 1171],
            [1231, 30, 44, 24, 7, 25, 1231], [1286, 30, 39, 24, 14, 25, 1286], [1354, 30, 46, 24, 10, 25, 1354],
            [1426, 30, 49, 24, 10, 25, 1426], [1502, 30, 48, 24, 14, 25, 1502], [1582, 30, 43, 24, 22, 25, 1582],
            [1666, 30, 34, 24, 34, 25, 1666]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0], [9, 17, 1, 9, 0, 0, 9], [16, 28, 1, 16, 0, 0, 16], [26, 22, 2, 13, 0, 0, 26],
            [36, 16, 4, 9, 0, 0, 36], [46, 22, 2, 11, 2, 12, 46], [60, 28, 4, 15, 0, 0, 60], [66, 26, 4, 13, 1, 14, 66],
            [86, 26, 4, 14, 2, 15, 86], [100, 24, 4, 12, 4, 13, 100], [122, 28, 6, 15, 2, 16, 122],
            [140, 24, 3, 12, 8, 13, 140], [158, 28, 7, 14, 4, 15, 158], [180, 22, 12, 11, 4, 12, 180],
            [197, 24, 11, 12, 5, 13, 197], [223, 24, 11, 12, 7, 13, 223], [253, 30, 3, 15, 13, 16, 253],
            [283, 28, 2, 14, 17, 15, 283], [313, 28, 2, 14, 19, 15, 313], [341, 26, 9, 13, 16, 14, 341],
            [385, 28, 15, 15, 10, 16, 385], [406, 30, 19, 16, 6, 17, 406], [442, 24, 34, 13, 0, 0, 442],
            [464, 30, 16, 15, 14, 16, 464], [514, 30, 30, 16, 2, 17, 514], [538, 30, 22, 15, 13, 16, 538],
            [596, 30, 33, 16, 4, 17, 596], [628, 30, 12, 15, 28, 16, 628], [661, 30, 11, 15, 31, 16, 661],
            [701, 30, 19, 15, 26, 16, 701], [745, 30, 23, 15, 25, 16, 745], [793, 30, 23, 15, 28, 16, 793],
            [845, 30, 19, 15, 35, 16, 845], [901, 30, 11, 15, 46, 16, 901], [961, 30, 59, 16, 1, 17, 961],
            [986, 30, 22, 15, 41, 16, 986], [1054, 30, 2, 15, 64, 16, 1054], [1096, 30, 24, 15, 46, 16, 1096],
            [1142, 30, 42, 15, 32, 16, 1142], [1222, 30, 10, 15, 67, 16, 1222], [1276, 30, 20, 15, 61, 16, 1276]
        ]
    ]
    version_information = ['000111110010010100', '001000010110111100', '001001101010011001', '001010010011010011',
                           '001011101111110110', '001100011101100010', '001101100001000111', '001110011000001101',
                           '001111100100101000', '010000101101111000', '010001010001011101', '010010101000010111',
                           '010011010100110010', '010100100110100110', '010101011010000011', '010110100011001001',
                           '010111011111101100', '011000111011000100', '011001000111100001', '011010111110101011',
                           '011011000010001110', '011100110000011010', '011101001100111111', '011110110101110101',
                           '011111001001010000', '100000100111010101', '100001011011110000', '100010100010111010',
                           '100011011110011111', '100100101100001011', '100101010000101110', '100110101001100100',
                           '100111010101000001', '101000110001101001']  # Note versions 7 to 40 inclusive
    remainder_bits = [0, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3,
                      3, 3, 3, 0, 0, 0, 0, 0, 0]
    alignment_pattern_locations = [[], [], [6, 18], [6, 22], [6, 26], [6, 30], [6, 34], [6, 22, 38], [6, 24, 42],
                                   [6, 26, 46], [6, 28, 50], [6, 30, 54], [6, 32, 58], [6, 34, 62], [6, 26, 46, 66],
                                   [6, 26, 48, 70], [6, 26, 50, 74], [6, 30, 54, 78], [6, 30, 56, 82], [6, 30, 58, 86],
                                   [6, 34, 62, 90], [6, 28, 50, 72, 94], [6, 26, 50, 74, 98], [6, 30, 54, 78, 102],
                                   [6, 28, 54, 80, 106], [6, 32, 58, 84, 110], [6, 30, 58, 86, 114],
                                   [6, 34, 62, 90, 118], [6, 26, 50, 74, 98, 122], [6, 30, 54, 78, 102, 126],
                                   [6, 26, 52, 78, 104, 130], [6, 30, 56, 82, 108, 134], [6, 34, 60, 86, 112, 138],
                                   [6, 30, 58, 86, 114, 142], [6, 34, 62, 90, 118, 146], [6, 30, 54, 78, 102, 126, 150],
                                   [6, 24, 50, 76, 102, 128, 154], [6, 28, 54, 80, 106, 132, 158],
                                   [6, 32, 58, 84, 110, 136, 162], [6, 26, 54, 82, 110, 138, 166],
                                   [6, 30, 58, 86, 114, 142, 170]]

    def __init__(self, error_correction_level=0, minimum_version=1, mask_pattern=-1):
        self.error_correction_level = error_correction_level
        self.minimum_version = minimum_version
        self.mask_pattern = mask_pattern
        self.ecc_info = self.ecc_word_and_block_info[error_correction_level][minimum_version]
        self.total_bits_required = self.ecc_info[0] * 8
        self.bit_string = ''
        self.groups = []
        self.eccs = []
        self.polynomial_manager = Polynomials()

    def generate(self, input_data):
        mode, cci, data = self.analyse_character_encoding(input_data)
        self.pad_bits(input_data, mode, cci, data)
        self.create_groups_and_blocks()
        self.interleave()

    def create_groups_and_blocks(self):
        self.create_blocks_for_group(0)

        if self.ecc_info[4] != 0:
            self.create_blocks_for_group(1)

    def create_blocks_for_group(self, group_num):
        num_blocks = self.ecc_info[2 * group_num + 2]
        block_length = 8 * self.ecc_info[2 * group_num + 3]
        offset = self.ecc_info[2] * self.ecc_info[3] * 8 * group_num
        group = [self.bit_string[offset + i * block_length: offset + (i + 1) * block_length] for i in range(num_blocks)]
        group_polynomials = [[format(int(block[i: i + 8], 2), 'd') for i in range(0, len(block), 8)]
                             for block in group]
        self.groups.append(group_polynomials)
        self.eccs.append(self.generate_error_correction_codewords(group_polynomials))

    def interleave(self):
        interleaved_data = []
        min_length = self.ecc_info[3]
        max_length = self.ecc_info[5]
        for i in range(min_length):
            for group in self.groups[0]:
                interleaved_data.append(int(group[i]))
            if self.ecc_info[4] != 0:
                for group in self.groups[1]:
                    interleaved_data.append(int(group[i]))
        if self.ecc_info[4] != 0:
            for i in range(min_length, max_length):
                for group in self.groups[1]:
                    interleaved_data.append(int(group[i]))
        interleaved_eccs = []
        min_length = self.ecc_info[1]
        for i in range(min_length):
            for group in self.eccs[0]:
                interleaved_eccs.append(group[i])
            if self.ecc_info[4] != 0:
                for group in self.eccs[1]:
                    interleaved_eccs.append(group[i])
        self.bit_string = self.create_bits_from_bytes([i for i in interleaved_data + interleaved_eccs])
        self.bit_string += '0' * self.remainder_bits[self.minimum_version]

    @staticmethod
    def create_bits_from_bytes(input_data):
        temp = ['{0:08b}'.format(i) for i in input_data]
        return ''.join(temp)

    def generate_error_correction_codewords(self, polynomials):
        eccs = []
        num_gen_poly_terms = self.ecc_info[1]
        generator_polynomial = self.polynomial_manager.generator(num_gen_poly_terms - 1, in_alpha_format=False)
        for polynomial in polynomials:
            int_poly = [int(x) for x in polynomial]
            remainder = self.polynomial_manager.divide_polynomials(int_poly, generator_polynomial)
            eccs.append(remainder)
        return eccs

    def pad_bits(self, input_data, mode, cci, data):
        bit_string_length, cci = self.verify_version_number(input_data, mode, cci, data)
        # Add on terminator bits
        min_terminator = min(self.total_bits_required - bit_string_length, 4)
        self.bit_string = mode + cci + data + '0' * min_terminator
        bit_string_length += min_terminator
        # pad current byte with zeros (if required)
        next_mult_of_8 = bit_string_length if bit_string_length % 8 == 0 else ((bit_string_length // 8) + 1) * 8
        byte_pad_length = next_mult_of_8 - bit_string_length
        byte_padding = '0' * byte_pad_length
        self.bit_string += byte_padding
        # pad bit string to fill total bits required
        byte_pads = ['11101100', '00010001']
        pointer = 0
        while len(self.bit_string) < self.total_bits_required:
            self.bit_string += byte_pads[pointer]
            pointer = 1 - pointer

    def verify_version_number(self, input_data, mode, cci, data):
        bit_string_length = len(mode) + len(cci) + len(data)
        while bit_string_length > self.total_bits_required:
            # need to recalculate minimum suitable version
            bits_per_version = [ecc[0] * 8 for ecc in self.ecc_word_and_block_info[self.error_correction_level]]
            min_calc_ver = next(idx for idx, value in enumerate(bits_per_version) if value >= bit_string_length)
            self.minimum_version = self.minimum_version if self.minimum_version >= min_calc_ver else min_calc_ver
            self.ecc_info = self.ecc_word_and_block_info[self.error_correction_level][self.minimum_version]
            self.total_bits_required = self.ecc_info[0] * 8
            cci = self.calculate_character_count_indicator(input_data)
            bit_string_length = len(mode) + len(cci) + len(data)
        return bit_string_length, cci

    def analyse_character_encoding(self, input_data):
        if all([c.isdigit() for c in input_data]):
            # Can all characters be represented as numbers?
            self.encoder = 'numeric'
            char_count_ind = self.calculate_character_count_indicator(input_data)
            return self.encoding[self.encoder], char_count_ind, self.create_numeric_data_segment(input_data)
        elif all([c in '1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:' for c in input_data]):
            # Can all characters be represented as alphanumerics (including spaces, $, %, *, +, -, ., /, and :)?
            self.encoder = 'alphanumeric'
            char_count_ind = self.calculate_character_count_indicator(input_data)
            return self.encoding[self.encoder], char_count_ind, self.create_alphanumeric_data_segment(input_data)
        else:
            # Default to bytes
            self.encoder = 'byte'
            char_count_ind = self.calculate_character_count_indicator(input_data)
            return self.encoding[self.encoder], char_count_ind, self.create_bytes_data_segment(input_data)

    def calculate_character_count_indicator(self, input_data):
        num_chars = len(input_data.encode('utf-8'))     # get string length in bytes
        if 1 <= self.minimum_version <= 9:
            char_count_indicator = f'{num_chars:0>10b}' if self.encoder == 'numeric' else f'{num_chars:0>9b}' if\
                self.encoder == 'alphanumeric' else f'{num_chars:0>8b}'
        elif 10 <= self.minimum_version <= 26:
            char_count_indicator = f'{num_chars:0>12b}' if self.encoder == 'numeric' else f'{num_chars:0>11b}' if\
                self.encoder == 'alphanumeric' else f'{num_chars:0>16b}'
        elif 27 <= self.minimum_version <= 40:
            char_count_indicator = f'{num_chars:0>14b}' if self.encoder == 'numeric' else f'{num_chars:0>13b}' if\
                self.encoder == 'alphanumeric' else f'{num_chars:0>16b}'
        else:
            raise ValueError(f'Invalid version number: {self.minimum_version}')
        return char_count_indicator

    @staticmethod
    def encode_number_group(number_group):
        result = ''
        if len(number_group) == 3:
            result += f'{int(number_group):0>10b}'
        elif len(number_group) == 2:
            result += f'{int(number_group):0>7b}'
        else:
            result += f'{int(number_group):0>4b}'
        return result

    def create_numeric_data_segment(self, input_data):
        result = ''
        num_groups = [input_data[i: i + 3] for i in range(0, len(input_data), 3)]
        for num_group in num_groups:
            result += self.encode_number_group(num_group)
        return result

    def encode_character_group(self, char_group):
        if len(char_group) > 1:
            value = 45 * self.alphanumeric_lookup.index(char_group[0]) + self.alphanumeric_lookup.index(char_group[1])
            return f'{value:0>11b}'
        else:
            value = self.alphanumeric_lookup.index(char_group[0])
            return f'{value:0>6b}'

    def create_alphanumeric_data_segment(self, input_data):
        result = ''
        char_groups = [input_data[i: i + 2] for i in range(0, len(input_data), 2)]
        for char_group in char_groups:
            result += self.encode_character_group(char_group)
        return result

    @staticmethod
    def create_bytes_data_segment(input_data):
        result = ''
        temp = [list(map(bin, bytearray(i, 'utf8'))) for i in input_data]
        for tt in temp:
            for t in tt:
                r = t.replace('0b', '')
                r = r.rjust(8, '0')
                result += r
        return result
