class Polynomials:
    """
    All calculations must be performed within Galois field GF(256).
    Polynomials are represented in alpha notation.
                  10    251 9    67 8    46 7    61 6    118 5    70 4    64 3    94 2    32     45
    For example: x   + α   x  + α  x  + α  x  + α  x  + α   x  + α  x  + α  x  + α  x  + α  x + α
      where α represents the number 2 in GF(256)
    """
    exponents = [1, 2, 4, 8, 16, 32, 64, 128, 29, 58, 116, 232, 205, 135, 19, 38, 76, 152, 45, 90, 180, 117, 234, 201,
                 143, 3, 6, 12, 24, 48, 96, 192, 157, 39, 78, 156, 37, 74, 148, 53, 106, 212, 181, 119, 238, 193, 159,
                 35, 70, 140, 5, 10, 20, 40, 80, 160, 93, 186, 105, 210, 185, 111, 222, 161, 95, 190, 97, 194, 153, 47,
                 94, 188, 101, 202, 137, 15, 30, 60, 120, 240, 253, 231, 211, 187, 107, 214, 177, 127, 254, 225, 223,
                 163, 91, 182, 113, 226, 217, 175, 67, 134, 17, 34, 68, 136, 13, 26, 52, 104, 208, 189, 103, 206, 129,
                 31, 62, 124, 248, 237, 199, 147, 59, 118, 236, 197, 151, 51, 102, 204, 133, 23, 46, 92, 184, 109, 218,
                 169, 79, 158, 33, 66, 132, 21, 42, 84, 168, 77, 154, 41, 82, 164, 85, 170, 73, 146, 57, 114, 228, 213,
                 183, 115, 230, 209, 191, 99, 198, 145, 63, 126, 252, 229, 215, 179, 123, 246, 241, 255, 227, 219, 171,
                 75, 150, 49, 98, 196, 149, 55, 110, 220, 165, 87, 174, 65, 130, 25, 50, 100, 200, 141, 7, 14, 28, 56,
                 112, 224, 221, 167, 83, 166, 81, 162, 89, 178, 121, 242, 249, 239, 195, 155, 43, 86, 172, 69, 138, 9,
                 18, 36, 72, 144, 61, 122, 244, 245, 247, 243, 251, 235, 203, 139, 11, 22, 44, 88, 176, 125, 250, 233,
                 207, 131, 27, 54, 108, 216, 173, 71, 142, 1]

    @staticmethod
    def fix_exponent(exponent):
        return (exponent % 256) + (exponent // 256)

    def add_factors(self, factor1, factor2):
        val1 = self.exponents[factor1]
        val2 = self.exponents[factor2]
        new_val = val1 ^ val2
        return 0 if new_val == 0 else self.exponents.index(new_val)

    def multiply_factors(self, factor1, factor2):
        total = self.exponents[(self.exponents.index(factor1) + self.exponents.index(factor2)) % 255]

        return total

    def power_factors(self, factor1, factor2):
        return self.exponents[(self.exponents.index(factor1) * factor2) % 255]

    def generator(self, number_of_terms, in_alpha_format=True):
        alpha_polynomial = [0, 0]    # our initial representation of αx + α
        for i in range(number_of_terms):
            high = []
            low = []
            new_poly = [0, i + 1]
            for j in alpha_polynomial:
                new_high = self.fix_exponent(j + new_poly[0]) if j + new_poly[0] > 255 else j + new_poly[0]
                high.append(new_high)
                new_low = self.fix_exponent(j + new_poly[1]) if j + new_poly[1] > 255 else j + new_poly[1]
                low.append(new_low)
            alpha_polynomial = [high[0]] + [self.add_factors(low[k], high[k + 1]) for k in range(len(high) - 1)] +\
                               [low[-1]]
        return alpha_polynomial if in_alpha_format else self.convert_alpha_to_list(alpha_polynomial)

    def convert_value_to_alpha(self, value):
        return self.exponents.index(value)

    def convert_alpha_to_value(self, value):
        return self.exponents[value]

    def convert_list_to_alpha(self, in_list):
        return [self.convert_value_to_alpha(i) for i in in_list]

    def convert_alpha_to_list(self, alpha_list):
        return [self.convert_alpha_to_value(i) for i in alpha_list]

    def evaluate_polynomial(self, polynomial):
        pass

    def evaluate_alpha_polynomial(self, polynomial, value_to_evaluate=0):
        total = polynomial[0]
        for coefficient in polynomial[1:]:
            multiplied = self.multiply_factors(total, value_to_evaluate)
            subtotal = multiplied ^ coefficient
            total = self.fix_exponent(subtotal)
        return total

    def divide_polynomials(self, dividend, divisor):
        num_divisor_poly_terms = len(divisor)
        num_dividend_poly_terms = len(dividend)

        result = [x for x in dividend] + [0 for _ in range(num_divisor_poly_terms - 1)]
        divisor_alpha = self.convert_list_to_alpha(divisor)
        multiply_poly = [x for x in divisor_alpha] + [0 for _ in range(num_dividend_poly_terms - 1)]
        xor_poly = [x for x in result]

        i = 0
        while i < num_dividend_poly_terms:
            # multiply divisor polynomial by lead term of the dividend polynomial
            dividend_lead_term = self.convert_value_to_alpha(result[0])
            for i2 in range(num_divisor_poly_terms):
                multiply_poly[i2] = (divisor_alpha[i2] + dividend_lead_term) % 255
                multiply_poly[i2] = self.convert_alpha_to_value(multiply_poly[i2])
            result = [multiply_poly[i] ^ msg for i, msg in enumerate(xor_poly)]
            result = result[1:] + [0]
            while result[0] == 0:
                result = result[1:] + [0]                     # a leading zero means we can divide again
                i += 1
            xor_poly = [x for x in result]
            i += 1
            result = result[:(num_divisor_poly_terms - 1)]
        return result

#
# if __name__ == '__main__':
#     poly = Polynomials()
#     divid = [32, 91, 11, 120, 209, 114, 220, 77, 67, 64, 236, 17, 236, 17, 236, 17]
#     divis = [1, 216, 194, 159, 111, 199, 94, 95, 113, 157, 193]
#     a = poly.divide_polynomials(divid, divis)
#     print(divid)
#     print(divis)
#     print(a)
