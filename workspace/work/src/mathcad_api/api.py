from scipy.stats import binom
from .utils import tolist
from matplotlib.pyplot import *


class MathcadAPI:
    @staticmethod
    def rbinom(size, max_num, p):
        return [dig for dig in binom.rvs(n=max_num, p=p, size=size)]

    @staticmethod
    def rnorm(m, mew, sigma):
        return np.random.normal(mew, sigma, m)

    @staticmethod
    def stack(*stacks):
        return np.concatenate(stacks)


    @staticmethod
    def submatrix(matrix, column1, column2, line1, line2):
        if line1 > line2:
            raise Exception('2 аргумент должен быть меньше 3 аргумента')

        if column1 > column2:
            raise Exception('4 аргумент должен быть меньше 5 аргумента')

        # Проверка на вектор
        lines = 0  # Если останется 0, то это вектор
        for i in matrix:
            if type(i) == list:
                lines += 1

        if not lines:  #  То это вектор
            if line1 == line2 == 0:
                return matrix[column1:column2 + 1]

        return [matrix[line][column1:column2+1] for line in range(line1, line2+1)]

    def subcolumn(self, matrix, column):
        matrix_tmp = list()
        for x in self.submatrix(matrix, column, column, 0, len(matrix) - 1):
            matrix_tmp.append(*x)

        return matrix_tmp

    @staticmethod
    def last(matrix):
        # Подсчитываем кол-во списков внутри главного списка
        count = 0
        for i in matrix:
            if type(i) == list:
                count += 1

            if count > 1:
                raise Exception('Длину последнего элемента узнать можно только в векторе')

        return len(matrix) - 1

    @staticmethod
    def get_errors(m1, m2):
        length = len(m1) if len(m1) < len(m2) else len(m2)
        return sum([m1[i] ^ m2[i] for i in range(length)])

    @staticmethod
    def get_error_chance(m1, m2):
        length = len(m1) if len(m1) < len(m2) else len(m2)
        return sum([m1[i] ^ m2[i] for i in range(length)]) / len(m1)

    @staticmethod
    def dec2bin(x, n):
        N = list()
        for i in range(n - 1):
            if x >= pow(2, n - 1 - i):
                N.insert(i, 1)
                x += pow(-2, n - 1 - i)
            N.insert(i, 0)
        return N

    def dec2binM(self, x, n):
        k = list()
        for i in range(1, self.last(x)):
            k = self.stack(k, self.dec2bin(x[i], n))
        return self.submatrix(k, 1, self.last(k), 0, 0)

    def bin2dec(self, t, j):
        B = list()
        for i in range(int(len(t) / j - 1)):
            B.insert(i, self.bin2decM(self.submatrix(t, j * i, j * (i + 1) - 1, 0, 0)))
        return B

    def bin2decM(self, x):
        s = 0
        for i in range(len(x) - 1):
            s += x[i] * pow(2, len(x) - i - 1)
        return s

    def trunc(self, z, y):
        new_z = []
        for i in z:
            new_z.append(i // y)

        new_z = [i * y for i in new_z]
        return new_z