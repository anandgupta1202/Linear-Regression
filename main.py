import numpy as np
from math import sqrt
from data import generate_data_line


class LinearReg:
    """ A bi-variable linear regressor class """

    def __init__(
        self, m=np.random.randint(0, 1),
        c=np.random.randint(0, 1), lr=0.1
    ):
        self.m = m
        self.c = c
        self.lr = lr

    def mean_squared_error(self, y, pred_y):
        return sqrt(np.mean((y - pred_y) ** 2))

    def update_parameters(self, err, x):
        N = x.shape[0]
        dm = (2.0 / N) * (-np.sum(np.multiply(x, (err))))
        dc = (2.0 / N) * (-np.sum(err))
        self.m = self.m + self.lr * dm
        self.c = self.c + self.lr * dc
        return

    def train(self, x, y, n_iter=20):
        for i in range(n_iter):
            i += 1
            pred_y = self.m * x + self.c
            err = self.mean_squared_error(y, pred_y)
            self.update_parameters(err, x)


def plot_data(x, y, LR):
    pass


if __name__ == "__main__":
    data = generate_data_line(num=5)
    x, y = np.array(list(zip(*data)))
    # x, y = zip(*data)
    # print(x)
    # print(y)

    LR = LinearReg(m=np.mean(x), c=np.mean(y))
    LR.train(x, y)

    print(LR.m)
    print(LR.c)
