import numpy as np
from math import sqrt
from data import generate_data_line, plot_generated_data
import matplotlib.pyplot as plt


def plot_data(x, y, LR, name):
    plt.scatter(x, y)
    # plt = abline(LR.m, LR.c, plt)
    # axes = plt.gca()
    x_vals = np.array([x.min(), x.max()])
    y_vals = LR.m * x_vals + LR.c
    plt.plot(x_vals, y_vals, "--")
    plt.savefig("images/" + name)
    plt.clf()
    return


class LinearReg:
    """ A bi-variable linear regressor class """

    def __init__(self, m=np.random.randint(0, 5), c=np.random.randint(0, 5), lr=0.1):
        """initialize the class with:
        slope(m), intercept(c) and learning rate(lr)"""
        self.m = m
        self.c = c
        self.lr = lr
        # print(f"Initialized with LR.m: {self.m:.4f}, LR.c: {self.c:.4f}, lr: {self.lr:.4f}")

    def mean_squared_error(self, y, pred_y):
        """Calculate the MSE for vectors

        Args:
            y (np.array): actual y values
            pred_y (np.array): predicted y values

        Returns:
            float: the MSE from y and pred_y
        """
        return sqrt(np.mean((y - pred_y) ** 2))

    def update_parameters(self, err, x):
        N = x.shape[0]
        dm = (2.0 / N) * (-np.sum(np.multiply(x, (err))))
        dc = (2.0 / N) * (-np.sum(err))
        # print(dm, dc)
        self.m = self.m - (self.lr * dm)
        self.c = self.c - (self.lr * dc)
        return

    def train(self, x, y, epochs=501):
        cost_history = []
        for epoch in range(epochs):
            pred_y = self.m * x + self.c
            err = y - pred_y
            self.update_parameters(err, x)

            cost = self.mean_squared_error(y, pred_y)
            cost_history.append(cost)

            if epoch % 100 == 0:
                print(f"LR.m: {self.m:.4f}, LR.c: {self.c:.4f}, cost: {cost:.4f}")
                name = str(epoch) + "_epoch.png"
                plot_data(x, y, self, name)

        return cost_history


if __name__ == "__main__":
    data = generate_data_line(num=10)
    x, y = np.array(list(zip(*data)))
    plot_generated_data(x, y)

    LR = LinearReg(m=np.mean(x), c=np.mean(y))
    cost_history = LR.train(x, y)

    print(LR.m)
    print(LR.c)

    plt.plot(cost_history)
    plt.savefig("images/cost_history.png")