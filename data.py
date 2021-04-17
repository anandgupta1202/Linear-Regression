import numpy as np
import matplotlib.pyplot as plt


def find_y(m, x, c):
    # print(type(m*x+c))
    return m * x + c


def generate_data_line(m=1, c=1, num=100):
    mean, std_dev = 1, 0.5
    x = np.random.normal(mean, std_dev, num)
    # print(x)
    y = find_y(m, x, c)
    # y = np.random.normal(mean, std_dev, num)
    # print(y)
    data = zip(x, y)
    return data


def plot_generated_data(x, y):
    plt.scatter(x, y)
    plt.savefig("images/generated_data.png")


if __name__ == "__main__":
    # print(find_y(x=0.42, m=0.0, c=0.01))
    data = generate_data_line(num=10)
    x, y = np.array(list(zip(*data)))

    plot_generated_data(x, y)
