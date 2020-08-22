import numpy as np


def find_y(m, x, c):
    # print(type(m*x+c))
    return m*x+c


def generate_data_line(m=0.3, c=0.01, num=100):
    mean, std_dev = 0, 0.1
    x = np.random.normal(mean, std_dev, num)
    # print(x)
    y = find_y(m, x, c)
    # print(y)
    data = zip(x, y)
    return data


if __name__ == "__main__":
    print(find_y(x=0.42, m=0.0, c=0.01))
    generate_data_line(num=5)
