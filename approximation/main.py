import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.01
max_iterations = 1000
delta_threshold = 0.01
number_of_points = 20


def generate_points():
    np.random.seed(42)
    X = np.random.rand(number_of_points, 1) * 10
    Y = 2 * X + 3 + np.random.randn(number_of_points, 1)
    return X, Y


def generate_factors():
    return np.random.randn(), np.random.randn()


def linear_function(a, b, x):
    return a * x + b


def mean_squared_error(Y, predicted_y):
    return np.mean((Y - predicted_y) ** 2)


def draw_plot(a, b, X, Y, iterations):
    plt.scatter(X, Y, facecolors='none', edgecolors='blue', marker='o', label='Punkty pomiarowe')
    plt.plot(X, linear_function(a, b, X), color='red', label=f'Prosta aproksymacyjna')
    plt.title(f'Aproksymacja linią prostą po {iterations} iteracjach')
    plt.xlabel('Czas')
    plt.ylabel('Wartości')
    plt.legend()
    plt.show()


def train(X, Y):
    a_factor, b_factor = generate_factors()
    for iteration in range(max_iterations):
        predicted_y = linear_function(a_factor, b_factor, X)

        error = mean_squared_error(Y, predicted_y)

        gradient_a = np.mean(X * (Y - predicted_y))
        gradient_b = np.mean(Y - predicted_y)

        a_factor = a_factor + learning_rate * gradient_a
        b_factor = b_factor + learning_rate * gradient_b

        new_error = mean_squared_error(Y, linear_function(a_factor, b_factor, X))

        delta_error = np.abs(new_error - error)

        if delta_error <= delta_threshold:
            print(f'Zbieżność osiągnięta po {iteration} iteracjach.')
            break
        if iteration % 50 == 0:
            draw_plot(a_factor, b_factor, X, Y, iteration)
    draw_plot(a_factor, b_factor, X, Y, iteration)


def main():
    x_list, y_list = generate_points()
    plt.scatter(x_list, y_list, facecolors='none', edgecolors='blue', marker='o')
    plt.title("Losowe punkty pomiarowe")
    plt.xlabel('Czas')
    plt.ylabel('Wartości')
    plt.show()
    train(x_list, y_list)


if __name__ == "__main__":
    main()
