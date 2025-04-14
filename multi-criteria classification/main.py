import math
import random
import matplotlib.pyplot as plt
import numpy as np

eras = 10
learning_factor = 0.1
beta = 5
input_list = [[4.0, 2.0, -1.0], [0.01, -1.0, 3.5], [0.01, 2.0, 0.01],
              [-1.0, 2.5, -2.0], [-1.5, 2.0, 1.5]]
learning_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
weights_list = [[0.0629, -0.0805, -0.0685], [0.0812, -0.0443, 0.0941], [-0.746, 0.0094, 0.0914],
                [0.0827, 0.0915, -0.0029], [0.0265, 0.0930, 0.0601]]
characteristics = ["Liczba nóg", "Czy żyje w wodzie?", "Czy umie latać?", "Czy ma pióra?", "Czy jest jajorodne?"]


class Neuron:
    def __init__(self, number):
        self.weights = np.array(weights_list).transpose()[number]  # generate_weights()
        self.excitation_signal = 0.0
        self.activated_value = 0.0

    def set_excitation_signal(self, u):
        self.excitation_signal = u

    def set_activated_value(self, y):
        self.activated_value = y


def generate_weights():
    return [random.uniform(-0.1, 0.1) for _ in range(5)]


def activate(value):
    return 1 / (1 + pow(math.e, -beta * value))


def train(P, T, neurons_list):
    whole_result = []
    for i in range(eras):
        example_number = random.randint(0, 2)
        X = P[example_number]
        for neuron in neurons_list:
            neuron.set_excitation_signal(np.dot(X, neuron.weights))
            neuron.set_activated_value(activate(neuron.excitation_signal))

        error = []
        for j in range(len(neurons_list)):
            error.append(T[example_number][j] - neurons_list[j].activated_value)
            neurons_list[j].weights += learning_factor * X * error[j]

        tmp = [neuron.weights.copy() for neuron in neurons_list]
        whole_result.append(tmp)

    return whole_result


def test(animal, neurons_list):
    tmp = []
    for neuron in neurons_list:
        tmp.append(np.dot(animal, neuron.weights))

    max_index = np.argmax(tmp)
    tmp = [0] * len(neurons_list)
    tmp[max_index] = 1

    return tmp


def which_animal(value):
    animal_dict = {(1, 0, 0): "Ssak", (0, 1, 0): "Ptak", (0, 0, 1): "Ryba"}
    return animal_dict.get(tuple(value), "Nieznane zwierzę")


def draw_plot(array, neuron_number, title):
    array = np.array(array)
    tmp = [array[i][neuron_number] for i in range(eras)]
    tmp = np.array(tmp)

    for i in range(len(tmp.transpose())):
        plt.plot(tmp.transpose()[i], label=f'Cecha: {characteristics[i]}')

    plt.title(f'Wartości wag dla neuronu {title}')
    plt.xlabel('Epoki')
    plt.ylabel('Wagi')
    plt.legend()
    plt.show()


def main():
    mammal = Neuron(0)
    bird = Neuron(1)
    fish = Neuron(2)
    neurons_list = [mammal, bird, fish]
    P = np.array(input_list).transpose()
    T = np.array(learning_list)

    res = train(P, T, neurons_list)

    draw_plot(res, 0, "Ssak")
    draw_plot(res, 1, "Ptak")
    draw_plot(res, 2, "Ryba")

    cow = [4, 0, 0, 0, 0]
    turkey = [2, 0, 1, 1, 1]
    salmon = [0, 1, 0, 0, 1]
    flamingo = [2, 0, 1, 1, 1]
    human = [2, 0, 0, 0, 0]

    animals = {'Krowa': cow, 'Indyk': turkey, 'Łosoś': salmon, 'Flaming': flamingo, 'Człowiek': human}

    for animal_name, animal_features in animals.items():
        result = test(animal_features, neurons_list)
        print(f'Zwierze: {animal_name} | Cechy: {animal_features} | Wynik: {result} | Gatunek: {which_animal(result)}')


if __name__ == "__main__":
    main()
