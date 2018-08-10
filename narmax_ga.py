import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x, a=1):
    return 1 / (1 + np.exp(-x*a))


class NARMAX:
    def __init__(self, inp_del=0, out_del=1):
        self._weights = np.random.uniform(low=-1.0, high=1.0, size=1+1+inp_del+out_del)
        self._input_delays = np.zeros(inp_del)
        self._output_delays = np.zeros(out_del)
        self._weights_len = len(self._weights)
        self._inp_del_num = inp_del
        self._out_del_num = out_del

    def __call__(self, inp_data):
        ld = len(inp_data)
        result = np.zeros((ld, 1))
        for i in range(ld):
            delays = np.concatenate((self._input_delays, self._output_delays))
            result[i] = sigmoid(self._weights[0] + self._weights[1]*inp_data[i] + np.matmul(self._weights[2:], delays))
            # Input delays
            if self._inp_del_num != 0:
                self._input_delays[0], self._input_delays[1:] = inp_data[i], self._input_delays[0:-1]
            # Output delays
            self._output_delays[0], self._output_delays[1:] = result[i], self._output_delays[0:-1]
        return result

    def fault(self, input_data, target_data):
        return np.mean(np.abs((target_data - self(input_data)) / target_data))

    def set_weights(self, new_weights):
        self._weights = new_weights

    def change_weight(self, position, val):
        self._weights[position] = self._weights[position] + self._weights[position] * val

    def get_weights(self):
        return self._weights


class GeneticAlgorithm:
    def __init__(self, inp_del=0, out_del = 1, population_size=20):
        self._population = []
        self._population_size = population_size
        for _ in range(population_size):
            self._population.append(NARMAX(inp_del=inp_del, out_del=out_del))

    def fault(self, input_data, target_data):
        faults = []
        for animal in range(self._population_size):
            faults.append(self._population[animal].fault(input_data, target_data))
        return faults


if __name__ == "__main__":
    Data = np.genfromtxt('Data_JC.csv', delimiter=',')
    time = Data[0:-1:1000, 0]
    fuel = Data[0:-1:1000, 1] / 4.0
    freq = Data[0:-1:1000, 2] / 200000.0
    temp = Data[0:-1:1000, 3] / 1000.0

    model = NARMAX(5, 5)
    res = model(fuel)
    err = model.fault(fuel, freq)
    print(err)

    pop = GeneticAlgorithm()
    pop.fault(fuel, )

    plt.plot(res)
    plt.show()
