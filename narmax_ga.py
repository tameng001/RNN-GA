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
        self._fault = None

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
        self._fault = np.mean(np.abs((target_data - self(input_data)) / target_data))
        return self._fault

    def set_weights(self, new_weights):
        self._weights = new_weights

    def change_weight(self, position, val):
        self._weights[position] = self._weights[position] + self._weights[position] * val

    def get_weights(self):
        return self._weights

    def get_fault(self):
        return self._fault

    def get_len_weights(self):
        return self._weights_len


class GeneticAlgorithm:
    def __init__(self, inp_data, tar_data, inp_del=0, out_del=1, population_size=20):
        self._population = {}
        self._inp_del = inp_del
        self._out_del = out_del
        self._inp_data = inp_data
        self._tar_data = tar_data
        self._population_size = population_size
        for p in range(population_size):
            self._population['narmax_{}'.format(p)] = NARMAX(inp_del=inp_del, out_del=out_del)

    def fault(self, input_data, target_data):
        faults = []
        for animal in self._population:
            faults.append(self._population[animal].fault(input_data, target_data))
        return faults

    def _selection(self, kill=0.25):
        animal_faults = {}
        for animal in self._population:
            animal_faults[animal] = self.get_animal(animal).get_fault()
        sorted_animals = sorted(animal_faults.items(), key=lambda item: item[1])
        new_population = {}
        for n in range(int(self.get_population_size() * (1 - kill))):
            animal = sorted_animals[n][0]
            new_population['narmax_{}'.format(n)] = self.get_animal(animal)
        self._population = new_population

    def _breading(self):
        size_exist_population = len(self._population)
        # animal_names = list(self.__population.keys())
        animal_names = ['narmax_0', 'narmax_1']
        len_w = self._population['narmax_0'].get_len_weights()
        point_cut = np.random.randint(low=1, high=len_w - 1, size=self._population_size)
        for n in range(size_exist_population, self._population_size, 1):
            new_animal = NARMAX(inp_del=self._inp_del, out_del=self._out_del)
            w1 = self._population[np.random.choice(animal_names)].get_weights()[0:point_cut[n]]
            w2 = self._population[np.random.choice(animal_names)].get_weights()[point_cut[n]:]
            new_animal.set_weights(np.concatenate((w1, w2)))
            self._population['narmax_{}'.format(n)] = new_animal

    def _mutate(self):
        pos = np.random.randint(0, self._population['narmax_0'].get_len_weights(), self._population_size)
        amp = 0.1
        val = np.random.uniform(-amp, amp, self._population_size)
        n = 0
        for animal in self._population:
            self._population[animal].change_weight(position=pos[n], val=val[n])
            n += 1

    def live(self, generations=10000, kill=0.25, max_fault=0.01):
        plt.figure()
        plt.ion()

        fig1 = plt.subplot(1, 2, 1)
        fig1.grid()
        fig1.set_xlabel('Generations')
        fig1.set_ylabel('Fault')

        fig2 = plt.subplot(1, 2, 2)
        fig2.grid()
        fig2.set_xlabel('Model number')
        fig2.set_ylabel('Fault')
        fig2.set_xlim([0, self._population_size])

        faults = self.fault(self._inp_data, self._tar_data)

        for g in range(generations):
            self._selection(kill=kill)
            self._breading()
            self._mutate()
            faults = self.fault(self._inp_data, self._tar_data)

            print('Step {}, error = {}'.format(g, faults))

            fig1.plot(g, min(faults), 'k.')

            fig2.clear()
            fig2.grid()
            fig2.set_xlabel('Network number')
            fig2.set_ylabel('Fault')
            fig2.set_xlim([0, self._population_size])
            fig2.set_ylim([0, 1])
            fig2.plot(faults, 'k.')

            plt.pause(0.01)

            if max(faults) < max_fault:
                break

        plt.ioff()
        plt.show()

    def get_animal(self, name):
        return self._population[name]

    def get_population_size(self):
        return self._population_size

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

    pop = GeneticAlgorithm(fuel, freq)
    pop.live()

    plt.plot(res)
    plt.show()
