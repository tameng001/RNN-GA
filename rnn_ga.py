import numpy as np
import pandas as pd
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import copy


def change_matrix_size(matrix, newmat_rows, newmat_cols):
    """
    Change matrix from one dimension into other dimension, saving matrix "character"
    :param matrix: numpy.array
    :param newmat_rows: int
    :param newmat_cols: int
    :type matrix: numpy.array
    :type newmat_rows: int
    :type newmat_cols: int
    :return: numpy.array
    """
    mat_rows, mat_cols = matrix.shape
    rm = np.linspace(0, 1, mat_rows)
    cm = np.linspace(0, 1, mat_cols)
    rn = np.linspace(0, 1, newmat_rows)
    cn = np.linspace(0, 1, newmat_cols)
    result = np.zeros((newmat_rows, newmat_cols))
    if newmat_cols == 1 and newmat_rows == 1:
        result = matrix.mean()
    elif mat_rows == 1 and mat_cols == 1:
        result = np.full((newmat_rows, newmat_cols), matrix[0, 0])
    elif mat_rows == 1:
        interp_spline = interp.interp1d(cm, matrix)
        for r in range(newmat_rows):
            result[r, :] = interp_spline(cn)
    elif mat_cols == 1:
        interp_spline = interp.interp1d(rm, matrix.transpose())
        for c in range(newmat_cols):
            result[:, c] = interp_spline(rn)
    else:
        interp_spline = interp.RectBivariateSpline(rm, cm, matrix, kx=1, ky=1, s=0)
        result = interp_spline(rn, cn)
    return result


def change_tensor3d_size(tensor3d, new_z, new_y, new_x):
    z, y, x = tensor3d.shape
    intermediate_tensor = np.zeros((z, new_y, new_x))
    for i in range(z):
        intermediate_tensor[i, :, :] = change_matrix_size(tensor3d[i, :, :], new_y, new_x)
    final_tensor = np.zeros((new_z, new_y, new_x))
    for j in range(new_x):
        final_tensor[:, :, j] = change_matrix_size(intermediate_tensor[:, :, j], new_z, new_y)
    return final_tensor


def sigmoid(x, a=1):
    return 1 / (1 + np.exp(-x*a))


def resigmoid(y, a=1):
    return np.log(y / (1 - y)) / a


def sigsum(matrix):
    return sum(sigmoid(matrix.sum(axis=1)))


def show_color_matrix(matrix):
    plt.imshow(matrix)
    plt.colorbar()
    plt.show()


class RNN:
    def __init__(self, structure=(1, 1, 1), delays=1, genome_size=10):
        """
        Recurrent neural network
        :param structure: (inputs, hidden_neurons, outputs)
        :param delays: Number of delays
        :param genome_size: Size of genome matrixes
        :type structure: (int, int, int)
        :type delays: int
        :type genome_size: int
        """
        self.__inp = structure[0]
        self.__hid = structure[1]
        self.__out = structure[2]
        self.__del = delays
        self.__genome_size = genome_size
        self.__info = {}

        self.__gen_w1 = np.random.uniform(low=-10, high=10, size=[genome_size, genome_size])
        self.__gen_w2 = np.random.uniform(low=-10, high=10, size=[genome_size, genome_size])
        self.__gen_wd = np.random.uniform(low=-10, high=10, size=[genome_size, genome_size, genome_size])

        self.__w1 = change_matrix_size(self.__gen_w1, self.__hid, self.__inp + 1)
        self.__w2 = change_matrix_size(self.__gen_w2, self.__out, self.__hid + 1)
        self.__wd = change_tensor3d_size(self.__gen_wd, self.__del, self.__hid, self.__out)

        self.__outdel = np.zeros([self.__del, self.__out, 1])
        self.__fault = None

    def __call__(self, data):
        """
        Calculating RNN
        :param data: numpy.array with input vectors in the columns and time steps in the rows
        :return: numpy.array with input vectors in the columns and time steps in the rows
        """
        result = np.zeros((len(data), self.__out))
        for d in range(len(data)):
            # Weighted sum of inputs with bias
            layer_result = data[d]
            layer_result = np.matmul(self.__w1, np.append(layer_result, 1.0))
            # layer_result = np.matmul(self.__w1, np.append(layer_result, 1.0)) + \
            #                         np.sum(np.matmul(self.__wd, self.__outdel), 0).transpose()
            # Calculating hidden layer
            layer_result = sigmoid(layer_result)
            # Weighted hidden layer
            layer_result = np.matmul(self.__w2, np.append(layer_result, 1.0))
            # Calculating output signal
            layer_result = sigmoid(layer_result)
            # Shift of output signal in the delay matrix
            for i in range(self.__del - 1, -1, -1):
                if i == 0:
                    self.__outdel[i, :, 0] = layer_result
                else:
                    self.__outdel[i, :, 0] = self.__outdel[i-1, :, 0]
            result[d] = layer_result
        return result

    def calculate_fault(self, input_data, target_data):
        """
        Calculating of model faults
        :param input_data:
        :param target_data:
        :return: None
        """
        error = target_data - self(input_data)
        self.__fault = np.mean(np.abs(error))
    """
    def updates_weights(self):

        Recalculate sizes and values of weights
        :return: None

        self.__w1 = change_matrix_size(self.__gen_w1, self.__hid, self.__inp + 1)
        self.__w2 = change_matrix_size(self.__gen_w2, self.__out, self.__hid + 1)
        self.__wd = change_tensor3d_size(self.__gen_wd, self.__del, self.__hid, self.__out)
        self.__outdel = np.zeros([self.__del, self.__out, 1])
        """

    def updates_weights(self):
        """
        Recalculate sizes and values of weights
        Алгоритм делает так, что сумма сигмоид от новых матриц весов равнялась аналогичной сумме в старых весах.
        Это нужно чтобы при изменении размеров матриц итоговый результат не менялся
        Меняет только для выходного слоя, т.к. что было в нейросети остаётся в нейросети и для обратной связи
        :return: None
        """
        self.__w1 = change_matrix_size(self.__gen_w1, self.__hid, self.__inp + 1)

        new_w2 = change_matrix_size(self.__gen_w2, self.__out, self.__hid + 1)
        w2_sum_new = new_w2.sum()
        w2_sum_old = self.__w2.sum()
        w2_sum_col = abs(new_w2).sum(axis=0)
        w2_sum_row = abs(new_w2).sum(axis=1)
        delta_w2 = w2_sum_old - w2_sum_new
        rows, cols = new_w2.shape
        for r in range(rows):
            for c in range(cols):
                delta_rows = abs(new_w2[r, c] / w2_sum_row[r])
                delta_cols = abs(new_w2[r, c] / w2_sum_col[c])
                new_w2[r, c] = new_w2[r, c] + delta_w2 * delta_rows * delta_cols
        self.__w2 = new_w2
        chek_w2 = w2_sum_old - self.__w2.sum(axis=1)

        new_wd = change_tensor3d_size(self.__gen_wd, self.__del, self.__hid, self.__out)
        wd_sum_new = new_wd.sum()
        wd_sum_old = self.__wd.sum()
        delta_wd = wd_sum_old - wd_sum_new
        wd_sum_lst = abs(new_wd).sum(axis=2).sum(axis=1)
        wd_sum_row = abs(new_wd).sum(axis=2).sum(axis=0)
        wd_sum_col = abs(new_wd).sum(axis=1).sum(axis=0)
        lists, rows, cols = new_wd.shape
        for l in range(lists):
            for r in range(rows):
                for c in range(cols):
                    delta_lsts = abs(new_wd[l, r, c] / wd_sum_lst[l])
                    delta_rows = abs(new_wd[l, r, c] / wd_sum_row[r])
                    delta_cols = abs(new_wd[l, r, c] / wd_sum_col[c])
                    new_wd[l, r, c] = new_wd[l, r, c] + delta_wd * delta_lsts * delta_rows * delta_cols
        self.__wd = new_wd
        check_wd = wd_sum_old - self.__wd.sum()

        self.__outdel = np.zeros([self.__del, self.__out, 1])

    def change_gen_w1(self, row, col, val):
        """
        Increase in 'val' times in the 'row' and 'col' matrix coordinate in the weights before hidden layer
        :param row: Row coordinate of genome matrix
        :param col: Column coordinate of genome matrix
        :param val: Value for change
        :return: None
        """
        self.__gen_w1[row, col] += self.__gen_w1[row, col] * val
        self.__fault = None

    def change_gen_w2(self, row, col, val):
        """
        Increase in 'val' times in the 'row' and 'col' matrix coordinate in the weights before output layer
        :param row: Row coordinate of genome matrix
        :param col: Column coordinate of genome matrix
        :param val: Value for change
        :return: None
        """
        self.__gen_w2[row, col] += self.__gen_w2[row, col] * val
        self.__fault = None

    def change_gen_wd(self, lay, row, col, val):
        """
        Increase in 'val' times in the 'lay', 'row' and 'col' matrix coordinate in the weights matrix
        :param lay: Layer of delay tensor. Each layer is connected with 'lay' output
        :param row: Row coordinate of genome matrix
        :param col: Column coordinate of genome matrix
        :param val: Value for change
        :return: None
        """
        self.__gen_wd[lay, row, col] += self.__gen_wd[lay, row, col] * val
        self.__fault = None

    def insert_gen_w1_row(self, new_row, row_number):
        self.__gen_w1[row_number, :] = copy.deepcopy(new_row)
        self.__fault = None

    def insert_gen_w2_row(self, new_row, row_number):
        self.__gen_w2[row_number, :] = copy.deepcopy(new_row)
        self.__fault = None

    def insert_gen_wd_row(self, new_row, lay_number, row_number):
        self.__gen_wd[lay_number, row_number, :] = copy.deepcopy(new_row)
        self.__fault = None

    def insert_gen_w1_col(self, new_col, col_number):
        self.__gen_w1[:, col_number] = copy.deepcopy(new_col)
        self.__fault = None

    def insert_gen_w2_col(self, new_col, col_number):
        self.__gen_w2[:, col_number] = copy.deepcopy(new_col)
        self.__fault = None

    def insert_gen_wd_col(self, new_col, lay_number, col_number):
        self.__gen_wd[lay_number, :, col_number] = copy.deepcopy(new_col)
        self.__fault = None

    def set_inputs(self, new_inputs):
        """
        Change number of inputs and recalculate matrixes
        :param new_inputs:
        :return:
        """
        self.__inp = new_inputs
        self.__fault = None

    def set_hid(self, new_hid):
        """
        Changing number of hidden neurons
        :param new_hid: new number of hidden neurons
        :return: None
        """
        self.__hid = new_hid
        self.__fault = None

    def set_outputs(self, new_outputs):
        self.__out = new_outputs
        self.__fault = None

    def set_delays(self, new_delays):
        """
        Changing number of delays
        :param new_delays: new number of hidden neurons
        :return: None
        """
        self.__del = new_delays
        self.__fault = None

    def set_genome_size(self, new_genome_size):
        """
        Change size of neural network genome matrixes
        :param new_genome_size: New genome size
        :return: None
        """
        self.__genome_size = new_genome_size
        self.__gen_w1 = change_matrix_size(self.__gen_w1, self.__genome_size, self.__genome_size)
        self.__gen_w2 = change_matrix_size(self.__gen_w2, self.__genome_size, self.__genome_size)
        self.__gen_wd = change_tensor3d_size(self.__gen_wd, self.__genome_size, self.__genome_size, self.__genome_size)

    def set_gen_w1(self, new_gen_w1):
        self.__gen_w1 = new_gen_w1

    def set_gen_w2(self, new_gen_w2):
        self.__gen_w2 = new_gen_w2

    def set_gen_wd(self, new_gen_wd):
        self.__gen_wd = new_gen_wd

    def get_inputs(self): return self.__inp

    def get_hid(self): return self.__hid

    def get_outputs(self): return self.__out

    def get_delays(self): return self.__del

    def get_genome_size(self): return self.__genome_size

    def get_fault(self): return self.__fault

    def get_w1(self): return self.__w1

    def get_w2(self): return self.__w2

    def get_wd(self): return self.__wd

    def get_gen_w1(self): return self.__gen_w1

    def get_gen_w2(self): return self.__gen_w2

    def get_gen_wd(self): return self.__gen_wd

    def show(self):
        """
        Print information about network's structure end weights
        :return: text in the console
        """
        print("Network structure: [{}, {}, {}]".format(self.__inp, self.__hid, self.__out))
        print("Number of delays: {}".format(self.__del))
        print("Network fault: {}".format(self.__fault))

    def display(self):
        """
        Display colored genome
        :return:
        """
        pass


class GeneticAlgorithm:
    def __init__(self, input_data, target_data, population_size=20, network_inputs=1, network_outputs=1,
                 max_hid=100, max_delays=10, genome_size=10):
        self.__input_data = input_data
        self.__target_data = target_data

        self.__population = {}
        self.__population_size = population_size
        self.__network_inputs = network_inputs
        self.__network_outputs = network_outputs
        self.__max_hid = max_hid
        self.__max_delays = max_delays
        self.__genome_size = genome_size

        for p in range(population_size):
            hidden_layer = int(np.random.uniform(1, max_hid, 1))
            delays = int(np.random.uniform(1, max_delays, 1))
            self.__population['nn{}'.format(p)] = RNN(structure=(network_inputs, hidden_layer, network_outputs),
                                                      delays=delays)

        for animal in self.__population:
            # Calculating animal faults
            self.get_animal(animal).calculate_fault(input_data=input_data, target_data=target_data)

    def live(self, generations=10000, kill=0.25, max_fault=0.01):
        """
        Life of population with death, sex and the meaning of life
        :param generations: Number of generations for population life
        :param kill: Part of population to be killed
        :param max_fault: Fault value to stop algorithm
        :return: Survived and mutated generation with the best characteristics
        """
        plt.figure()
        plt.ion()

        fig1 = plt.subplot(2, 2, 1)
        fig1.grid()
        fig1.set_xlabel('Generations')
        fig1.set_ylabel('Fault')

        fig2 = plt.subplot(2, 2, 2)
        fig2.grid()
        fig2.set_xlabel('Number of hidden neurons')
        fig2.set_ylabel('Number of delays in the networks')
        fig2.set_xlim([0, self.__max_hid + 1])
        fig2.set_ylim([0, self.__max_delays + 1])

        fig3 = plt.subplot(2, 2, 3)
        fig3.grid()
        fig3.set_xlabel('Network number')
        fig3.set_ylabel('Fault')
        fig3.set_xlim([0, self.__population_size])
        fig3.set_ylim([0, 1])

        fig4 = plt.subplot(2, 2, 4)
        fig4.grid()
        fig4.set_xlabel('Time')
        fig4.set_ylabel('Freq')

        for g in range(generations):
            self.__survive(kill)
            self.__breading()
            self.__mutate(mutate_animal=int(self.__population_size * (1 - kill)))
            self.calculate_population_faults()

            faults = self.__get_faults()

            print('Step {}, error = {}'.format(g, faults))

            fig1.plot(g, min(faults), 'k.')

            fig2.clear()
            fig2.grid()
            fig2.set_xlabel('Number of hidden neurons')
            fig2.set_ylabel('Number of delays in the networks')
            fig2.set_xlim([0, self.__max_hid + 1])
            fig2.set_ylim([0, self.__max_delays + 1])
            fig2.plot(self.__get_hiddens(), self.__get_delays(), 'k.')

            fig3.clear()
            fig3.grid()
            fig3.set_xlabel('Network number')
            fig3.set_ylabel('Fault')
            fig3.set_xlim([0, self.__population_size])
            fig3.set_ylim([0, 1])
            fig3.plot(faults, 'k.')

            fig4.clear()
            fig4.grid()
            fig4.set_xlabel('Time')
            fig4.set_ylabel('Freq')
            fig4.plot(self.get_animal('nn0')(self.__input_data))
            fig4.plot(self.__target_data)

            plt.pause(0.01)

            if min(self.__get_faults()) < max_fault:
                break
        plt.ioff()
        plt.show()

    def __survive(self, kill):
        """
        Take the best animals from population
        :param kill: Part of population which should be killed
        :return None
        """
        animal_faults = {}
        for animal in self.__population:
            animal_faults[animal] = self.get_animal(animal).get_fault()
        sorted_animals = sorted(animal_faults.items(), key=lambda item: item[1])
        new_population = {}
        for n in range(int(self.get_population_size() * (1 - kill))):
            animal = sorted_animals[n][0]
            new_population['nn{}'.format(n)] = self.get_animal(animal)
        self.__population = new_population

    def __mutate(self, mutate_animal=5, mutate_number=1):
        # choose_animal = np.random.choice(list(self.__population.keys()), size=mutate_animal)
        choose_animal = list(self.__population.keys())[self.__population_size-mutate_animal:]
        for animal in choose_animal:
            for m in range(mutate_number):
                choose_param = np.random.choice(['delays', 'hidden', 'w1', 'w2', 'wd'])
                if choose_param == 'delays':
                    new_delays = int(self.__population[animal].get_delays() + np.random.uniform(low=-2, high=2, size=1))
                    if new_delays < 1:
                        new_delays = 1
                    elif new_delays > self.__max_delays:
                        new_delays = self.__max_delays
                    self.__population[animal].set_delays(new_delays)
               
                if choose_param == 'hidden':
                    new_hidden = int(self.__population[animal].get_hid() + np.random.uniform(low=-2, high=2, size=1))
                    if new_hidden < 1:
                        new_hidden = 1
                    elif new_hidden > self.__max_hid:
                        new_hidden = self.__max_hid
                    self.__population[animal].set_hid(new_hidden)

                if choose_param == 'w1':
                    row = np.random.randint(low=0, high=self.__population[animal].get_gen_w1().shape[0], size=1)
                    col = np.random.randint(low=0, high=self.__population[animal].get_gen_w1().shape[1], size=1)
                    val = np.random.uniform(low=-0.1, high=0.1, size=1)
                    self.__population[animal].change_gen_w1(row=row, col=col, val=val)

                elif choose_param == 'w2':
                    row = np.random.randint(low=0, high=self.__population[animal].get_gen_w2().shape[0], size=1)
                    col = np.random.randint(low=0, high=self.__population[animal].get_gen_w2().shape[1], size=1)
                    val = np.random.uniform(low=-0.1, high=0.1, size=1)
                    self.__population[animal].change_gen_w2(row=row, col=col, val=val)

                else:
                    lay = np.random.randint(low=0, high=self.__population[animal].get_gen_wd().shape[0], size=1)
                    row = np.random.randint(low=0, high=self.__population[animal].get_gen_wd().shape[1], size=1)
                    col = np.random.randint(low=0, high=self.__population[animal].get_gen_wd().shape[2], size=1)
                    val = np.random.uniform(low=-0.1, high=0.1, size=1)
                    self.__population[animal].change_gen_wd(lay=lay, row=row, col=col, val=val)
            self.__population[animal].updates_weights()

    def __breading(self):
        size_exist_population = len(self.__population)
        # animal_names = list(self.__population.keys())
        animal_names = ['nn0', 'nn1']
        for n in range(size_exist_population, self.__population_size, 1):
            new_animal = RNN(structure=(1, 5, 1), delays=1)
            new_animal.set_inputs(self.get_inputs())
            new_animal.set_outputs(self.get_outputs())
            new_animal.set_delays(self.__population[np.random.choice(animal_names)].get_delays())
            new_animal.set_hid(self.__population[np.random.choice(animal_names)].get_hid())
            new_animal.set_gen_w1(self.__population[np.random.choice(animal_names)].get_gen_w1())
            new_animal.set_gen_w2(self.__population[np.random.choice(animal_names)].get_gen_w2())
            new_animal.set_gen_wd(self.__population[np.random.choice(animal_names)].get_gen_wd())
            """
            for rows in range(0, self.__genome_size):
                new_animal.insert_gen_w1_row(
                    new_row=self.__population[np.random.choice(animal_names)].get_gen_w1()[rows, :],
                    row_number=rows
                )
                new_animal.insert_gen_w2_row(
                    new_row=self.__population[np.random.choice(animal_names)].get_gen_w2()[rows, :],
                    row_number=rows
                )
                for lays in range(0, self.__genome_size):
                    new_animal.insert_gen_wd_row(
                        new_row=self.__population[np.random.choice(animal_names)].get_gen_wd()[lays, rows, :],
                        lay_number=lays,
                        row_number=rows
                    )
            """
            for cols in range(0, self.__genome_size):
                new_animal.insert_gen_w1_col(
                    new_col=self.__population[np.random.choice(animal_names)].get_gen_w1()[:, cols],
                    col_number=cols
                )
                new_animal.insert_gen_w2_col(
                    new_col=self.__population[np.random.choice(animal_names)].get_gen_w2()[:, cols],
                    col_number=cols
                )
                for lays in range(0, self.__genome_size):
                    new_animal.insert_gen_wd_col(
                        new_col=self.__population[np.random.choice(animal_names)].get_gen_wd()[lays, :, cols],
                        lay_number=lays,
                        col_number=cols
                    )
            
            new_animal.updates_weights()

            # new_animal.calculate_fault(self.__input_data, self.__target_data)
            self.__population['nn{}'.format(n)] = new_animal

    """
    def set_data(self, input_data, target_data):

        !!!Isn't tested!!! Change input and target data with recalculating all neural network faults.
        :param input_data: Data in the input of neural networks
        :param target_data: Desired output of neural networks
        :type input_data: numpy.array
        :type target_data: numpy.array
        :return: None

        self.__input_data = input_data
        self.__target_data = target_data

        _, inp_cols = input_data.shape
        _, tar_cols = target_data.shape

        if self.__network_inputs != inp_cols:
            for animal in self.__population:
                self.get_animal(animal).set_inputs(new_inputs=inp_cols)

        if self.__network_outputs != tar_cols:
            for animal in self.__population:
                self.get_animal(animal).set_outputs(new_outputs=tar_cols)

        self.calculate_population_faults()
    """
    def calculate_population_faults(self):
        """
        Calculating faults in overall population networks.
        :return: None
        """
        for animal in self.__population:
            self.__population[animal].calculate_fault(input_data=self.__input_data,
                                                      target_data=self.__target_data)
    """
    def set_population_size(self, new_population_size):

        Change population size with adding new networks or deleting networks
        in alphabetical order starting from the end.
        :param new_population_size: New population size
        :type new_population_size: int
        :return: None
        if new_population_size > self.__population_size:
            #for p in range(start=self.__population_size, stop=new_population_size, step=1):
            for p in np.arange(start=self.__population_size, stop=new_population_size, step=1, dtype=int):
                hidden_layer = int(np.random.uniform(1, self.__max_hid, 1))
                delays = int(np.random.uniform(1, self.__max_delays, 1))
                self.__population['nn{}'.format(p)] = RNN(structure=(self.__network_inputs,
                                                                     hidden_layer,
                                                                     self.__network_outputs),
                                                          delays=delays,
                                                          genome_size=self.__genome_size)
                self.get_animal('nn{}'.format(p)).calculate_fault(input_data=self.__input_data,
                                                                  target_data=self.__target_data)
        else:
            for p in np.arange(start=self.__population_size-1, stop=new_population_size, step=-1, dtype=int):
                del self.__population['nn{}'.format(p)]

        self.__population_size = new_population_size
    """
    def get_animal(self, name): return self.__population[name]

    def get_population(self): return self.__population

    def get_inputs(self): return self.__network_inputs

    def get_outputs(self): return self.__network_outputs

    def get_population_size(self): return self.__population_size

    def get_max_hid(self): return self.__max_hid

    def get_max_delays(self): return self.__max_delays

    def get_population_genome_size(self): return self.__genome_size

    def __get_faults(self):
        faults = []
        for p in self.__population:
            faults.append(self.get_animal(p).get_fault())
        return faults

    def __get_delays(self):
        delays = []
        for p in self.__population:
            delays.append(self.__population[p].get_delays())
        return delays

    def __get_hiddens(self):
        hiddens = []
        for p in self.__population:
            hiddens.append(self.__population[p].get_hid())
        return hiddens

    def print_population(self):
        """Print pandas table - each row is network, and columns are: (inputs, hidden, outputs), delays, fault"""
        structure = []
        delays = []
        faults = []
        for p in self.__population:
            structure.append('({}, {}, {})'.format(self.get_animal(p).get_inputs(),
                                                   self.get_animal(p).get_hid(),
                                                   self.get_animal(p).get_outputs()))
            delays.append(self.get_animal(p).get_delays())
            faults.append(self.get_animal(p).get_fault())
        zoo = pd.DataFrame({
            'structure': structure,
            'delays': delays,
            'faults': faults
        })
        print('Maximum population size: {}'.format(self.__population_size))
        print('Number of inputs in each network: {}'.format(self.__network_inputs))
        print('Number of outputs in each network: {}'.format(self.__network_outputs))
        print('Genome size: {}'.format(self.__genome_size))
        print('Maximum number of hidden neurons in each network: {}'.format(self.__max_hid))
        print('Maximum number of delays in each network: {}'.format(self.__max_delays))
        print(zoo)


if __name__ == "__main__":
    Data = np.genfromtxt('Data_JC.csv', delimiter=',')
    time = Data[0:-1:1000, 0]
    fuel = Data[0:-1:1000, 1] / 4.0
    freq = Data[0:-1:1000, 2] / 200000.0
    temp = Data[0:-1:1000, 3] / 1000.0

    time = time.reshape(len(time), 1)
    fuel = fuel.reshape(len(fuel), 1)
    freq = freq.reshape(len(freq), 1)
    temp = temp.reshape(len(temp), 1)

    pop = GeneticAlgorithm(input_data=fuel, target_data=freq, max_hid=20, population_size=50)
    pop.live(generations=100000, kill=0.8)
    pop.print_population()
    network = pop.get_animal('nn0')
    net_result = network(fuel)
    plt.plot(time, net_result, time, freq)
    plt.show()
