import gym
import random
import numpy as np
from numpy import *
import math
from statistics import median
from matplotlib import pyplot as plt
from scipy import signal

pendulum = gym.make('LunarLander-v2')
pendulum.reset()

i = 0


def ConvertWeights(_):
    X = i_node * h_l_1
    Y = X + h_l_2 * h_l_1
    Z = Y + h_l_2 * o_node

    i_w, h_1, h_2 = _[0, 0:X], _[0, X:Y], _[0, Y:Z]

    i_w = i_w.reshape(-1, i_node)
    # i_w = (i_w - mean(i_w))/std(i_w)
    h_1 = h_1.reshape(-1, h_l_1)
    # h_1 = (h_1 - mean(h_1))/std(h_1)
    h_2 = h_2.reshape(-1, h_l_2)
    # h_2 = (h_2 - mean(h_2))/std(h_2)
    return i_w, h_1, h_2


class Network():

    def __init__(self, inp_nodes, hidden_nodes_1, hidden_nodes_2, out_nodes):
        self.i_n = inp_nodes
        self.h_1 = hidden_nodes_1
        self.h_2 = hidden_nodes_2
        self.o_n = out_nodes
        self.func = [False, False, False, False]

    def SetActFunc(self, function):
        if function == 'sigmoid':
            self.func = [True, False, False, False]
        elif function == 'relu':
            self.func = [False, True, False, False]
        elif function == 'lrelu':
            self.func == [False, False, True, False]
        elif function == 'softmax':
            self.func = [False, False, False, True]
        else:
            print('Error: Mentioned', function,
                  'is not present in function list. Kindly choose among \'sigmoid\' for Sigmoid output, \'relu\' for ReLU outptu, \'lrelu\' for Leaky ReLU or \'softmax\' for Softmax function output')

    def Fit(self, inp):

        self.i_l = np.asarray(inp).reshape(-1, 1)
        weight_ = self.i_w
        out = np.dot(weight_, inp)

        self.h_l_1 = self.Act(out).reshape(-1, 1)
        weight_ = self.h_1
        out = np.dot(weight_, self.h_l_1)

        self.h_l_2 = self.Act(out).reshape(-1, 1)
        weight_ = self.h_2
        out = np.dot(weight_, self.h_l_2)

        self.func = [False, False, False, True]
        self.o_l = self.Act(out).reshape(-1, 1)
        return np.argmax(self.o_l)

    def Act(self, _):
        if self.func[0] and not self.func[1] and not self.func[2] and not self.func[3]:
            print(self.func[3])
            return 1 / (1 + np.exp(-_))
        elif not self.func[0] and self.func[1] and not self.func[2] and not self.func[3]:
            return np.maximum(0, _)
        elif not self.func[0] and not self.func[1] and self.func[2] and not self.func[3]:
            return _ * self.alpha
        elif self.func[0] and self.func[1] and not self.func[2] and not self.func[3]:
            return np.clip(_, -1, 1)
        else:
            return np.exp(_ - np.max(_)) / np.sum(np.exp(_ - np.max(_)))

    def SetAlpha(self, alpha):
        try:
            if self.func == [False, False, True, False]:
                self.alpha = alpha
        except AttributeError:
            print('Activation function not defined. Set Alpha value after defining activation function')

    def SetWeights(self, i_w, h_1, h_2):
        self.i_w = i_w
        self.h_1 = h_1
        self.h_2 = h_2


class GA():

    def __init__(self, i_n, h_n_1, h_n_2, o_n):
        self.i_n = i_n
        self.h_n_1 = h_n_1
        self.h_n_2 = h_n_2
        self.o_n = o_n

    def GetWeights(self):

        # He Weight Initialization
        i_w = np.random.randn(self.h_n_1, self.i_n) * np.sqrt(2 / self.i_n)
        h_1 = np.random.randn(self.h_n_2, self.h_n_1) * np.sqrt(2 / self.h_n_1)
        h_2 = np.random.randn(self.o_n, self.h_n_2) * np.sqrt(2 / self.h_n_2)
        threshold = np.random.rand()
        return i_w, h_1, h_2, threshold

    def mutate(self, child):
        k = random.randint(0, child.shape[1])
        des = random.randint(0, 10)
        if des <= 5:
            for i in range(k):
                limit = random.randint(0, child.shape[1])
                mutation = random.randint(-300, 300)
                child[0, limit] += mutation

        return child

    # def crossover(self, par_1, par_2):
    def Crossover(self, parents):
        new_population = []
        for _ in range(population - parent_pop):
            child = []
            n1 = random.randint(0, len(parents))
            parent_1 = parents[n1]
            # del parents[n]
            n2 = random.randint(0, len(parents))
            while n2 == n1:
                n2 = random.randint(0, len(parents))
            parent_2 = parents[n2]
            # del parents[n]

            for i in range(parent_1.shape[1]):
                if i % 2 == 0:
                    child = np.append(child, parent_1[0, i])
                else:
                    child = np.append(child, parent_2[0, i])
            mutated_child = self.mutate(child.reshape(1, -1))
            new_population.append(mutated_child)

        return parents + new_population

    def Evolution(self, score, gene):
        n = random.randint(0, len(score))
        parents = []
        parent_pop = random.randint(10, 70)
        for i in range(parent_pop):
            loc_1 = score.index(max(score))
            score.remove(max(score))
            parents.append(gene[loc_1].reshape(1, -1))
            del gene[loc_1]

        return self.Crossover(parents)

    def natures_first(self, population, iteration_time):
        population_award = []
        population_gene_pool = []
        for _ in range(population):
            i_w, h_1, h_2, threshold = GAmodel.GetWeights()

            # i_w, h_1, h_2 = ConvertWeights(np.load('Lunar_MED2.npy',allow_pickle=True).reshape(1, -1))
            observation = pendulum.reset()
            award = 0
            for __ in range(iteration_time):
                # pendulum.render()
                model = Network(4, 4, 2, 1)
                model.SetWeights(i_w, h_1, h_2)
                model.SetActFunc('relu')
                action = model.Fit(observation)
                observation, reward, done, info = pendulum.step(action)
                award += reward
                if done:
                    break
            population_award.append(award)
            chromosome = np.concatenate((i_w.flatten(), h_1.flatten(), h_2.flatten()))
            population_gene_pool.append(chromosome)
        return population_award, population_gene_pool


parent_pop = 0
generations = 70
population = 80
iteration_time = 700

i_node = 8
h_l_1 = 36
h_l_2 = 36
o_node = 4
GAmodel = GA(i_node, h_l_1, h_l_2, o_node)
model = Network(4, 7, 4, 1)
model.SetActFunc('relu')

pop_award, pop_gene = GAmodel.natures_first(population, iteration_time)

best_awards_gen = []
med_awards_gen = []
avg_awards_gen = []
it = []
PID = []
prev = 0
iterations = []
iterations_2 = []

avg = -99
current_award = -9999
generation_awards = []
i = 0
diff = 0

for gen in range(generations):
# while True:
    new_population = GAmodel.Evolution(pop_award, pop_gene)
    pop_award = []
    pop_gene = []
    for _ in new_population:
        observation = pendulum.reset()
        input_weight, hidden_1, hidden_2 = ConvertWeights(_)
        model.SetWeights(input_weight, hidden_1, hidden_2)

        award = 0
        for x in range(iteration_time):
            # pendulum.render()
            model.SetActFunc('relu')
            action = model.Fit(observation)

            observation, reward, done, info = pendulum.step(int(action))
            award += reward
            if done:
                break
        PID.append(award)
        pop_award.append(award)

        chromosome = np.concatenate((input_weight.flatten(), hidden_1.flatten(), hidden_2.flatten()))
        pop_gene.append(chromosome)

    avg_awards_gen = np.append(avg_awards_gen, np.average(PID))
    best_awards_gen = np.append(best_awards_gen, np.amax(pop_award))  # Store Maximum of Each Generation
    med_awards_gen = np.append(med_awards_gen, np.median(PID))

    if np.median(PID) >= current_award:
        current_award = max(current_award, np.median(PID))
        np.save('Lunar_MED2', pop_gene[pop_award.index(max(pop_award))])
    if max(pop_award) > prev:
        np.save('Lunar_BEST2', pop_gene[pop_award.index(max(pop_award))])
        prev = max(pop_award)
    i += 1
    print('[Generation: %3d] [Best Median:%5d]  [Median Score:%5d] [Top Score: %3d] [History: %3d]' % (
    i, round(current_award, 2), round(np.median(PID), 2), np.amax(pop_award), prev))
    if current_award > 200:
        break

pendulum.close()
t = linspace(0, i, i)
plt.plot(t, best_awards_gen, 'r', label='Best Fitness Scores')
plt.plot(t, avg_awards_gen, 'g', label='Average Fitness Scores')
plt.plot(t, med_awards_gen, 'b', label='Median Fitness Scores')

plt.title('Fitness vs Generation Plot')
plt.xlabel('Generations')
plt.ylabel('Fitness Score')
plt.grid()
print('Average', np.mean(best_awards_gen), 0)
plt.legend(loc='lower right')
plt.show()


