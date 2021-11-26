import math
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.random import rand
from scipy import optimize


def F(x, a, b, c, d):
    return (a * x + b) / (pow(x, 2) + c * x + d)


def f(x):
    return 1 / (pow(x, 2) - 3 * x + 2)


k = np.arange(0, 1001, 1)
delta = np.random.normal(size=len(k))

y = []
x_list = []
for i in k:
    x_k = 3 * i / 1000
    x_list.append(x_k)
    if f(x_k) < -100:
        y.append(-100 + delta[i])
    elif (f(x_k) >= -100) and (f(x_k) <= 100):
        y.append(f(x_k) + delta[i])
    else:
        y.append(100 + delta[i])

x = np.asarray(x_list)


def d_func_scp_lm(x):
    resudials = []
    for i in range(1001):
        f = (y[i] - F(x_list[i], x[0], x[1], x[2], x[3])) ** 2
        resudials.append(f)
    return np.array(resudials)


def d_func_scp(x):
    alg_sum = 0
    for i in range(1001):
        f = (y[i] - F(x_list[i], x[0], x[1], x[2], x[3])) ** 2
        alg_sum += f
    return np.mean(alg_sum)


bounds = np.asarray([[0.00001, 1.0], [0.00001, 1.0], [0.00001, 1.0], [0.00001, 1.0]])
x0 = np.asarray(bounds[:, 0])

res_lm = optimize.least_squares(d_func_scp_lm, x0, method="lm", verbose=2)
a_M, b_M, c_M, d_M = res_lm.x

plt.scatter(x, y)
plt.plot(x, F(x, a_M, b_M, c_M, d_M), 'r-')
plt.xlabel('x_k')
plt.ylabel('y_k')
plt.legend(["LM", "Generated data"], loc="lower right")
plt.show()

res0_10 = optimize.minimize(d_func_scp, x0, method='Nelder-Mead', options={'disp': True, 'maxiter': 3000})
a_N, b_N, c_N, d_N = res0_10.x
plt.scatter(x, y)
plt.plot(x, F(x, a_N, b_N, c_N, d_N), 'r-')
plt.xlabel('x_k')
plt.ylabel('y_k')
plt.legend(["Nelder-Mead method", "Generated data"], loc="lower right")
plt.show()
from scipy.optimize import differential_evolution

result = differential_evolution(d_func_scp, bounds, disp=True)
a_E, b_E, c_E, d_E = result.x

plt.scatter(x, y)
plt.plot(x, F(x, a_E, b_E, c_E, d_E), 'y-')
plt.xlabel('x_k')
plt.ylabel('y_k')
plt.legend(["Differential evolution method", "Generated data"], loc="lower right")
plt.show()

from scipy.optimize import dual_annealing

annealing = dual_annealing(d_func_scp, bounds)

a_A, b_A, c_A, d_A = annealing.x

plt.scatter(x, y)
plt.plot(x, F(x, a_A, b_A, c_A, d_A), 'k-')
plt.xlabel('x_k')
plt.ylabel('y_k')
plt.legend(["Annealing method", "Generated data"], loc="lower right")
plt.show()

plt.scatter(x, y)
plt.plot(x, F(x, a_A, b_A, c_A, d_A), 'k-')
plt.plot(x, F(x, a_E, b_E, c_E, d_E), 'y-')
plt.plot(x, F(x, a_N, b_N, c_N, d_N), 'r-')
plt.plot(x, F(x, a_M, b_M, c_M, d_M), 'g-')
plt.xlabel('x_k')
plt.ylabel('y_k')
plt.legend(["Annealing method", "Differential evolution method", "Nelder-Mead method", "Levenberg-Marquardt method",
            "Generated data"], loc="lower right")
plt.show()

import pandas as pd

lau_matrix = np.matrix(
    '0.549963E-07  0.985808E-08;-28.8733 -0.797739E-07;-79.2916 -21.4033;-14.6577 -43.3896;-64.7473 21.8982;-29.0585 -43.2167;-72.0785 0.181581;-36.0366 -21.6135;-50.4808 7.37447;-50.5859 -21.5882;-0.135819 -28.7293;-65.0866 -36.0625;-21.4983 7.31942;-57.5687 -43.2506;-43.0700  14.5548')
nodes = np.array(lau_matrix)


def vectorToDistMatrix(coords):
    return np.sqrt((np.square(coords[:, np.newaxis] - coords).sum(axis=2)))


def nearestNeighbourSolution(dist_matrix):
    node = random.randrange(len(dist_matrix))
    result = [node]

    nodes_to_visit = list(range(len(dist_matrix)))
    nodes_to_visit.remove(node)

    while nodes_to_visit:
        nearest_node = min([(dist_matrix[node][j], j) for j in nodes_to_visit], key=lambda x: x[0])
        node = nearest_node[1]
        nodes_to_visit.remove(node)
        result.append(node)

    return result


from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt


def animateTSP(history, points):
    ''' animate the solution over time
        Parameters
        ----------
        hisotry : list
            history of the solutions chosen by the algorithm
        points: array_like
            points with the coordinates
    '''

    ''' approx 1500 frames for animation '''
    key_frames_mult = len(history) // 1500

    fig, ax = plt.subplots()

    ''' path is a line coming through all the nodes '''
    line, = plt.plot([], [], lw=2)

    def init():
        ''' initialize node dots on graph '''
        x = [points[i][0] for i in history[0]]
        y = [points[i][1] for i in history[0]]
        plt.plot(x, y, 'co')

        ''' draw axes slighty bigger  '''
        extra_x = (max(x) - min(x)) * 0.05
        extra_y = (max(y) - min(y)) * 0.05
        ax.set_xlim(min(x) - extra_x, max(x) + extra_x)
        ax.set_ylim(min(y) - extra_y, max(y) + extra_y)

        '''initialize solution to be empty '''
        line.set_data([], [])
        return line,

    def update(frame):
        ''' for every frame update the solution on the graph '''
        x = [points[i, 0] for i in history[frame] + [history[frame][0]]]
        y = [points[i, 1] for i in history[frame] + [history[frame][0]]]
        line.set_data(x, y)
        return line

    ''' animate precalulated solutions '''


class SimulatedAnnealing:
    def __init__(self, coords, temp, alpha, stopping_temp, stopping_iter, curr_solution):
        self.coords = coords
        self.sample_size = len(coords)
        self.temp = temp
        self.alpha = alpha
        self.stopping_temp = stopping_temp
        self.stopping_iter = stopping_iter
        self.iteration = 1

        self.dist_matrix = vectorToDistMatrix(coords)
        self.curr_solution = curr_solution
        self.best_solution = self.curr_solution

        self.solution_history = [self.curr_solution]

        self.curr_weight = self.weight(self.curr_solution)
        self.initial_weight = self.curr_weight
        self.min_weight = self.curr_weight

        self.weight_list = [self.curr_weight]

        print('Intial weight: ', self.curr_weight)

    def weight(self, sol):
        return sum([self.dist_matrix[i, j] for i, j in zip(sol, sol[1:] + [sol[0]])])

    def acceptance_probability(self, candidate_weight):
        return math.exp(-abs(candidate_weight - self.curr_weight) / self.temp)

    def accept(self, candidate):
        candidate_weight = self.weight(candidate)
        if candidate_weight < self.curr_weight:
            self.curr_weight = candidate_weight
            self.curr_solution = candidate
            if candidate_weight < self.min_weight:
                self.min_weight = candidate_weight
                self.best_solution = candidate

        else:
            if random.random() < self.acceptance_probability(candidate_weight):
                self.curr_weight = candidate_weight
                self.curr_solution = candidate

    def anneal(self):
        while self.temp >= self.stopping_temp and self.iteration < self.stopping_iter:
            candidate = list(self.curr_solution)
            l = random.randint(2, self.sample_size - 1)
            i = random.randint(0, self.sample_size - l)

            candidate[i: (i + l)] = reversed(candidate[i: (i + l)])

            self.accept(candidate)
            self.temp *= self.alpha
            self.iteration += 1
            self.weight_list.append(self.curr_weight)
            self.solution_history.append(self.curr_solution)

        print('Minimum weight: ', self.min_weight)
        print('Improvement: ',
              round((self.initial_weight - self.min_weight) / (self.initial_weight), 4) * 100, '%')

    def get_solution(self):
        return self.solution_history[-1], self.coords

    def plotLearning(self):
        plt.plot([i for i in range(len(self.weight_list))], self.weight_list)
        line_init = plt.axhline(y=self.initial_weight, color='r', linestyle='--')
        line_min = plt.axhline(y=self.min_weight, color='g', linestyle='--')
        plt.legend([line_init, line_min], ['Initial weight', 'Optimized weight'])
        plt.ylabel('Weight')
        plt.xlabel('Iteration')
        plt.show()


temp = 1000
stopping_temp = 0.0000001
alpha = 0.97
stopping_iter = 1000
init_solution = [i for i in range(15)]

sa = SimulatedAnnealing(nodes, temp, alpha, stopping_temp, stopping_iter, init_solution)
sa.anneal()
nodes, positions = sa.get_solution()
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 15)

fig, ax = plt.subplots(2, sharex=True, sharey=True)  # Prepare 2 plots
ax[0].set_title('Raw nodes')
ax[1].set_title('Optimized tour')
ax[0].scatter(positions[:, 0], positions[:, 1])  # plot A
ax[1].scatter(positions[:, 0], positions[:, 1])  # plot B




start_node = 0
distance = 0.
for i in range(1, 15):
    start_pos = positions[init_solution[i - 1]]
    end_pos = positions[init_solution[i]]
    ax[1].annotate("",
                   xy=start_pos, xycoords='data',
                   xytext=end_pos, textcoords='data',
                   arrowprops=dict(arrowstyle="->",
                                   connectionstyle="arc3"))
    distance += np.linalg.norm(end_pos - start_pos)
    start_node = init_solution[i]

print(distance)
textstr = "Nodes count: %d\nTotal length: %.3f" % (15, distance)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax[1].text(0.05, 0.95, textstr, transform=ax[1].transAxes, fontsize=14,  # Textbox
           verticalalignment='top', bbox=props)


plt.show()



fig, ax = plt.subplots(2, sharex=True, sharey=True)  # Prepare 2 plots
ax[0].set_title('Raw nodes')
ax[1].set_title('Optimized tour')
ax[0].scatter(positions[:, 0], positions[:, 1])  # plot A
ax[1].scatter(positions[:, 0], positions[:, 1])  # plot B


start_node = 0
distance = 0.
for i in range(1, 15):
    start_pos = positions[nodes[i - 1]]
    end_pos = positions[nodes[i]]
    ax[1].annotate("",
                   xy=start_pos, xycoords='data',
                   xytext=end_pos, textcoords='data',
                   arrowprops=dict(arrowstyle="->",
                                   connectionstyle="arc3"))
    distance += np.linalg.norm(end_pos - start_pos)
    start_node = init_solution[i]


start_pos = positions[nodes[i]]
end_pos = positions[nodes[0]]
ax[1].annotate("",
                   xy=start_pos, xycoords='data',
                   xytext=end_pos, textcoords='data',
                   arrowprops=dict(arrowstyle="->",
                                   connectionstyle="arc3"))
print(distance)
textstr = "Nodes count: %d\nTotal length: %.3f" % (15, distance)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax[1].text(0.05, 0.95, textstr, transform=ax[1].transAxes, fontsize=14,  # Textbox
           verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

sa.plotLearning()
