import argparse
from copy import deepcopy
import copy
from matplotlib import colors, pyplot as plt
import matplotlib
import numpy as np
from numpy import random
from enum import Enum
from queue import Queue
from tqdm import tqdm

font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 18}
font = {'family': 'Times New Roman', 'size': 18}
matplotlib.rc('font', **font)
matplotlib.rcParams.update(
    {'font.family': 'Times New Roman', 'font.size': 18, })
plt.rcParams["font.family"] = "Times New Roman"

linew = 2
markerss = 5


class ACTION(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    PLANNING = 4


class Hole():
    def __init__(self, args, x, y) -> None:
        self.life_expectancy = random.randint(args.l_min, args.l_max+1)
        self.score = random.randint(args.s_min, args.s_max+1)
        self.age = 0
        self.x = x
        self.y = y


class TileWorld():
    def __init__(self, args) -> None:
        self.args = args
        self.grid_l = args.grid_l
        self.holes = []
        self.map = np.zeros((args.grid_l, args.grid_l), dtype=int).tolist()
        self.obstacles = args.obstacles
        self.g_min = args.g_min
        self.g_max = args.g_max
        self.a_x, self.a_y = None, None
        self.total_score = 0
        for x, y in self.obstacles:
            self.map[x][y] = 1
        self.g_time = 0

    def update(self, action) -> int:
        if self.g_time == 0:
            self._gen_holes()
        else:
            self.g_time -= 1

        if action == ACTION.DOWN:
            self.a_x += 1
        elif action == ACTION.UP:
            self.a_x -= 1
        elif action == ACTION.LEFT:
            self.a_y -= 1
        elif action == ACTION.RIGHT:
            self.a_y += 1
        else:
            pass
        assert self.a_x < self.args.grid_l and self.a_x >= 0 and self.a_y < self.args.grid_l and self.a_y >= 0

        score = 0
        for hole in self.holes[:]:
            if hole.age == hole.life_expectancy:
                self.holes.remove(hole)
        for hole in self.holes[:]:
            if hole.x == self.a_x and hole.y == self.a_y:
                score = hole.score
                self.holes.remove(hole)

        for hole in self.holes:
            hole.age += 1

        self.map = np.zeros(
            (self.args.grid_l, self.args.grid_l), dtype=int).tolist()

        for x, y in self.obstacles:
            self.map[x][y] = 1
        for hole in self.holes:
            self.map[hole.x][hole.y] = 2
        self.map[self.a_x][self.a_y] = 3
        return score

    def render_init(self):
        self.fig = plt.figure()
        plt.ion()

    def render(self):
        plt.clf()
        ax = self.fig.add_subplot(111)
        plt.title("Tile World")
        cmap = colors.ListedColormap(['red', 'blue', 'yellow', 'gray'])
        bounds = [0, 0.5, 1.5, 2.5, 3.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        ax.imshow(self.map, cmap=cmap, norm=norm)
        ax.grid(which='major', axis='both',
                linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-.5, self.args.grid_l, 1))
        ax.set_yticks(np.arange(-.5, self.args.grid_l, 1))
        plt.pause(0.01)

    def render_off(self):
        plt.ioff()
        plt.show()

    def _gen_holes(self):
        x = random.randint(0, self.grid_l)
        y = random.randint(0, self.grid_l)
        while (x, y) in self.obstacles:
            x = random.randint(0, self.grid_l)
            y = random.randint(0, self.grid_l)
        self.map[x][y] = 2
        self.holes.append(Hole(self.args, x, y))
        self.total_score += self.holes[-1].score
        self.g_time = random.randint(self.g_min, self.g_max+1)


class Agent():
    def __init__(self, args, env) -> None:
        self.args = args
        self.actions = []
        self.m = args.m
        self.k = args.k
        self.p = args.p
        self.time = 0
        self.step = 0
        self._born(env)
        self.score = 0
        self.target = None
        self.memory_hole = None
        self.reaction_strategy = args.reaction_strategy
        assert self.reaction_strategy == "blind" or self.reaction_strategy == "disapper" or self.reaction_strategy == "any_hole" or self.reaction_strategy == "nearer_hole"

    def update(self, env) -> ACTION:
        if self.time != 0:
            self.time -= 1
            return ACTION.PLANNING
        else:
            if self.step == self.args.k or len(self.actions) == 0 or (self.reaction_strategy != "blind" and self._check(env)):
                self._planning(env)
                return ACTION.PLANNING
            else:
                self.time = self.m-1
                self.step += 1
                return self.actions.pop(0)

    def _check(self, env):
        if env.map[self.target[0]][self.target[1]] != 2:
            return True
        if self.reaction_strategy == "any_hole":
            for hole in env.holes:
                hole = (hole.x, hole.y)
                if hole not in self.memory_hole:
                    return True
        if self.reaction_strategy == "nearer_hole":
            for hole in env.holes:
                hole = (hole.x, hole.y)
                if hole not in self.memory_hole and self._manhattan_distance(hole, (env.a_x, env.a_y)) < self._manhattan_distance(self.target, (env.a_x, env.a_y)):
                    return True
        return False

    def _manhattan_distance(self, x, y):
        return np.sum(np.abs(np.array(x)-np.array(y)))

    def add_score(self, score):
        self.score += score

    def _planning(self, env):
        if len(env.holes) == 0:
            return ACTION.PLANNING
        self.time = self.p-1
        self.step = 0
        self.actions, self.target = self._find_way(env)
        self.memory_hole = []
        for hole in env.holes:
            self.memory_hole.append((hole.x, hole.y))

    def _find_way(self, env):
        mmap = deepcopy(env.map)
        id2actions = [ACTION.UP, ACTION.DOWN, ACTION.RIGHT, ACTION.LEFT]

        holes = []

        h, w = len(mmap), len(mmap[0])

        class Node:
            def __init__(self, x, y, parent, dis, action) -> None:
                self.x = x
                self.y = y
                self.parent = parent
                self.dis = dis
                self.action = action
        node = Node(env.a_x, env.a_y, None, 0, None)
        queue = Queue()
        queue.put(node)
        while not queue.empty():
            node = queue.get()
            for i, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, 1), (0, -1)]):
                x, y = node.x+dx, node.y+dy
                if x >= 0 and x < h and y >= 0 and y < w and mmap[x][y] != 1:
                    new_node = Node(x, y, node, node.dis+1, id2actions[i])
                    queue.put(new_node)
                    if mmap[x][y] == 2:
                        holes.append(new_node)
                    mmap[x][y] = 1

        max_utility = -np.inf
        node = None
        for hole in holes:
            utility = self._utility(hole, env)
            if utility > max_utility:
                max_utility = utility
                node = hole
        target = (node.x, node.y)
        actions = []
        while node.x != env.a_x or node.y != env.a_y:
            actions.append(node.action)
            node = node.parent
        return actions[::-1], target

    def _utility(self, node, env):
        for hole in env.holes:
            if node.x == hole.x and node.y == hole.y:
                utility = self.args.d_coef*node.dis+self.args.s_coef * \
                    hole.score+self.args.a_coef*hole.age
                return utility

    def _born(self, env):

        x = random.randint(0, env.grid_l)
        y = random.randint(0, env.grid_l)
        while (x, y) in env.obstacles:
            x = random.randint(0, self.grid_l)
            y = random.randint(0, self.grid_l)
        env.a_x = x
        env.a_y = y
        env.map[x][y] = 3


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=[5, 2, 0], type=list)
    parser.add_argument("--p", default=1, type=float,
                        help="Clock cycle of the agent planning.")
    parser.add_argument("--k", default=4, type=int,
                        help="Agent reconsider its plan every k steps.")
    parser.add_argument("--m", default=1, type=int,
                        help="Clock cycle of the agent action.")
    parser.add_argument("--obstacles", default=[(0, 1), (3, 1), (8, 8), (8, 9), (7, 7), (
        4, 12), (13, 2), (10, 7), (14, 4), (7, 13)], type=list, help="positions of obstacles.")
    parser.add_argument("--l_min", default=240, type=int,
                        help="Minimum of the life-expectancy.")
    parser.add_argument("--l_max", default=960, type=int,
                        help="Maximum of the life-expectancy.")
    parser.add_argument("--g_min", default=60, type=int,
                        help="Minimum of the gestation.")
    parser.add_argument("--g_max", default=240, type=int,
                        help="Maximum of the gestation.")
    parser.add_argument("--s_min", default=1, type=int,
                        help="Minimum score of a hole.")
    parser.add_argument("--s_max", default=10, type=int,
                        help="Maximum score of a hole.")
    parser.add_argument("--d_coef", default=-1, type=int,
                        help="Coefficient of the distance when computing utility.")
    parser.add_argument("--a_coef", default=-1, type=int,
                        help="Coefficient of the age when computing utility.")
    parser.add_argument("--s_coef", default=5, type=int,
                        help="Coefficient of the score when computing utility.")
    parser.add_argument("--grid_l", default=15, type=int,
                        help="The length of the grid.")
    parser.add_argument("--gamma", default=1, type=int,
                        help="The Rate of changes of the world.")
    parser.add_argument("--iterations", default=3000, type=int,
                        help="The number of iterations of the simulation.")
    parser.add_argument("--is_vis", action="store_false",
                        help="Is visualize the process?")
    parser.add_argument("--reaction_strategy", default="blind", type=str)
    args = parser.parse_args()

    args = reset_args(args)
    return args


def reset_args(args):
    args.l_min = int(args.l_min/args.gamma)
    args.l_max = int(args.l_max/args.gamma)
    args.g_min = int(args.g_min/args.gamma)
    args.g_max = int(args.g_max/args.gamma)
    return args


def run(args):
    epsilons = []
    for seed in args.seed:
        random.seed(seed)
        env = TileWorld(args)
        agent = Agent(args, env)

        if args.is_vis:
            env.render_init()

        bar = tqdm(range(args.iterations))
        for iteration in bar:
            action = agent.update(env)
            score = env.update(action)
            agent.add_score(score)
            bar.set_description(
                "Processing iters: {}/{}".format(iteration, args.iterations))
            if args.is_vis:
                env.render()
        if args.is_vis:
            env.render_off()
        print("env score: ", env.total_score)
        print("agent score: ", agent.score)
        epsilons.append(agent.score/env.total_score)
    return np.mean(epsilons)


def test1():
    args = init_args()
    args.is_vis = False
    epsilons = []
    log10gammas = np.linspace(0, 2, 15, endpoint=True)
    gammas = 10**log10gammas
    for gamma in gammas:
        test_args = copy.deepcopy(args)
        test_args.gamma = gamma
        test_args = reset_args(test_args)
        epsilons.append(run(test_args))
    plt.plot(gammas, epsilons, linestyle='-', marker='D',
             markersize=markerss, linewidth=linew, color='r', label='experimental')

    plt.legend(fancybox=True, framealpha=0, fontsize=17, handletextpad=0.1,
               handlelength=1, columnspacing=0.5, markerscale=1.3, labelspacing=0.3)
    ax = plt.subplot(1, 1, 1)
    plt.ylabel(r'$\epsilon$', fontsize=24)
    plt.xlabel(r'$\gamma$', fontsize=24)
    plt.ylim([0, 1.05])
    plt.grid()
    plt.tight_layout()
    plt.savefig('test1-1.png', bbox_inches='tight', dpi=150)
    plt.close()

    plt.plot(gammas, epsilons, linestyle='-', marker='D',
             markersize=markerss, linewidth=linew, color='r', label='experimental')
    plt.legend(fancybox=True, framealpha=0, fontsize=17, handletextpad=0.1,
               handlelength=1, columnspacing=0.5, markerscale=1.3, labelspacing=0.3)
    ax = plt.subplot(1, 1, 1)
    plt.ylabel(r'$\epsilon$', fontsize=24)
    plt.xlabel(r'$\log_{10}\gamma$', fontsize=24)
    plt.ylim([0, 1.05])
    plt.grid()
    plt.tight_layout()
    plt.savefig('test1-2.png', bbox_inches='tight', dpi=150)
    plt.close()


def test2():
    args = init_args()
    args.is_vis = False
    log10gammas = np.linspace(0, 2, 15, endpoint=True)
    gammas = 10**log10gammas
    args.k = 2*args.grid_l
    epsilons = []
    for p in [1, 2, 4, 10]:
        epsilons.append([])
        args.p = p
        for gamma in gammas:
            test_args = copy.deepcopy(args)
            test_args.gamma = gamma
            test_args = reset_args(test_args)
            epsilons[-1].append(run(test_args))

    plt.plot(log10gammas, epsilons[0], linestyle='-', marker='D',
             markersize=markerss, linewidth=linew, color='r', label='p=1')
    plt.plot(log10gammas, epsilons[1], linestyle='--', marker='*',
             markersize=markerss, linewidth=linew, color='b', label='p=2')
    plt.plot(log10gammas, epsilons[2], linestyle='-', marker='x',
             markersize=markerss, linewidth=linew, color='g', label='p=4')
    plt.plot(log10gammas, epsilons[3], linestyle='--', marker='v',
             markersize=markerss, linewidth=linew, color='gray', label='p=10')

    plt.legend(fancybox=True, framealpha=0, fontsize=17, handletextpad=0.1,
               handlelength=1, columnspacing=0.5, markerscale=1.3, labelspacing=0.3)
    ax = plt.subplot(1, 1, 1)
    plt.ylabel(r'$\epsilon$', fontsize=24)
    plt.xlabel(r'$\log_{10}\gamma$', fontsize=24)
    plt.ylim([0, 1.05])
    plt.grid()
    plt.tight_layout()
    plt.savefig('test2-1.png', bbox_inches='tight', dpi=150)
    plt.close()

    args.k = 1
    epsilons = []
    for p in [1, 2, 4, 10]:
        epsilons.append([])
        args.p = p
        for gamma in gammas:
            test_args = copy.deepcopy(args)
            test_args.gamma = gamma
            test_args = reset_args(test_args)
            epsilons[-1].append(run(test_args))
    plt.plot(log10gammas, epsilons[0], linestyle='-', marker='D',
             markersize=markerss, linewidth=linew, color='r', label='p=1')
    plt.plot(log10gammas, epsilons[1], linestyle='--', marker='*',
             markersize=markerss, linewidth=linew, color='b', label='p=2')
    plt.plot(log10gammas, epsilons[2], linestyle='-', marker='x',
             markersize=markerss, linewidth=linew, color='g', label='p=4')
    plt.plot(log10gammas, epsilons[3], linestyle='--', marker='v',
             markersize=markerss, linewidth=linew, color='gray', label='p=10')

    plt.legend(fancybox=True, framealpha=0, fontsize=17, handletextpad=0.1,
               handlelength=1, columnspacing=0.5, markerscale=1.3, labelspacing=0.3)
    ax = plt.subplot(1, 1, 1)
    plt.ylabel(r'$\epsilon$', fontsize=24)
    plt.xlabel(r'$\log_{10}\gamma$', fontsize=24)
    plt.ylim([0, 1.05])
    plt.grid()
    plt.tight_layout()
    plt.savefig('test2-2.png', bbox_inches='tight', dpi=150)
    plt.close()


def test3():
    args = init_args()
    args.is_vis = False
    log10gammas = np.linspace(0, 2, 15, endpoint=True)
    gammas = 10**log10gammas

    for i, p in enumerate([4, 2, 1]):
        args.p = p
        epsilons = []
        for k in [1, 4, 5*args.grid_l]:
            epsilons.append([])
            args.k = k
            for gamma in gammas:
                test_args = copy.deepcopy(args)
                test_args.gamma = gamma
                test_args = reset_args(test_args)
                epsilons[-1].append(run(test_args))

        plt.plot(log10gammas, epsilons[0], linestyle='-', marker='D',
                 markersize=markerss, linewidth=linew, color='r', label='cautious')
        plt.plot(log10gammas, epsilons[1], linestyle='--', marker='*',
                 markersize=markerss, linewidth=linew, color='b', label='normal')
        plt.plot(log10gammas, epsilons[2], linestyle='-', marker='x',
                 markersize=markerss, linewidth=linew, color='gray', label='bold')
        plt.legend(fancybox=True, framealpha=0, fontsize=17, handletextpad=0.1,
                   handlelength=1, columnspacing=0.5, markerscale=1.3, labelspacing=0.3)
        ax = plt.subplot(1, 1, 1)
        plt.ylabel(r'$\epsilon$', fontsize=24)
        plt.xlabel(r'$\log_{10}\gamma$', fontsize=24)
        plt.ylim([0, 1.05])
        plt.grid()
        plt.tight_layout()
        plt.savefig('test3-{}.png'.format(i+1), bbox_inches='tight', dpi=150)
        plt.close()


def test4():
    args = init_args()
    args.is_vis = False
    args.k = 5*args.grid_l
    log10gammas = np.linspace(0, 2, 15, endpoint=True)
    gammas = 10**log10gammas
    for i, p in enumerate([2, 1]):
        args.p = p
        epsilons = []
        for reaction_strategy in ["blind", "disapper", "any_hole", "nearer_hole"]:
            epsilons.append([])
            args.reaction_strategy = reaction_strategy
            for gamma in gammas:
                test_args = copy.deepcopy(args)
                test_args.gamma = gamma
                test_args = reset_args(test_args)
                epsilons[-1].append(run(test_args))

        plt.plot(log10gammas, epsilons[0], linestyle='-', marker='D',
                 markersize=markerss, linewidth=linew, color='r', label='blind')
        plt.plot(log10gammas, epsilons[1], linestyle='--', marker='*',
                 markersize=markerss, linewidth=linew, color='b', label='notices target disappearance')
        plt.plot(log10gammas, epsilons[2], linestyle='-', marker='x',
                 markersize=markerss, linewidth=linew, color='g', label='target dis. or any new hole')
        plt.plot(log10gammas, epsilons[3], linestyle='--', marker='v',
                 markersize=markerss, linewidth=linew, color='gray', label='target dis. or near hole')

        plt.legend(fancybox=True, framealpha=0, fontsize=17, handletextpad=0.1,
                   handlelength=1, columnspacing=0.5, markerscale=1.3, labelspacing=0.3)
        ax = plt.subplot(1, 1, 1)
        plt.ylabel(r'$\epsilon$', fontsize=24)
        plt.xlabel(r'$\log_{10}\gamma$', fontsize=24)
        plt.ylim([0, 1.05])
        plt.grid()
        plt.tight_layout()
        plt.savefig('test4-{}.png'.format(i+1), bbox_inches='tight', dpi=150)
        plt.close()

    args.p = 1
    args.reaction_strategy = "disapper"
    epsilons = []
    for k in [1, 4, 5*args.grid_l]:
        epsilons.append([])
        args.k = k
        for gamma in gammas:
            test_args = copy.deepcopy(args)
            test_args.gamma = gamma
            test_args = reset_args(test_args)
            epsilons[-1].append(run(test_args))
    plt.plot(log10gammas, epsilons[0], linestyle='-', marker='D',
             markersize=markerss, linewidth=linew, color='r', label='cautious')
    plt.plot(log10gammas, epsilons[1], linestyle='--', marker='*',
             markersize=markerss, linewidth=linew, color='b', label='normal')
    plt.plot(log10gammas, epsilons[2], linestyle='-', marker='x',
             markersize=markerss, linewidth=linew, color='gray', label='bold')
    plt.legend(fancybox=True, framealpha=0, fontsize=17, handletextpad=0.1,
               handlelength=1, columnspacing=0.5, markerscale=1.3, labelspacing=0.3)
    ax = plt.subplot(1, 1, 1)
    plt.ylabel(r'$\epsilon$', fontsize=24)
    plt.xlabel(r'$\log_{10}\gamma$', fontsize=24)
    plt.ylim([0, 1.05])
    plt.grid()
    plt.tight_layout()
    plt.savefig('test4-3.png', bbox_inches='tight', dpi=150)
    plt.close()


if __name__ == "__main__":
    args = init_args()
    run(args)
    test1()
    test2()
    test3()
    test4()
