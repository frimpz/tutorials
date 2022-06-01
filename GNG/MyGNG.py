'''
This class represents the neural gas class
'''
import networkx as nx
import numpy as np


def get_random_sample(max_index, sizes):
    indices = np.random.choice(max_index, sizes)
    return indices


def get_point(dist, max_index):
    if dist == 'random':
        return np.random.choice(max_index, 1)
    elif dist == 'shuffle':
        x = (np.arange(max_index))
        np.random.shuffle(x)
        return x


class MyGrowingNeuralGas:

    def __init__(self, **kwargs):
        self.graph = None
        self.num_of_nodes = 0
        self.probability = kwargs['probability'] if 'probability' in kwargs else 'random'
        self.max_units = kwargs['max_units'] if 'max_units' in kwargs else 1000
        self.eb = kwargs['eb'] if 'eb' in kwargs else 0.2
        self.en = kwargs['en'] if 'en' in kwargs else 0.005
        self.max_age = kwargs['max_age'] if 'max_age' in kwargs else 50
        self.lmda = kwargs['lambda'] if 'lambda' in kwargs else 100
        self.iterations = 0
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.5
        self.d = kwargs['d'] if 'd' in kwargs else 0.995

    def instantiate(self, data):
        self.graph = nx.Graph()
        # step 0 - Start with two units a and b at random positions,
        # w_a and w_b in R^n
        a, b = get_random_sample(len(data), 2)

        self.graph.add_node(self.num_of_nodes, weight=np.array(data[a, :], dtype=np.float), error=0)
        self.num_of_nodes += 1
        self.graph.add_node(self.num_of_nodes, weight=np.array(data[b, :], dtype=np.float), error=0)
        self.num_of_nodes += 1
        # self.graph.add_edge(0, 1)

    def train(self, data):
        # 1.
        input_signal = get_point(self.probability, len(data))
        for cnt, cur in enumerate(input_signal):

            if self.graph is None:
                self.instantiate(data)

            self.iterations += 1

            # 1.
            selected_sample = data[cur, :]

            # 2.
            distance = self.get_nearest_units(selected_sample)
            s_1 = distance[0][0]
            s_2 = distance[1][0]

            # 3.
            for edge in self.graph.edges(s_1, data=True):
                edge[2]['age'] += 1

            # 4.
            self.graph.nodes[s_1]['error'] += distance[0][1]

            # 5.
            self.graph.nodes[s_1]['weight'] += self.eb * (selected_sample - self.graph.nodes[s_1]['weight'])
            # update neighbours
            for neighbor in self.graph.neighbors(s_1):
                self.graph.nodes[neighbor]['weight'] += self.en * (selected_sample - self.graph.nodes[neighbor]['weight'])

            # 6.
            if self.graph.has_edge(s_1, s_2):
                self.graph[s_1][s_2]['age'] = 0
            else:
                self.graph.add_edge(s_1, s_2, age=0)

            self.graph.remove_edges_from(self.gets_edges_to_remove())
            self.graph.remove_nodes_from(list(nx.isolates(self.graph)))

            # 8.
            if self.iterations % self.lmda == 0 and len(self.graph.nodes) < self.max_units:
                # [(0, array([ 91., -27.]), 0)]
                q = sorted(self.graph.nodes(data=True), key=lambda x: x[1]['error'], reverse=True)[0]
                f = self.graph.neighbors(q[0])
                f = [(i, self.graph.nodes[i]['weight'], self.graph.nodes[i]['error']) for i in f]
                # (1, {'weight': array([  25.4, -128.4]), 'error': 43.56604182158393})
                f = sorted(f, key=lambda x: x[2], reverse=True)
                # (0, array([ 91., -27.]), 0)
                pos_r = 0.5 * (q[1]['weight'] + f[0][1])
                self.graph.add_node(self.num_of_nodes, weight=np.array(pos_r, dtype=np.float), error=0)
                self.graph.add_edge(q[0], self.num_of_nodes, age=0)
                self.graph.add_edge(self.num_of_nodes, f[0][0], age=0)
                self.graph.remove_edge(q[0], f[0][0])
                self.graph.nodes[q[0]]['error'] *= self.alpha
                self.graph.nodes[f[0][0]]['error'] *= self.alpha
                self.graph.nodes[self.num_of_nodes]['error'] = self.graph.nodes[q[0]]['error']
                self.num_of_nodes += 1

            # 9.
            for node in self.graph.nodes(data=True):
                self.graph.nodes[node[0]]['error'] *= self.d

    def get_nearest_units(self, current_point):
        all_points = [i for i in self.graph.nodes(data=True)]
        indicies = [i[0] for i in all_points]
        positions = [i[1]['weight'] for i in all_points]
        # np.linalg.norm(i[1] - current_point, axis=1)
        distance = sorted([(i[0], np.linalg.norm(i[1] - current_point)) for i in zip(indicies, positions)], key=lambda x: x[1])
        # print(nm)
        # #print(list(zip(indicies, positions)))
        # exit()
        # unit_weights = np.vstack(np.array([i for i in positions], dtype=np.float))
        # distance = np.linalg.norm(unit_weights - current_point, axis=1)
        # distance = sorted(list(zip(indicies, distance)), key=lambda x: x[1])
        return distance

    def gets_edges_to_remove(self):
        edges_to_remove = []
        for edge in self.graph.edges(data=True):
            if edge[2]['age'] >= self.max_age:
                edges_to_remove.append(edge)
        return edges_to_remove
