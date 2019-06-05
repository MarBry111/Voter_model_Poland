import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import IPython.display as dis
import matplotlib.animation as animation
from PIL import Image

import os
import pandas as pd
import seaborn as sns

import geopandas as gpd
import libpysal
import pysal
# from libpysal.weights.contiguity import Queen
from libpysal.weights import Queen, Rook, KNN, Kernel
from splot.libpysal import plot_spatial_weights

from shapely.ops import cascaded_union


class Graph:
    def __init__(self, G):
        self.nodes = np.array(G.nodes())
        edges_arr = list(G.edges())
        self.edges = self.extract_edges(self.nodes, edges_arr, G)
        self.opinions = np.random.randint(2, size=self.nodes.size)
        # self.level = self.edges[0].shape[-1]
        self.voting_prefferences = np.ones((self.nodes.size, 2))
        # z zeros na ones
        #self.voting_prefferences[:, self.opinions] = 1

    def extract_edges(self, nodes, edges, g):
        d = dict.fromkeys(np.array(g.nodes()), [])
        for e in edges:
            d[e[0]] = d[e[0]] + [e[1]]
        for e in edges:
            d[e[1]] = d[e[1]] + [e[0]]
        for key in d.keys():
            d[key] = np.array(d[key])
        return d

    def one_step(self, alpha, noise=0):
        temp_opinions = np.copy(self.opinions)
        for node in self.nodes:
            n0 = 0
            n1 = 0
            n = len(self.edges[node])
            for neighbour in self.edges[node]:
                if self.opinions[neighbour] == 1:
                    n1 += 1
                elif self.opinions[neighbour] == 0:
                    n0 += 1

            if n1 > n0:
                temp_opinions[node] = 1
            elif n0 > n1:
                temp_opinions[node] = 0

            a0, a1 = 0, 0
            if np.random.random_sample() < noise:
                if temp_opinions[node] == 0:
                    a1 = 1
                else:
                    a0 = 1

            if temp_opinions[node] == 0:
                self.voting_prefferences[node, 0] += (n0 + 1 + a0) / n
                self.voting_prefferences[node, 1] += (n1 + a1) / n
            else:
                self.voting_prefferences[node, 1] += (n1 + 1 + a1) / n
                self.voting_prefferences[node, 0] += (n0 + a0) / n

            # print('node:',node,' n0,n1: ',n0,n1, ' opinion: ',self.opinions[node],'->'  ,temp_opinions[node])
        self.voting_prefferences = self.voting_prefferences * alpha.reshape(13, 2)

    def one_move(self):
        self.opinions = np.argmax(self.voting_prefferences, axis=1)
        self.voting_prefferences = np.zeros((self.nodes.size, 2))
        self.voting_prefferences[:, self.opinions] = 1

    def parties(self):
        op1 = np.where(self.opinions == 1)
        op0 = np.where(self.opinions == 0)
        party1 = self.nodes[op1]
        party0 = self.nodes[op0]
        return list(op0[0]), list(op1[0])


euro = gpd.read_file("euro/euro.shp")

weuro = Queen.from_dataframe(euro)
pos_e = {}

woj = euro
woj['centroid'] = woj['geometry'].centroid
for point in woj.iterrows():
    pos_e[point[0]] = [point[1]['centroid'].x, point[1]['centroid'].y]

contig_matrix = weuro

# build list of edges - this will create edges going both ways from connected nodes, so you might need to remove duplicates
nodes = contig_matrix.weights.keys()  # to get dict of keys, alternatively use contig_matrix.id2i.keys()
edges = [(node, neighbour) for node in nodes for neighbour in contig_matrix[node]]
graph_euro = nx.Graph(edges)

pos_e[4] = [pos_e[4][0] + 50000, pos_e[4][1]]

graph_euro.add_edge(3, 4)
graph_euro.add_edge(12, 0)
graph_euro.add_edge(6, 0)

labels_e = {}
for i in range(len(graph_euro)):
    labels_e[i] = i + 1

gE = Graph(graph_euro)

op2009 = np.ones(len(gE.opinions))
op2009[[4, 7, 8, 9]] = 0

op2014 = np.ones(len(gE.opinions))
op2014[[2, 4, 5, 7, 8, 9]] = 0

op2019 = np.ones(len(gE.opinions))
op2019[[2, 4, 5, 7, 8, 9, 10]] = 0

opinion_arr = [op2009, op2014, op2019]

gE.opinions = np.ones(len(gE.opinions))
gE.opinions[[3, 4, 7, 8]] = 0


n_agents = 300

vgE = np.empty((n_agents,), dtype=object)
parameter1 = np.random.random_sample((n_agents, 2 * gE.nodes.size))
#parameter1[:] = np.fromstring('0.023756457765446656 0.5633382126875802 0.33915336654728223 0.4115316630264049 0.9804097686282733 0.4762462342128407 0.1072024758178421 0.6868474902259009 0.26094592158458946 0.42699388565653906 0.5993330027151821 0.5196467440915867 0.3500708025742209 0.8420836817670585 0.6054322048843772 0.4863156358404752 0.8321364818399454 0.6567315806753641 0.893621800800366 0.4561245458193595 0.7885925922984101 0.3443805220522544 0.30845211165595765 0.4155645653073518 0.6881527326713318 0.8261937413856438 ', sep=' ')

iteration = 0
cost_max = 0
last_max = 0
mnoznik = 1.00
dodawanie = True
mnoznik_change = 0.05

for iteration in range(100):
    # while cost_max < 39:
    print('iteration: ', iteration)
    cost_funct = np.zeros((n_agents))
    # 1 po
    # 0 pis
    vgE[:] = gE

    for j in range(15):
        for agent in range(n_agents):
            vgE[agent].one_step(parameter1[agent], 0.05)
            # vgE[agent].one_move()
            if (j + 1) % 5 == 0:
                vgE[agent].one_move()
                cost_funct[agent] += np.sum(vgE[agent].opinions == opinion_arr[(j + 1) % 5 - 1])

    cost_max = np.max(cost_funct)
    print('Max cost:', cost_max, '  mnoznik: ', mnoznik)

    if cost_max == 39:
        print(cost_max)
        with open('polsza/best_alpha', "a+") as f:
            f.write("Cost, after move every time frame: " + str(cost_max))
            f.write("\n")
            for par in parameter1[np.argwhere(cost_funct == np.max(cost_funct))]:
                for arr in par:
                    for a in arr:
                        f.write(str(a))
                        f.write(' ')
                    f.write("\n")
            f.write("\n\n")

    cost_avg = (np.max(cost_funct) + np.min(cost_funct)) / 2
    best_alpha_index = np.argwhere(cost_funct >= cost_avg)
    worst_alpha_index = np.argwhere(cost_funct < cost_avg)

    if cost_max > last_max and iteration > 0:
        if dodawanie:
            mnoznik += mnoznik_change
        else:
            mnoznik -= mnoznik_change
            dodawanie = False
    elif cost_max < last_max and iteration > 0:
        if dodawanie:
            mnoznik -= mnoznik_change
        else:
            mnoznik += mnoznik_change
            dodawanie = True

    last_max = cost_max
    for child in worst_alpha_index:
        parents = np.random.choice(len(best_alpha_index), 2)
        parameter1[child] = ((parameter1[best_alpha_index[parents[0]]] + parameter1[best_alpha_index[parents[1]]]) / 2) * mnoznik
        if np.random.random_sample() < 0.05:
            parameter1[child] = np.random.random_sample()

    # iteration += 1


cost_max = np.max(cost_funct)
print(cost_max)
# print(np.argwhere(cost_funct == np.max(cost_funct)))
# print(parameter1[np.argwhere(cost_funct == np.max(cost_funct))])
