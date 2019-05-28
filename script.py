import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import IPython.display as dis
import matplotlib.animation as animation
from PIL import Image

import os

from numba import jit


class Graph:
    def __init__(self, G=None):
        self.nodes = np.array(G.nodes())
        edges_arr = list(G.edges())
        self.edges = self.extract_edges(self.nodes, edges_arr, G)
        self.opinions = np.random.randint(2, size=self.nodes.shape)
        self.level = self.edges[0].shape[-1]

    def extract_edges(self, nodes=None, edges=None, G=None):
        d = dict.fromkeys(np.array(G.nodes()), [])
        for e in edges:
            d[e[0]] = d[e[0]] + [e[1]]
        for e in edges:
            d[e[1]] = d[e[1]] + [e[0]]
        for key in d.keys():
            d[key] = np.array(d[key])
        return d

    @jit
    def one_step(self):
        temp_opinions = np.copy(self.opinions)
        for node in self.nodes:
            n0 = 0
            n1 = 0
            for neighbour in self.edges[node]:
                if self.opinions[neighbour] == 1:
                    n1 += 1
                elif self.opinions[neighbour] == 0:
                    n0 += 1
            if n1 > n0:
                temp_opinions[node] = 1
            elif n0 > n1:
                temp_opinions[node] = 0
            #print('node:',node,' n0,n1: ',n0,n1, ' opinion: ',self.opinions[node],'->'  ,temp_opinions[node])
        self.opinions = temp_opinions

    def parties(self):
        op1 = np.where(self.opinions == 1)
        op0 = np.where(self.opinions == 0)
        party1 = self.nodes[op1]
        party0 = self.nodes[op0]
        return list(op0[0]), list(op1[0])


def main():
    for v_G in [2, 3, 4, 5, 6, 7, 8]:
        #os.mkdir('graph/vG' + str(v_G))

        for it in range(100):
            print('vG: ' + str(v_G) + ' iteration: ' + str(it))
            # Network topology
            #g = nx.erdos_renyi_graph(100, 0.1)
            N = 100
            g = None
            g = nx.random_regular_graph(v_G, N)
            path = 'graph/vG' + str(v_G) + '/it' + str(it)

            # os.mkdir(path)

            gv = Graph(g)
            pos = nx.spring_layout(g)

            plt.clf()

            voters0 = []
            voters1 = []

            j = 0
            # for j in range(10):
            while np.sum(gv.opinions) != N and np.sum(gv.opinions) != 0 and j < 15:
                voters0.append(np.sum(gv.opinions) / N)
                voters1.append(1 - np.sum(gv.opinions) / N)
                '''
                party0, party1 = gv.parties()
                plt.figure(figsize=(12, 12))
                plt.title("Step " + str(j))
                nx.draw_networkx_edges(g, pos)
                nx.draw_networkx_nodes(g, pos, nodelist=party0, node_color='r')
                nx.draw_networkx_nodes(g, pos, nodelist=party1, node_color='b')
                # nx.draw_networkx_labels(g,pos,labels,font_size=16)
                plt.savefig(path + "/step_" + str(j) + ".png")
                '''
                gv.one_step()
                j += 1

            voters0.append(np.sum(gv.opinions) / N)
            voters1.append(1 - np.sum(gv.opinions) / N)
            '''
            party0, party1 = gv.parties()
            plt.figure(figsize=(12, 12))
            plt.title("Step " + str(j))
            nx.draw_networkx_edges(g, pos)
            nx.draw_networkx_nodes(g, pos, nodelist=party0, node_color='r')
            nx.draw_networkx_nodes(g, pos, nodelist=party1, node_color='b')
            # nx.draw_networkx_labels(g,pos,labels,font_size=16)
            plt.savefig(path + "/step_" + str(j) + ".png")

            plt.figure(figsize=(10, 10))
            plt.rcParams.update({'font.size': 18})
            plt.plot(voters1)
            plt.plot(voters0)

            plt.suptitle('Stosunek liczby popleczników danej partii \n do liczby wszystkich agentów \n dla N=' + str(N) + ' oraz V(G)=' + str(v_G))
            plt.ylim(top=1, bottom=0)
            plt.xlabel("Krok czasowy")

            #yint = range(j + 1)
            # plt.xticks(yint)
            plt.ylabel(r"Stosunek agentów danej partii $\frac{n_i}{N}$")
            plt.savefig(path + '/noverN_vG' + str(v_G) + '.png')
            '''
            with open(path[:9] + '/vnoverN_vG' + str(v_G), "a+") as f:
                if(voters0[-1]>voters1[-1]):
                    for v0 in voters0:
                        f.write(str(v0) + ' ')
                else:
                    for v0 in voters1:
                        f.write(str(v0) + ' ')
                f.write("\n")
            '''
            with open(path[:9] + '/v1noverN_vG' + str(v_G), "a+") as f:
                for v1 in voters1:
                    f.write(str(v1) + ' ')
                f.write("\n")
            '''


main()
