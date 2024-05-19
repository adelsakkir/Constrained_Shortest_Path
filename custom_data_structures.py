import json
from collections import defaultdict
import sys
import random
from collections import deque
import numpy as np
from ortools.graph.python import min_cost_flow
from decimal import *
from ortools.linear_solver import pywraplp
from sklearn.neural_network import MLPRegressor
import pandas as pd
import pickle
from datetime import datetime
import string
import networkx as nx
import matplotlib.pyplot as plt
from collections import ChainMap
import matplotlib.pyplot as plt
from IPython import display
import copy

# importing self built classes and functions
##from agents import GreedyAgent1, RandomAgent, DQNAgent_offline
##from environments import State, CSP_Environment
##from tools import generate_graph, visualise_graph, build_offline_dataset, compute_td_target, visualize_loss, print_shortest_path_source_to_all_nodes
##from graph_classes import Node, Neighbour, Demand, Arc, Graph

class Heap:

    def __init__(self):
        self.array = []
        self.size = 0

    def buildMinHeap(self):
        for index in range((len(self.array)//2)+1,-1,-1):
            self.minHeapify(index)

    def swapMinHeapNode(self, a, b):
        self.array[a].pos = b
        self.array[b].pos = a
        self.array[a], self.array[b] = self.array[b], self.array[a]

    def minHeapify(self, index):
        smallest = index
        left = 2*index + 1
        right = 2*index + 2

        for child_index in [left, right]:
            if child_index < self.size and self.array[child_index].dist < self.array[smallest].dist:
                smallest = child_index

        if smallest != index:

            # Swap nodes
            self.swapMinHeapNode(smallest, index)
            self.minHeapify(smallest)


    def extractMin(self):

        if self.isEmpty() == True:
            return

        self.swapMinHeapNode(0, self.size - 1)

        # Reduce heap size and heapify root
        self.size -= 1
        root = self.array.pop()
        root.PERM = True
        self.minHeapify(0)

        return root

    def isEmpty(self):
        return True if self.size == 0 else False


    def updateDist(self, node, dist):

        i = node.pos
        node.dist = dist

        # Travel upto root - O(Logn)
        while (i > 0 and self.array[i].dist < self.array[(i - 1) // 2].dist):

            self.swapMinHeapNode(i, (i - 1)//2)
            i = (i - 1) // 2;

    def Print(self):
        for i in range(0, self.size): #(self.size//2)+3
            if 2 * i + 1 >= self.size:
                print(" PARENT : "+ str(self.array[i].id)+ ", "+ str(self.array[i].pos)+ ", " + str(self.array[i].dist)+
                  " LEFT CHILD : -- " +
                  " RIGHT CHILD : -- ")
            elif 2 * i + 2 >= self.size:
                print(" PARENT : "+ str(self.array[i].id)+ ", "+ str(self.array[i].pos)+ ", " + str(self.array[i].dist)+
                  " LEFT CHILD : "+ str(self.array[2 * i + 1].id)+ ", " + str(self.array[2 * i + 1].dist)+
                  " RIGHT CHILD : --")
            else:
                print(" PARENT : "+ str(self.array[i].id)+ ", "+ str(self.array[i].pos)+ ", " + str(self.array[i].dist)+
                      " LEFT CHILD : "+ str(self.array[2 * i + 1].id)+ ", " + str(self.array[2 * i + 1].dist)+
                      " RIGHT CHILD : "+ str(self.array[2 * i + 2].id)+ ", " + str(self.array[2 * i + 2].dist))

