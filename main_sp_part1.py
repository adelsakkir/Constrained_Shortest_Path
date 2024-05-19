import json
import copy
import sys
import pickle
import string
import random
import numpy as np
import pandas as pd
import networkx as nx
from decimal import *
from IPython import display
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
from collections import ChainMap
from collections import defaultdict
from ortools.linear_solver import pywraplp
from ortools.graph.python import min_cost_flow
from sklearn.neural_network import MLPRegressor

# importing self built classes and functions
from agents import GreedyAgent1, RandomAgent, DQNAgent_offline
from custom_data_structures import Heap
from environments import State, CSP_Environment
from tools import generate_graph, visualise_graph, build_offline_dataset, compute_td_target, visualize_loss, print_shortest_path_source_to_all_nodes
from graph_classes import Node, Neighbour, Demand, Arc, Graph

#importing data
sp_file = open('exercise_baseline.json')
data_1 = json.load(sp_file)

graph = Graph(len(data_1["nodes"]))
for edge in data_1["edges"]:
    graph.addEdge(edge["from"], edge["to"], edge["transit_time"])

visualise_graph(data_1, size =(14, 8))
# Running Shortest paths Algorithms
print("\nPlease find below the shortest paths from a given source to all nodes")
graph.shortest_path_source_to_all_nodes(src = 'co', export_output = True, show_output = True)
print("\nPlease find below the shortest paths from a given source to given destination")
graph.shortest_path_source_to_dest(src = 'co', dest = 'wr', export_output = True, show_output = True)
print("\nPlease find below the shortest paths from all nodes to a given destination")
graph.shortest_path_dest_from_all_nodes(dest = 'co', export_output = True, show_output = True)
graph.shortest_path_all_pairs(export_output = True, show_output = False)


#Testing on Randomly Generated graphs
Tests = [(6, 7), (8,15), (12, 40), (10,8)]
i= 1

flag = True
while flag:
    val = int(input("""We will now run the shortest path algorithm on random Graph instances.
    Please choose from the below options -
                1. Vertices = 6, Edges = 12
                2. Vertices = 8, Edges =15
                3. Vertices = 12, Edges = 40
                4. Vertices = 10, Edges =8
                5. Custom Input
                 - """))

    if val == 5:
        vertices = int(input("Enter vertices - "))
        edges = int(input("Enter Edges - "))
    else:
        vertices = Tests[val-1][0]
        edges = Tests[val-1][1]
        
    random.seed(123)
    print("\nRandomly Generated graph instance - ", i)
    i+=1
    print("Vertices - ", vertices, ", Edges - ", edges)
    new_graph = generate_graph(vertices = vertices, edges = edges, show_graph_visual =True, size = (6,6), export_output= True)
    if new_graph:
        graph = Graph(len(new_graph["nodes"]))
        for edge in new_graph["edges"]:
            # print(edge)
            graph.addEdge(edge["from"], edge["to"], edge["transit_time"])
        
        print("\nShortest Path from source to all nodes")
        graph.shortest_path_source_to_all_nodes(src = 'ni', show_output = True, export_output = False)
        print("\nShortest Path from source to destination")
        graph.shortest_path_source_to_dest(src = 'ni', dest = 'kc', show_output = True, export_output = False)
        print("\nShortest Path from all nodes to destination")
        graph.shortest_path_dest_from_all_nodes(dest = 'kc', show_output = True, export_output = False)
        graph.shortest_path_all_pairs(export_output = True, show_output = False)

    check = input("Would you like to try another instance? (yes/no)- ")
    if check in ["yes", "Yes", "y"]:
        flag = True
    else:
        flag = False
