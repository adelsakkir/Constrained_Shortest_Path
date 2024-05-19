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
