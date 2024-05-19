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
from agents import GreedyAgent1, RandomAgent, DQNAgent_offline
from custom_data_structures import Heap
from environments import State, CSP_Environment
from tools import generate_graph, visualise_graph, build_offline_dataset, compute_td_target, visualize_loss, print_shortest_path_source_to_all_nodes
from graph_classes import Node, Neighbour, Demand, Arc, Graph

#importing data
csp_file = open('exercise_bonus.json')
data_2 = json.load(csp_file)

graph = Graph(len(data_2["nodes"]))
for edge in data_2["edges"]:
    graph.addEdge(edge["from"], edge["to"], edge["transit_time"], edge["capacity"])

graph.add_supply(data_2["demands"])

##visualise_graph(data_2, size =(14, 8))

print("\nGenerating Routes for given demand using an MILP")
graph.generate_routes_csp(show_routes = True, export_output = True)
print("\n\n\nGenerating Routes for given demand using a greedy sequential demand allocation")
graph.calculate_flow_greedy(show_output = True, export_output = True)
