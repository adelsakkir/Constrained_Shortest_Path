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
csp_file = open('exercise_bonus.json')
data_2 = json.load(csp_file)

graph = Graph(len(data_2["nodes"]))
for edge in data_2["edges"]:
    graph.addEdge(edge["from"], edge["to"], edge["transit_time"], edge["capacity"])

graph.add_supply(data_2["demands"])

##visualise_graph(data_2, size =(14, 8))

print("\nGenerating Routes for given demand using an Greedy Agent through the RL enviroment")
graph.rl_greedy(show_output = True, export_output = True)
print("\n\n\nGenerating Routes for given demand using a offline Reinforcement Learning Agent")
offline_dataset = build_offline_dataset(graph, iterations = 100, action_agent = RandomAgent())
graph.rl_offline(offline_data = offline_dataset, fittedQ_iterations = 20, show_output = True, export_output = True)


val = input("""\nOn the csv file named 'results_rl.csv' you find the the total transit time of the routes found by the offline DQN RL agent in 10 experiments. The first row shows the total transit time of the greedy agent.
            Would you like to re-run this experiment? - """)
if val in ["yes","y","Yes"]:
    columns = ["Experiment", "Agent", "Total transit time"]
    results_df = pd.DataFrame(columns=columns)
    i=1
    results_df.loc[len(results_df.index)] = [i, "Greedy", graph.rl_greedy(show_output = False, export_output = False)]
    print("\n")
    i+=1
    while i<=10:
        offline_dataset = build_offline_dataset(graph, iterations = 100, action_agent = RandomAgent())
        results_df.loc[len(results_df.index)] = [i, "Offline DQN", graph.rl_offline(offline_data = offline_dataset, fittedQ_iterations = 20, show_output = False, export_output = False)]
        i+=1
        print("\n")
        print(results_df)
    results_df.to_csv('results_rl.csv', index=False)  
else:
    print("\nThank you!")

