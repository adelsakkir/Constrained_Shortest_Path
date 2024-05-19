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
##from custom_data_structures import Heap
from environments import State, CSP_Environment
##from tools import generate_graph, visualise_graph, build_offline_dataset, compute_td_target, visualize_loss, print_shortest_path_source_to_all_nodes
##from graph_classes import Node, Neighbour, Demand, Arc, Graph

class GreedyAgent1:
    def __init__(self):
        pass

    def select_action(self, state):
        greedy_action = list(state.demands_unfulfilled.values())[0]
        return greedy_action

class RandomAgent:
    def __init__(self):
        pass

    def select_action(self, state):
        random_action = random.choice(list(state.demands_unfulfilled.values()))
        return random_action

class DQNAgent_offline:
    def __init__(self, estimator):
        self.estimator = estimator

    def select_action(self, state):
        best_action = None
        best_value = float('inf')

        # select best action from available actions
        for action in state.demands_unfulfilled.values():
            
            temp_graph = copy.deepcopy(state.graph)
            temp_demands = copy.deepcopy(state.all_demands)
            env_in = CSP_Environment(temp_graph, temp_demands)
            env_in.current_state = copy.deepcopy(state)
            temp_action = copy.deepcopy(action)
            next_state, reward, done = env_in.step(temp_action)

            # compute the value of the next state
            next_state_value = self.estimator.predict([next_state.get_feature_vector()])

            # select the action that minimises the value of the next state
            if next_state_value < best_value:
                best_value = next_state_value
                best_action = action

        return best_action
