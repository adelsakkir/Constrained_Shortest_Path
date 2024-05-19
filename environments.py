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
##from custom_data_structures import Heap
##from tools import generate_graph, visualise_graph, build_offline_dataset, compute_td_target, visualize_loss, print_shortest_path_source_to_all_nodes
##from graph_classes import Node, Neighbour, Demand, Arc, Graph


class State:
    def __init__(self, demands_fulfilled, demands_unfulfilled, temp_graph):
        self.demands_fulfilled = demands_fulfilled  #dict of demands fulfilled
        self.demands_unfulfilled = demands_unfulfilled  #dict of demands unfulfilled
        self.graph = temp_graph
        self.all_demands = dict(ChainMap(self.demands_fulfilled, self.demands_unfulfilled))
        self.available_actions = list(self.demands_unfulfilled.values())
        self.feature_vector = self.get_feature_vector()


    def get_feature_vector(self):
        feature_vector = np.zeros(len(self.demands_fulfilled) + len(self.demands_unfulfilled), dtype = int)
        for idx in self.demands_fulfilled.keys():
            feature_vector[idx] = 1
        return feature_vector
    
    def display_state(self):
        demands_fulfilled_id = []
        for demand in self.demands_fulfilled.values():
            demands_fulfilled_id.append(demand.id)
        return (demands_fulfilled_id)

# A RL environment for the agent to interact with
class CSP_Environment:
    def __init__(self, graph, demands):
        self.graph = graph
        self.demands = demands #all demands
        self.current_state = State(demands_fulfilled = {}, demands_unfulfilled = demands, temp_graph = self.graph)

    def reset(self):
        """Return intial state"""
        for demand in self.demands.values():
            demand.arcs = []
            demand.routes = []

        self.graph.arcs = copy.deepcopy(self.graph.original_arcs)

        self.current_state = State(demands_fulfilled = {}, demands_unfulfilled = self.demands, temp_graph = self.graph)
        return self.current_state

    def step(self, action):
        """
        Parameters:  action - demand
        Returns: next_state, reward, done
        """
        done = False
        next_demand = action
        reward = self.current_state.graph.calculate_flow_milp(demand_subset = {next_demand.id : next_demand}) #temporary graph arc capacities will get updated
        #update capacity on arcs
        for arc, flow in next_demand.arcs:
            arc.capacity -= flow
        updated_graph = self.current_state.graph #apply deepcopy if necessary

        demands_unfulfilled = {}
        for demand in self.current_state.demands_unfulfilled.values():
            if demand.id != next_demand.id:
                demands_unfulfilled[demand.id] = demand

        demands_fulfilled = copy.copy(self.current_state.demands_fulfilled) #shallow copy
        demands_fulfilled[next_demand.id] = next_demand


        next_state = State(demands_fulfilled = demands_fulfilled, demands_unfulfilled = demands_unfulfilled, temp_graph = updated_graph)
        
        if len(demands_unfulfilled)==0:
            done = True

        return next_state, reward, done
