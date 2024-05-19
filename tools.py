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
from environments import State, CSP_Environment
##from graph_classes import Node, Neighbour, Demand, Arc, Graph
from agents import GreedyAgent1, RandomAgent, DQNAgent_offline

#function to generate a random graph
def generate_graph(vertices, edges, show_graph_visual = True, size = (14,8), export_output= False):
    if edges < vertices -1 or edges > ((vertices)*(vertices - 1))/2: #tree criteria and complete graph criteria
        print("""Cannot generate a coneected graph with given input parameters!!\nPlease ensure that the number of edges is atleast number vertices - 1""")
        return
    nodes = []
    # node_dict = {}
    i = 0
    while len(nodes)!= vertices:
        node_name = ''.join(random.choices(string.ascii_lowercase, k=2))
        if node_name not in nodes:
            nodes.append(node_name)
            i+=1

    edges_list = []
    rem_nodes = copy.copy(nodes)
    tree = [rem_nodes.pop()]
    nodeDict = defaultdict(list)

    while len(rem_nodes) != 0:
        start_node = random.choice(tree)
        next_node = random.choice(rem_nodes)
        tree.append(next_node)
        rem_nodes.remove(next_node)
        edges_list.append({"from" : start_node, "to" : next_node, "transit_time" : random.randint(10,100)})
        nodeDict[start_node].append(next_node)
        nodeDict[next_node].append(start_node) #undirected graph

    remaining_edges = edges - (vertices - 1) # number of edges of a tree = v-1
    while remaining_edges != 0:
        while True:
            start_node = random.choice(nodes)
            next_node = random.choice(nodes)
            if next_node != start_node and next_node not in nodeDict[start_node]:
                break
        edges_list.append({"from" : start_node, "to" : next_node, "transit_time" : random.randint(10,100)})
        nodeDict[start_node].append(next_node)
        nodeDict[next_node].append(start_node) #undirected graph
        remaining_edges -= 1
    
    if show_graph_visual:
        visualise_graph({"nodes" : nodes, "edges" : edges_list}, size)

    output = {"nodes" : nodes, "edges" : edges_list}

    if export_output:
        with open("RandomlyGenerated_Graph.json",'w') as fi:
            json.dump(output,fi)
    return output

def visualise_graph(graph_data, size = (14,8)):
    g = nx.Graph()
    for edge in graph_data["edges"]:
        g.add_edge(edge["from"], edge["to"], weight = edge["transit_time"])
    plt.figure(3,figsize= size)
    nx.draw(g, node_color="#33FF39", with_labels =True)
    plt.show()

#a function to generate an offline dataset based on random actions for the RL agent 
def build_offline_dataset(graph, iterations = 100, action_agent = RandomAgent(), hide_output = True):
    
    #Reseting to original parameters
    graph.reset()
    graph.arcs = copy.deepcopy(graph.original_arcs)
    ##########

    offline_graph = copy.deepcopy(graph)
    offline_demands = copy.deepcopy(graph.demands)
    env = CSP_Environment(offline_graph, offline_demands)
    offline_data = []
    for i in range(iterations):
        print("Offline dataset building, iteration- ", i, " / ", iterations)
        done = False
        env.reset()
        while not done:

            current_state = env.current_state
            state_representation = current_state.get_feature_vector()
            if not hide_output:
                print ("Current State - ", current_state.display_state())
            # epsilon greedy - irrelavant if random agent
            if random.random() < 0.9:
                action = action_agent.select_action(env.current_state)
            else:
                action = random.choice(env.current_state.available_actions)
                
            next_state, reward, done = env.step(action)
            next_state_representation = next_state.get_feature_vector()
            env.current_state = next_state

            # get all possible future states to compute td-target
            possible_next_states = []
            current_state_copy = copy.deepcopy(next_state)

            for action in env.current_state.available_actions:
                # start_time = datetime.now()

                graph_copy = copy.deepcopy(env.graph)
                demands_copy = copy.deepcopy(env.demands)
                env_copy = CSP_Environment(graph_copy, demands_copy)
                env_copy.current_state = current_state_copy

                # end_time = datetime.now()
                # print('Duration DeepCopy: {}'.format(end_time - start_time))
                
                # start_time = datetime.now()
                future_state, reward, done = env_copy.step(action)
                # end_time = datetime.now()
                # print('Duration Step: {}'.format(end_time - start_time))

                # get a feature vector representation of the state
                future_state_representation = future_state.get_feature_vector()
                possible_next_states.append(future_state_representation)


            # add the observation to the dataset
            offline_data.append([state_representation, reward, next_state_representation, done, possible_next_states])
            # print("Appended to offline dataset")

        # break


    return pd.DataFrame(offline_data, columns=["state", "reward", "next_state", "is_terminal", "possible_next_states"])


#a function to compute td-target for the RL agent 
def compute_td_target(next_state, reward, is_terminal, discount_factor, estimator, possible_states):
    if is_terminal:
        return reward
    else:
        # compute the value of the next state
        min_value = 1000000000
        for possible_state in possible_states:

            possible_value = estimator.predict([possible_state])[0]
            if possible_value < min_value:
                min_value = possible_value
        return reward + discount_factor * min_value

def visualize_loss(loss_values):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Loss Value')
    plt.plot(range(len(loss_values)), loss_values)
    plt.ylim(ymin=0)
    plt.show(block=False)

# a helper function to print the shortest paths
def print_shortest_path_source_to_all_nodes(nodeDict, constant, vertex = "Destination"):
    print ("Source\tDestination\tTransit Time\t Shortest Path")
    for node in nodeDict.values():
        if vertex == "Destination":
            dest = node.id
            src = constant
        else:
            src = node.id
            dest = constant
        print ( src,"\t", dest,"\t\t", node.dist,"\t\t", list(node.shortest_path))
