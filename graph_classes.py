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
from tools import generate_graph, visualise_graph, build_offline_dataset, compute_td_target, visualize_loss, print_shortest_path_source_to_all_nodes
from environments import State, CSP_Environment


class Node:
    def __init__(self, id, numeric_id, supply = {}, dist = float('inf'), prev = None, pos = None):
        self.id = id
        self.supply = {}
        self.numeric_id = numeric_id

        #for shortest paths
        self.prev = None
        self.dist = dist
        self.pos = pos
        self.PERM = False
        self.shortest_path = deque()


class Neighbour:
    def __init__(self, node, transit_time, capacity, arc_id = None):
        self.id = node.id
        self.node = node
        self.transit_time = transit_time
        self.capacity = capacity
        self.flow = 0

        #for min flow
        self.arc_id = arc_id

class Demand:
    def __init__(self, id, from_node, to_node, payload):
        self.id = id
        self.from_node = from_node
        self.to_node = to_node
        self.payload = payload
        self.arcs = []
        self.routes = []

class Arc:
    def __init__(self, id, from_node, to_node, transit_time, capacity):
        self.id = id
        self.from_node = from_node
        self.to_node = to_node
        self.transit_time = transit_time
        self.capacity = capacity


class Graph:

    def __init__(self, V = None):
        self.V = V
        self.graph = defaultdict(list)
        self.nodeDict = {}
        self.id_lookup = {}
        self.node_index = 0
        self.demands = {} # A dictionary of demand objects
        self.arcs = {}
        self.arc_index = 0
        self.demand_id = 0

        #for extracting paths from flow
        self.simplePaths = []
        self.visited = []
        self.currentPath = []
        self.simplePaths_arcs = []
        self.currentPath_arcs = []

    def reset(self):
        
        for node in self.nodeDict.values():
            node.dist = float('inf')
            node.prev = None
            node.pos = None
            node.PERM = False
            node.shortest_path = deque()

        self.simplePaths = []
        self.visited = []
        self.currentPath = []
        self.simplePaths_arcs = []
        self.currentPath_arcs = []

        for demand in self.demands.values():
            demand.arcs = []
            demand.routes = []

    def init_min_cost_flow_arguments(self):

        #arguments for min_cost_flow or-tools module
        self.start_nodes = np.array([])
        self.end_nodes = np.array([])
        self.capacities = np.array([])
        self.unit_costs = np.array([])
        self.supplies = [0 for i in range(self.V)]

    def addVertex(self, node):
        self.nodeDict[node.id] = node


    def addEdge(self, src, dest, transitTime = None, capacity = None, flow = None, directed = False, arc_id = None):
        
        if not directed:
        #add vertex if not already present
            if len(self.graph[src]) == 0:
                srcNode = Node(src, self.node_index)
                self.addVertex(srcNode)
                self.id_lookup[self.node_index] = src
                self.node_index +=1
            else:
                srcNode = self.nodeDict[src]

            if len(self.graph[dest]) == 0:
                end_index = self.node_index
                destNode = Node(dest, self.node_index)
                self.addVertex(destNode)
                self.id_lookup[self.node_index] = dest
                self.node_index +=1
            else:
                destNode = self.nodeDict[dest]

            newNeighbour1 = Neighbour(destNode, transitTime, capacity)
            self.graph[src].insert(0, newNeighbour1)

            newNeighbour2 = Neighbour(srcNode, transitTime, capacity)
            self.graph[dest].insert(0, newNeighbour2)

            self.arcs[self.arc_index] = Arc(self.arc_index, srcNode, destNode, transitTime, capacity)
            self.arc_index +=1
            self.arcs[self.arc_index] = Arc(self.arc_index, destNode, srcNode,  transitTime, capacity)
            self.arc_index +=1

        elif directed:
            if src in self.nodeDict.keys():
                srcNode = self.nodeDict[src]
            else:
                srcNode = Node(src, self.node_index)
                self.addVertex(srcNode)
                self.id_lookup[self.node_index] = src
                self.node_index +=1

            if dest in self.nodeDict.keys():
                destNode = self.nodeDict[dest]
            else:
                destNode = Node(dest, self.node_index)
                self.addVertex(destNode)
                self.id_lookup[self.node_index] = dest
                self.node_index +=1

            # sometimes we want to have a pre - decided arc id for every arc. Helps in finding arcs easily for finding flow through routes and updating capacity of arcs
            if arc_id == None:
                arc_idx = self.arc_index
                self.arc_index +=1
            else:
                arc_idx = arc_id

            newNeighbour1 = Neighbour(destNode, transitTime, capacity, arc_id = arc_idx)
            self.graph[src].insert(0, newNeighbour1)
            

            self.arcs[arc_idx] = Arc(arc_idx, srcNode, destNode, transitTime, capacity)
            

        self.V = len(self.nodeDict)
        self.original_arcs = copy.deepcopy(self.arcs)

    # a function to add demands to nodes
    def add_supply(self, demand_dict):

        # for each demand - create a new demand object with a demand_id
        # add demand value to each node at demand id

        self.demand_dict = demand_dict
        for demand in demand_dict:
            self.demands[self.demand_id] = Demand(self.demand_id, self.nodeDict[demand["from"]], self.nodeDict[demand["to"]], demand["payload"])

            for node in self.nodeDict.values():
                if demand["from"] == node.id:
                    node.supply[self.demand_id] = demand["payload"]
                elif demand["to"] == node.id:
                    node.supply[self.demand_id] = -demand["payload"]
                else:
                    node.supply[self.demand_id] = 0

            self.demand_id +=1

    def add_single_supply(self, demand):
        self.supplies = [0 for i in range(self.V)]
        self.supplies[demand.from_node.numeric_id] = demand.payload
        self.supplies[demand.from_node.numeric_id] = demand.payload

    def calculate_flow_milp(self, demand_subset = None, hide_output = True):
        # self.reset()
        solver =pywraplp.Solver.CreateSolver('GLOP')

        if demand_subset:
            demands = demand_subset
        else:
            demands = self.demands

        #Decision Variables
        x={}
        # x[i] store n decision variables for n demands for arc i, storing the flow of a specific demand
        # Therefore, x[i][j] is flow in arc i, demand j
        for i in range(len(self.arcs)):
            x[i]=[solver.NumVar(0,solver.infinity(),'x[%d][%d]'%((i),(j))) for j in demands.keys()]

        #Constraints
        #Arc Capacity - The sum of all flows(of different demands) should be less than the total capacity of the arc
        for i in range(len(self.arcs)):
            expr =[x[i][j] for j in range(len(demands))]
            solver.Add(sum(expr) <= self.arcs[i].capacity)

        #Flow Conservation - Flow has to be conserved at each node for each demand
        for j in range(len(demands)):
            for node in self.nodeDict.values():
                expr1= []
                expr2= []
                for i in range(len(self.arcs)):
                    if node.id == self.arcs[i].from_node.id:
                        expr1.append(x[i][j])
                    if node.id == self.arcs[i].to_node.id:
                        expr2.append(x[i][j])

                solver.Add(sum(expr1)  - sum(expr2) <= node.supply[demands[list(demands.keys())[j]].id]) 
                solver.Add(sum(expr1)  - sum(expr2) >= node.supply[demands[list(demands.keys())[j]].id]) 

        if not hide_output:
            print("Number of Constraints - ", solver.NumConstraints())

        #Objective
        objective_terms=[]
        for i in range (len(self.arcs)):
            for j in range (len(demands)):
                objective_terms.append(self.arcs[i].transit_time *x[i][j])

        solver.Minimize(solver.Sum(objective_terms))
        status =solver.Solve()

        #Printing Solution
        if status == pywraplp.Solver.OPTIMAL:
            if not hide_output:
                print ('Minimum Cost = ', solver.Objective().Value())
                print()
            for j in range(len(demands)):
                for i in range(len(self.arcs)):
                    if x[i][j].solution_value() ==0: continue
                    if not hide_output:
                        print('From:',self.arcs[i].from_node.id, ', To:',self.arcs[i].to_node.id, ', Demand Id: ', demands[list(demands.keys())[j]].id, 'Demand_From:', demands[list(demands.keys())[j]].from_node.id, ', Demand_To:', demands[list(demands.keys())[j]].to_node.id, ', Payload: ', demands[list(demands.keys())[j]].payload,  x[i][j] , ' = ', x[i][j].solution_value(), ' Arc Capacity: ', self.arcs[i].capacity)
                    demands[list(demands.keys())[j]].arcs.append((self.arcs[i], x[i][j].solution_value()))
                # demands[list(demands.keys())[j]].routes = self.generate_routes(demands[list(demands.keys())[j]])
        else:
            print ('The problem does not have an optimal solution')
        
        return solver.Objective().Value()

    def generate_routes_csp(self, hide_output = True, export_output = False, show_routes = True):
        total_time = self.calculate_flow_milp(hide_output = hide_output)
        print("Total transit time - ", total_time)
        output = {}
        for demand in self.demands.values():
            demand.routes = self.generate_routes(demand)
            output[demand.id] = {"from": demand.from_node.id, "to" : demand.to_node.id, "payload": demand.payload, "routes" : demand.routes}
            if show_routes:
                print("Demand From", demand.from_node.id, ", Demand To", demand.to_node.id,", Payload ", demand.payload, "Routes - ", demand.routes)

        if export_output:
            with open("MILP_Routes.json",'w') as fi:
                json.dump(output,fi, indent =4)

    #Unused - could be used for integral flows with rounded down supplies
    def calculate_min_cost_flow(self, demand):
        smcf = min_cost_flow.SimpleMinCostFlow()
        all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(self.start_nodes, self.end_nodes, self.capacities, self.unit_costs)
        # Add supply for each nodes.
        smcf.set_nodes_supplies(np.arange(0, len(self.supplies)), self.supplies)

        status = smcf.solve()

        if status != smcf.OPTIMAL:
            print("There was an issue with the min cost flow input.")
            print(f"Status: {status}")
            # exit(1)
        print(f"Minimum cost: {smcf.optimal_cost()}")
        print("")
        print(" Arc    Flow / Capacity Cost")
        solution_flows = smcf.flows(all_arcs)
        costs = solution_flows * self.unit_costs
        for arc, flow, cost in zip(all_arcs, solution_flows, costs):
            if flow ==0 : continue
            # print(smcf.tail(arc), type(smcf.tail(arc)))
            print(
                f"{self.id_lookup[smcf.tail(arc)]:1} -> {self.id_lookup[smcf.head(arc)]}  {flow:3}  / {smcf.capacity(arc):3}       {cost}"
            )
            for neighbour in self.graph[self.id_lookup[smcf.tail(arc)]]:
                if neighbour.id == self.id_lookup[smcf.head(arc)]:
                    neighbour.flow = flow
                
                    #update demand object
                    demand.arcs.append((self.arcs[neighbour.arc_id], flow))
                    #update capacity of the arc
                    self.arcs[neighbour.arc_id].capacity -= flow

        # generate all the routes from the arcs that contribute to this demand             
        demand.routes = self.generate_routes(demand)

    def calculate_flow_greedy(self, show_output = False, export_output = False):
        self.reset()
        self.arcs = copy.deepcopy(self.original_arcs)
        total_cost = 0
        output = {}
        # self.demands = sorted(self.demands, key=lambda x: x.payload, reverse=True)
        for demand in self.demands.values():
            total_cost += self.calculate_flow_milp(demand_subset = {demand.id : demand})
            demand.routes = self.generate_routes(demand)
            output[demand.id] = {"from": demand.from_node.id, "to" : demand.to_node.id, "payload" : demand.payload, "routes" : demand.routes}
            if show_output:
                print("Demand From", demand.from_node.id, ", Demand To", demand.to_node.id,", Payload ", demand.payload, "Routes - ", demand.routes)
            #update arc capacities
            for arc, flow in demand.arcs:
                arc.capacity -= flow

        if export_output:
            with open("Greedy_Routes.json",'w') as fi:
                json.dump(output,fi,indent =4)

        print("Total Transit Time - ", total_cost)


    def rl_greedy(self, show_output = False, export_output = False):
        self.reset()
        self.arcs = copy.deepcopy(self.original_arcs) #updating to get the original capacities of the arcs

        output = {}
        env = CSP_Environment(self, self.demands)
        agent = GreedyAgent1()
        # agent = RandomAgent()
        current_state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(current_state)
            next_state, reward, done = env.step(action)
            env.current_state = next_state
            total_reward += reward
            current_state = next_state
            #obs.display_state() # uncomment for debugging
            if done:
                current_state.display_state()
                print("Total Transit Time: ", total_reward)
                break

        for demand in self.demands.values():
            demand.routes = self.generate_routes(demand)
            output[demand.id] = {"from": demand.from_node.id, "to" : demand.to_node.id, "payload": demand.payload, "routes" : demand.routes}
            if show_output:
                print("Demand From", demand.from_node.id, ", Demand To", demand.to_node.id,", Payload ", demand.payload, "Routes - ", demand.routes)

        if export_output:
            with open("RL_Greedy_Routes.json",'w') as fi:
                json.dump(output,fi, indent =4)
        return total_reward

    def rl_offline(self, offline_data = [], fittedQ_iterations = 20, show_output = False, export_output = False):
        self.reset()
        self.arcs = copy.deepcopy(self.original_arcs)
        output = {}
        if len(offline_data) == 0:
            offline_data = build_offline_dataset(self, iterations = 100, action_agent = RandomAgent())
            print("Offline Dataset Generated!!")
        class ZeroEstimator:
            def predict(self, X):
                return np.zeros(len(X))
        # FQI algorithm
        discount_factor = 0.9
        iterations = fittedQ_iterations
        estimator = ZeroEstimator()

        state_rep = np.array(offline_data["state"].tolist())
        next_state_rep = np.array(offline_data["next_state"].tolist())

        total_rewards = []
        min_reward = 100000000000000

        training_graph = copy.deepcopy(self)
        training_demands = copy.deepcopy(self.demands)

        for i in range(iterations):

            X = state_rep
            y = []
            for index, row in offline_data.iterrows():
                state, reward, next_state, is_terminal, possible_states = row
                y.append(compute_td_target(next_state, reward, is_terminal, discount_factor, estimator, possible_states))

            y = np.array(y)
            # print("X Dimensions: ", X.shape, "Y Dimensions", y.shape)

            estimator = MLPRegressor(max_iter=500, hidden_layer_sizes=(256, 128, 64, 32))
            estimator.fit(X, y)

            # compute the reward using the current estimator
            env = CSP_Environment(training_graph, training_demands)
            agent = DQNAgent_offline(estimator)
            obs = env.reset()
            total_reward = 0
            # current_state = env.reset()
            done = False
            # t=0
            while not done:
                action = agent.select_action(obs)
                next_state, reward, done = env.step(action)
                env.current_state = next_state
                total_reward += reward
                obs = next_state
                # obs.display_state()
                if done:
                    # obs.display_state()
                    print("Total Cost: ", total_reward)
                    total_rewards.append(total_reward)
                    break

            if total_reward < min_reward:
                min_reward = total_reward
                best_estimator = copy.deepcopy(estimator)

            
            visualize_loss(total_rewards)

            print(f'Iteration {i+1} out of {iterations} complete')


        filename = 'finalized_model.sav'
        pickle.dump(best_estimator, open(filename, 'wb'))
        filename = 'finalized_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))

        env = CSP_Environment(self, self.demands)
        agent = DQNAgent_offline(loaded_model)
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(obs)
            next_state, reward, done = env.step(action)
            env.current_state = next_state
            total_reward += reward
            obs = next_state
            # obs.display_state()
            if done:
                obs.display_state()
                print("Total Transit Time DQNAgent_offline: ", total_reward)
     
        for demand in self.demands.values():
            demand.routes = self.generate_routes(demand)
            output[demand.id] = {"from": demand.from_node.id, "to" : demand.to_node.id, "payload": demand.payload, "routes" : demand.routes}
            if show_output:
                print("Demand From", demand.from_node.id, ", Demand To", demand.to_node.id,", Payload ", demand.payload, "Routes - ", demand.routes)

        if export_output:
            with open("RL_OfflineAgent_Routes.json",'w') as fi:
                json.dump(output,fi, indent =4)

        return total_reward

            

    #a depth first search to generate all simple paths in a digraph
    def DFS(self, temp_graph, start, end):
        if self.visited[temp_graph.nodeDict[start].numeric_id] == True:
            return 

        self.visited[temp_graph.nodeDict[start].numeric_id] = True
        self.currentPath.append(start)

        if start == end:
            self.simplePaths.append(self.currentPath)
            self.simplePaths_arcs.append(self.currentPath_arcs)
            self.simplePaths = copy.deepcopy(self.simplePaths)
            self.simplePaths_arcs = copy.deepcopy(self.simplePaths_arcs)
            self.visited[temp_graph.nodeDict[start].numeric_id] =  False
            self.currentPath.pop()
            self.currentPath_arcs.pop()
            return 

        for neighbour in temp_graph.graph[start]:
            self.currentPath_arcs.append(neighbour.arc_id)
            self.DFS(temp_graph, neighbour.id, end)

        self.currentPath.pop()
        if len(self.currentPath) !=0:
            self.currentPath_arcs.pop()
        self.visited[temp_graph.nodeDict[start].numeric_id] = False
        return 


    # a function to generate routes given the arcs utilised to fulfill demand
    def generate_routes(self, demand):
        """demand.arcs = [(arc1, flow1), (arc2, flow2)]"""

        # Create a temporary digraph. We add an arc if between the two nodes there is a flow for the particular instance of demand
        temp_graph = Graph()
        for arc, flow in demand.arcs:
            temp_graph.addEdge(arc.from_node.id, arc.to_node.id, directed = True, flow = flow, arc_id = arc.id)

        self.visited = [False for i in range(temp_graph.V)] 
        self.simplePaths = []
        self.currentPath = []
        self.simplePaths_arcs = []
        self.currentPath_arcs = []
        self.DFS(temp_graph, demand.from_node.id, demand.to_node.id)

        routes = copy.deepcopy(self.simplePaths)
        routes_with_flow = []
        for i in range (len(self.simplePaths)):
            min_flow = 10000000
            for arc_id in self.simplePaths_arcs[i]:
                for arc, flow in demand.arcs:
                    if arc.id == arc_id and flow <= min_flow:
                        min_flow = flow
                        break

            routes_with_flow.append((routes[i], min_flow))


        return routes_with_flow

    ## Methods for shortest paths
    def extract_shortest_path(self, node, reverse):
        current = node
        node.shortest_path.appendleft(current.id)
        while current.prev != None:
            if reverse:
                node.shortest_path.append(current.prev.id)
            else:
                node.shortest_path.appendleft(current.prev.id)
            current = current.prev


    #O(ElogV)
    def dijkstra(self, src, dest = None, reverse = False, show_output = True):
        self.reset()
        V = self.V
        output = {}
        # minHeap represents set E
        minHeap = Heap()

        srcNode = self.nodeDict[src]
        index=0
        for node in self.nodeDict.values():
            node.pos = index
            minHeap.array.append(node)
            index +=1

        minHeap.size = V
        minHeap.buildMinHeap()
        minHeap.updateDist(srcNode, 0)

        while minHeap.isEmpty() == False:

            # Extract the vertex with minimum distance label
            newHeapNode = minHeap.extractMin()
            self.extract_shortest_path(newHeapNode, reverse)

            for neighbour in self.graph[newHeapNode.id]:

                # update distance label if shortest distance not found already
                if (neighbour.node.PERM == False and neighbour.transit_time + newHeapNode.dist < neighbour.node.dist):
                    neighbour.node.prev = newHeapNode
                    minHeap.updateDist(neighbour.node, neighbour.transit_time + newHeapNode.dist)

            if dest != None and newHeapNode.id == dest:
                if show_output:
                    print ("Source\tDestination\tTransit Time\t Shortest Path")
                    print ( src, "\t", newHeapNode.id,"\t\t", newHeapNode.dist,"\t\t", list(newHeapNode.shortest_path))
                output[newHeapNode.id] = {"route" : list(newHeapNode.shortest_path), "transit_time" : newHeapNode.dist}
                return output

        if show_output:
            if reverse:
                dest = src
                print_shortest_path_source_to_all_nodes(self.nodeDict, constant = dest, vertex = "Source") 
            else:
                print_shortest_path_source_to_all_nodes(self.nodeDict, constant = src, vertex = "Destination")
        
        for node in self.nodeDict.values():
            output[node.id] = {"route" : list(node.shortest_path), "transit_time" : node.dist}
        return output

    def shortest_path_source_to_all_nodes(self, src, export_output = False, show_output = True):
        output = self.dijkstra(src, show_output = show_output)
        if export_output:
            with open("SrcToAllNodes_ShortestPaths.json",'w') as fi:
                json.dump(output,fi, indent =4)
        return output

    def shortest_path_dest_from_all_nodes(self, dest, export_output = False, show_output = True):
        output = self.dijkstra(dest, reverse = True, show_output = show_output)
        if export_output:
            with open("DestFromAllNodes_ShortestPaths.json",'w') as fi:
                json.dump(output,fi, indent =4)
        return output

    def shortest_path_source_to_dest(self, src, dest, export_output = False, show_output = True):
        #Passing destination for a single route
        output = self.dijkstra(src, dest, show_output = show_output)
        if export_output:
            with open("SrcToDest_ShortestPath.json",'w') as fi:
                json.dump(output,fi, indent =4)
        return output

    def shortest_path_all_pairs(self, export_output = False, show_output = False):
        output_main = {}
        for node in self.nodeDict.values():
            output_main[node.id] = self.dijkstra(src = node.id, show_output = show_output)
        if export_output:
            with open("AllPairs_ShortestPaths.json",'w') as fi:
                json.dump(output_main,fi, indent =4)
        return output_main         

        


