Constrained Shortest Path/ Multi-commodity Flow

This is a short summary of the solutions for the problems provided in the task, instructions to run files and results obtained. The problem is solved through the implementation of the following scripts - 
1. tools.py
2. graph_classes.py
3. agents.py
4. environments.py
5. custom_data_structures.py

The main files to be run for the three sections are - 
1. "main_sp_part1.py"
2. "main_csp_part2.py"
3. "main_rl_part3.py"<br />

## Part 1 - **Shortest-Path** - "main_sp_part1.py"
![image](https://github.com/adelsakkir/maersk_task_adel_sakkir/assets/63802234/4f2656e6-d38d-4b58-af17-49229e998a88)
This was the instance of the graph provided in "exercise_baseline.json". The shortest path algorithms are developed as methods to the Graph object. Please run ""**main_sp_part1.py**" for the following sections.
##### a) Shortest path between any two vertices - graph.shortest_path_source_to_dest(src = 'co', dest = 'wr', export_output = True, show_output = True)
![image](https://github.com/adelsakkir/maersk_task_adel_sakkir/assets/63802234/bce2266a-b544-43e1-933a-375e50d87486)

Output exported to  - "SrcToDest_ShortestPath.json"
##### b)	Shortest paths to a single vertex from every other vertex - graph.shortest_path_dest_from_all_nodes(dest = 'co', export_output = True, show_output = True) 
![image](https://github.com/adelsakkir/maersk_task_adel_sakkir/assets/63802234/4115d906-190e-4b63-8b47-8890ff70150e)<br />
Output exported to  - "DestFromAllNodes_ShortestPaths.json"<br />
##### c)	Shortest path from a single vertex to every other vertex - graph.shortest_path_source_to_all_nodes(src = 'co', export_output = True, show_output = True)<br />
![image](https://github.com/adelsakkir/maersk_task_adel_sakkir/assets/63802234/ef2ac191-916c-493e-a50b-b50ca3412efb)<br />
Output exported to  - "SrcToAllNodes_ShortestPaths.json"<br />
##### d)	Shortest paths between every pair of vertices - graph.shortest_path_all_pairs(export_output = True, show_output = False)<br />
Output hidden. Please run ""main_sp_part1.py" to see the output of the all pairs shortest paths algorithm. Set the (show_output = True) argument to "True" to see the shortest paths for all pairs on your window. <br />
Output exported to  - "AllPairs_ShortestPaths.json"<br />
##### e)	Test runs using randomly generated graphs - generate_graph(vertices, edges, show_graph_visual, size = (14,8), export_output= True)
Random graphs are generated using the generate_graph(vertices, edges, show_graph_visual, size = (14,8), export_output= True) method in the "tools.py" script.<br />
![image](https://github.com/adelsakkir/maersk_task_adel_sakkir/assets/63802234/7fea9f5e-64ec-463a-86eb-44c8d1989f55) <br />
The above is an example of a randomly generated graph with 12 vertices and 40 edges. The graph is exported to "RandomlyGenerated_Graph.json" <br/>
Output exported to  - "AllPairs_ShortestPaths.json"<br />

## Part 2 - **Constrained Shortest Path** - "main_csp_part2.py"
![image](https://github.com/adelsakkir/maersk_task_adel_sakkir/assets/63802234/5f585074-0e52-4f20-8be1-f08bfe25cf52)
This was the instance of the graph provided in "exercise_bonus.json".
The constrained shortest path problem is solved with linear programming. The "calculate_flow_milp()" method of the graph class creates a LP model for the same using google OR tools. The script "main_csp_part2.py" displays results for two separate methods for the same 
1. A complete LP approach whose solution is optimal
![image](https://github.com/adelsakkir/maersk_task_adel_sakkir/assets/63802234/f870fc7a-54b1-43de-ac13-07d278f07971)
This is a screenshot of the partial output of the algorithm. The total transit time comes to - **100954.6**. Each of the remaining below can be read as a demand from node A to node B of payload x is fulfilled through (Route1, Flow1), (Route2, Flow2), etc. The sum of all the flows will equal the total payload of the demand. <br/>
The output is exported to "MILP_Routes.json" <br/>
2. A greedy method by a sequential allocation of demand.
   The greedy methods iterates through each demand and allocates them immediately. Once a demand is allocated, the capacities of the arc used are updated, and a flow for the remaining demands are done in the same fashion. The total transit time comes to - **103806.2**

## Part 3 - **Reinforcement Learning based shortest paths** - "main_rl_part3.py"
The problem is modelled as a sequential demand allocation problem with the objective of finding the optimal sequence of allocation of demands to minimize total transit times. The state, action, rewards of the MDP are defined below. <br/>
**State** -  The demands that are already allocated and the capacity of the arcs based on the current allocation. <br/>
**Action** - The next demand to be allocated. <br/>
**Rewards** - Computed using an LP. We find the optimal routes and flows of a demand given current capacity of the arcs. The objective value of the LP is used as the reward for the state-action pair. <br/>

#### Description of the method
An offline reinforment learning algorithm is implemented with fitted Q iterations, a batch learning method. We begin by creating a dataset using 100 iterations of the problem and random actions. Each row consists of features of the current state, the next state given the action chosen and the associated reward, a binary variable indicating whether the next state is terminal and all possible states that can be reached after another action is chosen. With the dataset ready, the learning process begins with initialising a ZeroEstimator that predicts zero values for any inputs. At each iteration, a target value y for each state in the dataset is computed. We use these values to train an MLPestimator - a feedforward artificial neural network - which seeks to approximate the optimal Q-function. Following the training in each iteration, the updated estimator is used to compute the next TD target value. Actions are chosen based on the estimator's predictions, with the best action being the one that leads to a state with the lowest value. <br/><br/>

We observe the following on 10 runs of the reinforcement learning algorithm. We obtain an average total transit time of - **102716.32** for the offline DQN algorithm. <br/>
#### Total Transit Time
1) MILP (Optimal) - **100954.6**
2) Greedy (Baseline) - **103806.2**
3) Offline DQN (Average) - **102716.32** <br/>
![results_picture](https://github.com/adelsakkir/maersk_task_adel_sakkir/assets/63802234/c4e615d1-5b2d-4c63-8e24-cdc607b0603f) <br/>
The routes and flows generated by the reinforcement learning algorithm is exported into the "RL_OfflineAgent_Routes.json" file. <br/>

Thanks, 
Adel Sakkir



