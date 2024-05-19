# Maersk technical assessment - Adel Sakkir 2024

This is a short summary of the solutions for the problems provided in the task, instructions to run files and results of obtained. The problem is solved through the implementation of the following scripts - 
1. tools.py
2. graph_classes.py
3. agents.py
4. environments.py
5. custom_data_structures.py

The main files to be run for the three sections are - 
1. "main_sp_part1.py"
2. "main_csp_part2.py"
3. "main_rl_part3.py"
   <br />

1. **Shortest-Path** - "main_sp_part1.py"
![image](https://github.com/adelsakkir/maersk_task_adel_sakkir/assets/63802234/4f2656e6-d38d-4b58-af17-49229e998a88)
This was the instance of the graph provided in "exercise_baseline.json". The shortest path algorithms are developed as methods to the Graph object. Please run ""**main_sp_part1.py**" for the folling sections.
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
Output hidden. Please run ""main_sp_part1.py" to see the output of the all pairs shortest paths algorithm.<br />
Output exported to  - "AllPairs_ShortestPaths.json"<br />
##### e)	Test runs using randomly generated graphs
Random graphs are generated using the generate_graph(vertices, edges, show_graph_visual, size = (14,8), export_output= False) method in the "tools.py" script.<br />
Output exported to  - "AllPairs_ShortestPaths.json"<br />

3. **Constrained Shortest-Path**
4. **Reinforcement Learning based shortest paths**

   
![results_picture](https://github.com/adelsakkir/maersk_task_adel_sakkir/assets/63802234/c4e615d1-5b2d-4c63-8e24-cdc607b0603f)

