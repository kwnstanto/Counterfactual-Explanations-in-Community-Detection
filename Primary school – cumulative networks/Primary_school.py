# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 22:17:25 2025

@author: Konstanto
"""

import pandas as pd

from collections import defaultdict
import networkx as nx
import csv
import igraph as ig
import itertools
from datetime import timedelta
import time
import json
import os
from copy import deepcopy
from sklearn.metrics.cluster import normalized_mutual_info_score

# Load and process the file
file_path = r".\Datasets\Primary school – cumulative networks\Primary School-Day 1(Duration).txt"

# Read the file and extract unique node IDs
node_connections = []
unique_nodes = set()

with open(file_path, "r") as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) == 3:
            node1, node2, weight = map(int, parts)
            node_connections.append((node1, node2, weight))
            unique_nodes.update([node1, node2])

# Sort unique nodes and assign new IDs
sorted_nodes = sorted(unique_nodes)
node_mapping = {old_id: new_id for new_id,
                old_id in enumerate(sorted_nodes, start=1)}

# Apply the mapping to connections
remapped_connections = [(node_mapping[n1], node_mapping[n2], w)
                        for n1, n2, w in node_connections]

# Save the remapped data to a new file
remapped_file_path = r".\Primary school – cumulative networks\Remapped_Nodes.txt"
with open(remapped_file_path, "w") as file:
    for n1, n2, w in remapped_connections:
        file.write(f"{n1}\t{n2}\t{w}\n")

# Skip the first two lines (header and metadata) and load the actual edge data into a DataFrame
column_names = ["node1", "node2", "weight"]
edge_data = pd.read_csv(
    remapped_file_path, delim_whitespace=True, skiprows=2, names=column_names)

# Rescale weights to the range [0, 1] by dividing by the maximum weight
edge_data['weight'] = edge_data['weight'] / edge_data['weight'].max()
# Round weights to 5 decimal places for consistency
edge_data['weight'] = edge_data['weight'].round(5)

edge_data.to_csv(r'.\Primary school – cumulative networks\Primary School-Day 1(Duration)_rescaled.txt',
                 header=None, index=None, sep=' ', quoting=csv.QUOTE_NONE, escapechar=' ')
# Create a graph using NetworkX from the rescaled edge data
G = nx.read_weighted_edgelist(
    r'.\Primary school – cumulative networks\Primary School-Day 1(Duration)_rescaled.txt', create_using=nx.Graph())
# create file for Gephi
nx.write_gexf(G, r'.\Primary school – cumulative networks\Primary School-Day 1(Duration).gexf')

threshold = 0.3 # Set the threshold τ={0.3,0.5,0.8}
l=4 # set the bound of contingency set
epsilon = 0.001 # Set the epsilon constant of error for binary search

def com_det_tranform_list_to_diction(lst):
    temp = {}
    for i in range(len(lst)):
        for j in lst[i]:
            temp[str(j)] = i
    return temp

def Louvain(Graph):
    original_ids = list(Graph.nodes())
    id_mapping = {new_id:old_id  for new_id,old_id  in enumerate(original_ids)} # store the mapping between nodes
    G_ig = ig.Graph.from_networkx(Graph) # Transform NetworkX graph to iGraph
    com_det = G_ig.community_multilevel(weights='weight')
    com_list = [c for c in com_det]
    communities=com_det_tranform_list_to_diction(com_list) # transform the list to proper dictionary
    communities_updated = {id_mapping[int(k)]: v for k, v in communities.items()}
    return communities_updated

# Compute the initial community structure with Louvain from iGraph
part=Louvain(G)

# Save dictionaries as JSON
def save_dict(d, filename, path):
    # Combine the path and filename into a full file path
    filepath = os.path.join(path, filename)
    with open(filepath, 'w') as f:
        json.dump(d, f, indent=4)
    print(f"Saved dictionary to {filepath}")

file_path = r"F:\Users\Xristos\Downloads\Math\PhD\Datasets\Weighted\Primary school – cumulative networks"
save_dict(part, "Partition.json", file_path)

# Prepare the edge list from the DataFrame
edge_list = edge_data[['node1', 'node2']].values.tolist()

# Create a dictionary with incident edges
def conn_edges(lst):
    neigh_dict = {}
    for l in lst:
        node1, node2 = int(l[0]), int(l[1])
        neigh_dict.setdefault(str(node1), []).append(node2)
        # Ensure the second node exists in the dictionary
        neigh_dict.setdefault(str(node2), [])
    return neigh_dict

# Fill any missing values to ensure symmetry in connections
def misval(dct):
    for node, neighbors in dct.items():
        for neighbor in neighbors:
            if node not in dct[str(neighbor)]:
                dct[str(neighbor)].append(int(node))

    # Convert all neighbor values to integers for consistency
    for key, value in dct.items():
        # Remove duplicates and sort for consistency
        dct[key] = sorted(set(value))
    return dct

# Reorder the dictionary by keys in ascending order
def sort_neighbors_dict(dct):
    return {str(key): dct.get(str(key), []) for key in range(1, 237)}

# Process the edge list into a dictionary
neighbors_dict = conn_edges(edge_list)
neighbors_dict = misval(neighbors_dict)
neighbors_dict = sort_neighbors_dict(neighbors_dict)  # Sort the dictionary
save_dict(neighbors_dict, "Neighbors.json", file_path)

# Function to find the Endogenous (Ee) and Exogenous (Ex) sets
def endog(edges, part, num_nodes=236):
    Ee = {}  # Endogenous edges as a regular dictionary
    Ex = {}  # Exogenous edges as a regular dictionary

    # Initialize all nodes to have empty lists
    for node in map(str, range(1, num_nodes + 1)):
        Ee[node] = []
        Ex[node] = []

    # Process each edge
    for node1, node2 in edges:
        # Use string keys to match `part` dictionary
        node1, node2 = str(node1), str(node2)
        if part[node1] == part[node2]:  # Same partition (endogenous)
            Ee[node1].append(int(node2))
            Ee[node2].append(int(node1))
        else:  # Different partitions (exogenous)
            Ex[node1].append(int(node2))
            Ex[node2].append(int(node1))

    # Sort and deduplicate edges for consistency
    for node in Ee:
        Ee[node] = sorted(set(Ee[node]))
    for node in Ex:
        Ex[node] = sorted(set(Ex[node]))

    return Ee, Ex

E_e, E_x = endog(edge_list, part)
E_e = misval(E_e)

# Create dictionary for each node with weights of endogenous edges
def dictweights(Ee, edges_with_their_weights):
    dict_weights = defaultdict(dict)

    # Convert edges_with_weights to a dictionary for fast lookups
    edge_weights = {
        (min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1]))): float(edge[2])
        for edge in edges_with_their_weights
    }

    # Populate weights dictionary
    for node, neighbors in Ee.items():
        for neighbor in neighbors:
            edge = (min(int(node), neighbor), max(int(node), neighbor))
            if edge in edge_weights:
                dict_weights[node][neighbor] = edge_weights[edge]

    # Sort the keys and nested keys for each node
    sorted_weights = {str(k): {str(neighbor): weight for neighbor, weight in sorted(v.items())}
                      for k, v in sorted(dict_weights.items(), key=lambda x: int(x[0]))}

    return sorted_weights

# Sort the dictionary of weights and the nodes in Ee in decreasing order
def sortweights(dictionary_weights, Ee):
    sorted_weights = {}
    sorted_Ee = {}

    for node, neighbor_weights in dictionary_weights.items():
        # Sort weights by descending order of weight values
        sorted_neighbors = sorted(
            neighbor_weights.items(), key=lambda x: x[1], reverse=True)
        sorted_weights[node] = {k: v for k, v in sorted_neighbors}
        sorted_Ee[node] = list(sorted_weights[node].keys())

    # Convert values in sorted_Ee to integers
    int_E = {key: [int(neighbor) for neighbor in value]
             for key, value in sorted_Ee.items()}

    return sorted_weights, int_E

# Prepare input data
edges_with_weights = edge_data[['node1', 'node2', 'weight']].values.tolist()
weights = dictweights(E_e, edges_with_weights)  # Weights dictionary
weights, E_e = sortweights(weights, E_e)

# Save the sorted dictionaries
save_dict(weights, "Weights.json", file_path)
save_dict(E_e, "Endogenous_set.json", file_path)

def nmi_score(dict1, dict2):
    numpyArray_part = list(dict1.values())
    numpyArray_new_partition = list(dict2.values())
    nmi = normalized_mutual_info_score(
        numpyArray_part, numpyArray_new_partition)
    return nmi

# funtion to create a list with the nodes of a community
def nodes_of_community(diction, node):
    node = str(node)
    comm = diction[node]
    lst = []
    for key in diction.keys():
        if str(key) != node and diction[str(key)] == comm:
            lst.append(int(key))
    return lst

def sorensen_dice(list_post_interv, list_past_interv):
    list_1 = set(list_post_interv)
    list_2 = set(list_past_interv)
    DSC_score = (2*len(list_1.intersection(list_2))) / \
        (len(list_1) + len(list_2))
    return DSC_score

def binary_search(graph, node, cause, com_past, d_l, d_r, partit, interv_set, epsilon, threshold):
    """
    Perform binary search to find the optimal weight reduction for a counterfactual cause.
    """
    max_iter=1000
    DSC = 1  # Initialize DSC to ensure the loop starts
    iter_count = 0  # Safety counter
    
    while (d_r - d_l > epsilon) or (DSC > threshold): # Stop when both conditions are met
        iter_count += 1
        d_m = (d_l + d_r) / 2  # Midpoint
        if iter_count > max_iter:  # Prevent infinite loops
            print(f"Warning: Max iterations ({max_iter}) reached. Last d_m = {d_m}, DSC = {DSC}.")
            break
        
        graph[str(node)][str(cause)]['weight'] = d_m  # Apply intervention
        
        # Recalculate communities using the Louvain algorithm
        new_partition = Louvain(graph)
        
        # Find the nodes of communities after intervention
        community_post = nodes_of_community(new_partition, node)
            
        # Calculate Sorensen-Dice coefficient (DSC)
        DSC = sorensen_dice(com_past, community_post)

        if  DSC <= threshold:
            d_l = d_m  # Narrow down to the right
        else:
            d_r = d_m  # Narrow down to the left

    # Finalize results
    d_m = round(d_m, 6)
    optimal_weight_reduction = round(weights[str(node)][str(cause)] - d_m, 6)
    
    return optimal_weight_reduction, d_m, DSC

def find_counterf_causes(graph, partition, endogenous_edges, epsilon, threshold):
    """
    Discover counterfactual causes for every node in the network.
    """
    counterfactuals = {str(node): [] for node in range(1, len(endogenous_edges) + 1)}
    contingencies = {str(node): [] for node in range(1, len(endogenous_edges) + 1)}
    cause_details = []  # List of detailed counterfactual cause information

    num_communities = len(set(partition.values()))

    for node in range(1, len(endogenous_edges) + 1):
        node_str = str(node)
        neighbors = endogenous_edges[node_str]
        num_neighbors = len(neighbors)
        
        # Precompute community nodes
        comm_past = nodes_of_community(partition, node_str)
        
        # Check if node has only one neighbor (becomes isolated after intervention)
        if num_neighbors == 1:
            neighbor = neighbors[0]
            new_partition = deepcopy(partition)
            new_partition[node_str] = num_communities + 1  # Assign node to a new singleton community
            
            # NMI calculation
            nmi = nmi_score(partition, new_partition)
            
            # Make a copy of the graph before modifying weights
            graph_copy = deepcopy(graph)
            
            # Perform binary search to determine weight reduction
            opt_reduction, new_weight, DSC = binary_search(
                graph_copy, node_str, neighbor, comm_past, 0, weights[node_str][str(neighbor)], 
                partition, [(node_str, neighbor)], epsilon, threshold
            )

            # Store results
            responsibility = 1  # Fully responsible
            counterfactuals[node_str].append(neighbor)
            cause_details.append({
                "Node": node,
                "Cause": neighbor,
                "DSC score": 0.0,
                "NMI score": nmi,
                "Old Weight": weights[node_str][str(neighbor)],
                "New Weight": new_weight,
                "Weight Reduction": opt_reduction,
                "Responsibility score": responsibility
            })
            print(
                f"For node {node}, counterfactual cause found: {neighbor}, "
                f"responsibility = {responsibility}. Node {node} becomes isolated"
            )
            continue  # Skip further processing for this noders
        for neighbor in neighbors:
            edge = (node_str, str(neighbor))  # Edge to intervene
            w = weights[node_str][str(neighbor)]  # Save the initial weight

            # Make a copy of the graph before modifying weights
            graph_copy = deepcopy(graph)
            graph_copy.remove_edge(*edge)
            
            # Recalculate communities using the Louvain from igraph
            new_partition = Louvain(graph_copy)
            
            # Find the nodes of the communities after intervention
            comm_post = nodes_of_community(new_partition, node_str)
            
            # Calculate Sorensen-Dice coefficient (DSC)
            DSC_init = sorensen_dice(comm_past, comm_post)
            
            if DSC_init <= threshold:
                graph_copy.add_edge(*edge, weight=w)
                # Perform binary search to find optimal weight reduction
                opt_reduction, new_weight, DSC = binary_search(
                    graph_copy, node_str, neighbor, comm_past, 0, w, 
                    partition, [(node_str, neighbor)], epsilon, threshold
                )
                
                # NMI calculation
                nmi = nmi_score(partition, new_partition)

                # Ensure responsibility is well-defined
                responsibility = 1 / (1 + opt_reduction)
                
                # Store results
                counterfactuals[node_str].append(neighbor)
                cause_details.append({
                    "Node": node,
                    "Cause": neighbor,
                    "DSC score": DSC,
                    "NMI score": nmi,
                    "Old Weight": w,
                    "New Weight": new_weight,
                    "Weight Reduction": opt_reduction,
                    "Responsibility score": responsibility
                })
                print(
                    f"For node {node}, counterfactual cause found: {neighbor}, "
                    f"old weight = {w}, new weight = {new_weight}, "
                    f"reduction = {opt_reduction}, responsibility = {responsibility}"
                )

        # If no counterfactual cause was found, update contingency set
        if not counterfactuals[node_str]:
            contingencies[node_str] = endogenous_edges[node_str][1:]  

    return counterfactuals, contingencies, cause_details

# Execute and measure performance
start_time = time.time()
Counterf_causes,Cont_sets,count_causes=find_counterf_causes(G, part, E_e, epsilon, threshold)
elapsed=time.time() - start_time
elapsed=timedelta(seconds=elapsed)
time_for_countf_causes=elapsed
print("Execution time for finding counterfactual causes:",str(elapsed))

# Delete missing values
Counterf_causes = {k: v for k, v in Counterf_causes.items() if v}
Cont_sets = {k: v for k, v in Cont_sets.items() if v}

# Save the sorted dictionaries
save_dict(Counterf_causes, "Counterf_causes.json", file_path)
save_dict(count_causes, "count_causes.json", file_path)
save_dict(Cont_sets, "Cont_sets.json", file_path)

# Create contingencies with l edges at most
Cont_sets_ldim = deepcopy(Cont_sets)
for key in Cont_sets_ldim:
    if len(Cont_sets_ldim[key]) > l:
        del Cont_sets_ldim[key][l:]

# Intervention Function
def intervention(Gra, node_1, node_set, partit, threshold):
    """
    Perform an intervention by removing an edge and recalculating communities.
    """
    weight_of_interv_node = Gra[str(node_1)][str(node_set)]['weight']  # Save the weight of the edge to be removed
    community_past=nodes_of_community(partit,node_1)
    ebunch=[(str(node_1),str(node_set))] # Define the edge
    Gra.remove_edges_from(ebunch) #Intervention
    
    # Recalculate communities using the Louvain
    new_parttt = Louvain(Gra) 

    community_post=nodes_of_community(new_parttt,node_1)
    
    # Calculate Sorensen-Dice coefficient(DSC)
    DSC_score=sorensen_dice(community_past,community_post)
    
    # Undo intervention
    Gra.add_edge(str(node_1), str(node_set), weight=weight_of_interv_node)
    
    if DSC_score<=threshold:
        return True,community_past,new_parttt
    else:
        return False,community_past,new_parttt

def find_actual_causes(Gra, Endog_set, contingency, weights_set, parttt, threshold, epsilon):
    lst = []
    
    for node in Gra.nodes:
        if str(node) not in Counterf_causes.keys():
            e = Endog_set[str(node)][0]  # Define fixed node e
            print("Finding causes for node", node, "with actual cause", e)
            Dw_e = weights_set[str(node)][str(e)]  # Weight reduction of fixed node e
            
            # Make a copy of the graph before modifying weights
            graph_copy = deepcopy(Gra) 
            graph_copy.remove_edge(str(node), str(e))  # Max intervention on fixed node e
            
            max_dimensions = len(contingency[str(node)])
            max_resp = 0
            list_causes = {}

            for i in range(1, max_dimensions + 1):
                for comb in itertools.combinations(contingency[str(node)], i):  # Generate subsets of size i
                    removed_nodes = list(comb[:-1])  # Remove all except last node for binary search
                    binary_node = comb[-1]  # Last node is used for binary search

                    # Remove the selected nodes
                    removed_weights = {n: weights_set[str(node)][str(n)] for n in removed_nodes}
                    for n in removed_nodes:
                        graph_copy.remove_edge(str(node), str(n))

                    # Perform intervention on the last node
                    inter, comm_past, new_partittt = intervention(graph_copy, node, binary_node, parttt, threshold)
                    
                    if inter:
                        NMI = nmi_score(parttt, new_partittt)
                        ebunch = [(str(node), str(e))] + [(str(node), str(n)) for n in removed_nodes]
                        w = weights_set[str(node)][str(binary_node)]

                        optimal_red, n_weight, DSC = binary_search(graph_copy, node, binary_node, comm_past, 0, w, parttt, ebunch, epsilon, threshold)
                        graph_copy[str(node)][str(binary_node)]['weight'] = w  # Undo binary intervention

                        resp = 1 / (1 + Dw_e + sum(removed_weights.values()) + optimal_red)

                        if resp > max_resp:
                            max_resp = resp
                            list_causes = {
                                'Node': int(node),
                                'Fixed_node_e': e,
                                'Contingency set': list(comb),
                                'Removed nodes': removed_nodes,
                                'Binary': binary_node,
                                'DSC score': DSC,
                                'NMI score': NMI,
                                'Old Weight': weights[str(node)][str(binary_node)],
                                'New weight': n_weight,
                                'Weight reduction': optimal_red,
                                'Responsibility score': max_resp
                            }
                            lst.append(list_causes)
                    
                    # Undo interventions
                    for n, w in removed_weights.items():
                        graph_copy.add_edge(str(node), str(n), weight=w)

                if list_causes:  # If a valid cause is found, stop further searching
                    break

    return lst

def keep_max_resp(lst):
    max_resp = {}
    # Determine max responsibility score for each node
    for entry in lst:
        node = entry["Node"]
        if node not in max_resp or entry["Responsibility score"] > max_resp[node]["Responsibility score"]:
            max_resp[node] = entry
    # Return only the entries with max responsibility per node
    return list(max_resp.values())

# def to create a list with all the founded causes (counterfactual and actual) and a list with their NMI scores
def calc_causes_with_NMI_scores(list_cont, list_count):
    list_causes = []
    list_NMI_scores = []
    for i in range(len(list_cont)):  # iteration in actual causes set
        if list_cont[i]['Node'] not in list_causes:
            list_causes.append(list_cont[i]['Node'])
            list_NMI_scores.append(list_cont[i]['NMI score'])
    for i in range(len(list_count)):  # iteration in actual causes set
        if list_count[i]['Node'] not in list_causes:
            list_causes.append(list_count[i]['Node'])
            list_NMI_scores.append(list_count[i]['NMI score'])
    return list_causes, list_NMI_scores


def F1_calc(graph, total_causes, total_nmi, t1, t2): # Computation of F1: Success Rate(SR) and NMI
    Suc_R=len(total_causes)/len(graph.nodes) # Success Rate(SR)
    NMI_sc=sum(total_nmi)/len(total_nmi) # Sum up NMI scores of actual causes and normalize it
    F1_score=2*Suc_R*NMI_sc/(Suc_R+NMI_sc)
    tot_time = t1 + t2
    print(f'The F1 score for threshold t={threshold} and constant l={l}, is: F1 = {F1_score}. The Success Rate is: SR = {Suc_R}. '
          f'The total time for finding causes of every node of the graph is: Time = {str(tot_time)}'
                  ) 
    return F1_score, Suc_R, tot_time

# Execute and measure performance
start_time = time.time()    
cont_all = find_actual_causes(G, E_e, Cont_sets_ldim, weights, part, threshold,epsilon)
elapsed = time.time() - start_time
elapsed = timedelta(seconds=elapsed)
time_for_actual_causes=elapsed
print("Execution time for finding actual causes:", str(elapsed))

save_dict(cont_all, "Actual Causes.json", file_path)

count_causes_max_resp = keep_max_resp(count_causes) ## counterfactual causes list with max resp
cont_all_max_resp = keep_max_resp(cont_all) ## actual causes list with max resp
Founded_causes,NMI_scores=calc_causes_with_NMI_scores(cont_all,count_causes)
F1,SR,total_time=F1_calc(G,Founded_causes,NMI_scores,time_for_countf_causes,time_for_actual_causes)