#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:57:07 2025

@author: Konstanto
"""

import pandas as pd
import networkx as nx
import itertools
import random
import csv
import igraph as ig
from datetime import timedelta
import time
import json
import os
from copy import deepcopy
from sklearn.metrics.cluster import normalized_mutual_info_score

file_path = r".\Datasets\opsahl-powergrid(pow)\out.opsahl-powergrid"
column_names = ["node1", "node2"] # Skip the first two lines (header and metadata) and load the actual edge data into a DataFrame
edge_data = pd.read_csv(file_path, delim_whitespace=True, skiprows=2, names=column_names)

edge_data.to_csv(r'.\Datasets\opsahl-powergrid(pow)\out.opsahl-powergrid.txt', header=None, index=None, sep=' ', quoting=csv.QUOTE_NONE, escapechar=' ') 
# Create a graph using NetworkX from the rescaled edge data
G = nx.read_edgelist(r'.\Datasets\opsahl-powergrid(pow)\out.opsahl-powergrid.txt', create_using=nx.Graph())
## create file for Gephi
nx.write_gexf(G, r".\Datasets\opsahl-powergrid(pow)\out.opsahl-powergrid.gexf")

# Set the threshold τ={0.3,0.5,0.8}
threshold=0.5
# Set the constant l={μ/2,μ,2μ}
m=len(G.edges)/len(G.nodes)
l=round(2*m)-1 # [0,1,3] 

def com_det_tranform_list_to_diction(lst): 
    temp={}
    for i in range(len(lst)):
        for j in lst[i]:
            temp[str(j)] = i
    return temp

# Function to perform community detection using the Greedy method
def Greedy(Graph):
    original_ids = list(Graph.nodes())
    id_mapping = {new_id:old_id  for new_id,old_id  in enumerate(original_ids)} # store the mapping between nodes
    G_ig = ig.Graph.from_networkx(Graph) # Tranform NetworkX graph to iGraph
    com_det = G_ig.community_fastgreedy()
    com_list = [c for c in com_det.as_clustering()] # converted to VertexClustering object
    communities=com_det_tranform_list_to_diction(com_list) # transform the list to proper dictionary
    communities_updated = {id_mapping[int(k)]: v for k, v in communities.items()}
    
    return communities_updated

# Compute the initial community structure between Greedy, Louvain, Walktrap
part=Greedy(G)

# Save dictionaries as JSON
def save_dict(d, filename,path):
   # Combine the path and filename into a full file path
    filepath = os.path.join(path, filename)
    with open(filepath, 'w') as f:
        json.dump(d, f, indent=4)
    # print(f"Saved dictionary to {filepath}")
   
file_path = r".\Datasets\opsahl-powergrid(pow)"
save_dict(part, "Partition.json",file_path)

# Prepare the edge list from the DataFrame
edge_list = edge_data[['node1', 'node2']].values.tolist()

# Create a dictionary with incident edges
def conn_edges(lst):
    neigh_dict = {}
    for l in lst:
        node1, node2 = int(l[0]), int(l[1])
        neigh_dict.setdefault(str(node1), []).append(node2)
        neigh_dict.setdefault(str(node2), [])  # Ensure the second node exists in the dictionary
    return neigh_dict

# Fill any missing values to ensure symmetry in connections
def misval(dct):
    for node, neighbors in dct.items():
        for neighbor in neighbors:
            if node not in dct[str(neighbor)]:
                dct[str(neighbor)].append(int(node))
    
    # Convert all neighbor values to integers for consistency
    for key, value in dct.items():
        dct[key] = sorted(set(value))  # Remove duplicates and sort for consistency
    return dct

# Reorder the dictionary by keys in ascending order
def sort_neighbors_dict(dct):
    return {str(key): dct.get(str(key), []) for key in range(1,4942)}

# Process the edge list into a dictionary
neighbors_dict = conn_edges(edge_list)
neighbors_dict = misval(neighbors_dict)
neighbors_dict = sort_neighbors_dict(neighbors_dict) # Sort the dictionary
save_dict(neighbors_dict, "Neighbors.json", file_path)

# Function to find the Endogenous (Ee) and Exogenous (Ex) sets
def endog(edges, part, num_nodes=4941):
    Ee = {}  # Endogenous edges as a regular dictionary
    Ex = {}  # Exogenous edges as a regular dictionary

    # Initialize all nodes to have empty lists
    for node in map(str, range(1, num_nodes + 1)):
        Ee[node] = []
        Ex[node] = []

    # Process each edge
    for node1, node2 in edges:
        node1, node2 = str(node1), str(node2)  # Use string keys to match `part` dictionary
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

# Save the sorted dictionaries
save_dict(E_e, "Endogenous_set.json", file_path)

def nmi_score(dict1,dict2):
    numpyArray_part=list(dict1.values())
    numpyArray_new_partition=list(dict2.values())
    nmi=normalized_mutual_info_score(numpyArray_part,numpyArray_new_partition)
    return nmi

def nodes_of_community(diction,node): # funtion to create a list with the nodes of a community
    node=str(node)
    comm=diction[node]
    lst=[]
    for key in diction.keys():
        if str(key)!=node and diction[str(key)]==comm:
            lst.append(int(key))
    return lst
    
def sorensen_dice(list_post_interv, list_past_interv):
    list_1=set(list_post_interv)
    list_2=set(list_past_interv)
    DSC_score=(2*len(list_1.intersection(list_2))) / (len(list_1) + len(list_2))
    return DSC_score
        
def find_counterf_causes_100(graph, partition, endogenous_edges, thresh):
    """
    Discover counterfactual causes for nodes in the network.
    """
    counterfactuals = {}  # Counterfactual causes dictionary
    contingencies = {}  # Contingencies dictionary
    cause_details = []  # List of detailed counterfactual cause information

    num_communities = len(set(partition.values()))

    for node in endogenous_edges:
        node_str = str(node)
        counterfactuals[node_str] = []
        contingencies[node_str] = []

        neighbors = endogenous_edges[node_str]
        num_neighbors = len(neighbors)

        # Precompute community nodes
        comm_post = nodes_of_community(partition, node_str)

        # Nodes with one neighbor (becoming isolated after intervention)
        if num_neighbors == 1:
            neighbor = neighbors[0]
            new_partition = deepcopy(partition)
            singleton_comm = num_communities + 1  # Assign node to a new singleton community
            new_partition[node_str] = singleton_comm

            # NMI calculation
            nmi = nmi_score(partition, new_partition)

            # Store results
            responsibility = 1
            counterfactuals[node_str].append(neighbor)
            cause_details.append({
                "Node": node,
                "Cause": neighbor,
                "DSC score": 0,
                "NMI score": nmi,
                "Responsibility score": responsibility
            })
            continue  # Skip further processing for this node
        else:
            for neighbor in neighbors:
                # Intervention: remove edge
                edge = (node_str, str(neighbor))
                graph.remove_edge(*edge)
    
                # Recalculate communities using the Greedy algorithm
                new_partition = Greedy(graph)
               
                # Find the nodes of the community after intervention
                comm_past = nodes_of_community(new_partition, node_str)
    
                # Calculate Sorensen-Dice coefficient (DSC)
                DSC = sorensen_dice(comm_post, comm_past)
    
                if DSC <= thresh:
                    # NMI calculation
                    nmi = nmi_score(partition, new_partition)
    
                    # Store results
                    responsibility = 1
                    counterfactuals[node_str].append(neighbor)
                    cause_details.append({
                        "Node": node,
                        "Cause": neighbor,
                        "DSC score": DSC,
                        "NMI score": nmi,
                        "Responsibility score": responsibility
                    })
                    graph.add_edge(*edge)  # Restore the original edge
                    break  # If a cause is found, stop the iteration
    
                graph.add_edge(*edge)  # Restore the original edge
            if len(counterfactuals[node_str])==0: # No counterfactual cause was found
                contingencies[node_str]=endogenous_edges[node_str][1:] # Contingency set=E_e - fixed node (removed later in def actual causes) 
                
    return counterfactuals, contingencies, cause_details  
      
# Intervention Function
def intervention(Gra, node_1, edges_to_remove, partit, communities_post, thresh):
    """
    Perform an intervention by removing an edge and recalculating communities.
    """
    Gra.remove_edges_from(edges_to_remove)
    
    # Greedy
    new_parttt = Greedy(Gra)

    community_past=nodes_of_community(new_parttt,node_1)
    
    # Calculate Sorensen-Dice coefficient(DSC)
    DSC_score=sorensen_dice(communities_post,community_past)
    # Undo intervention
    Gra.add_edges_from(edges_to_remove)
    
    if DSC_score<=thresh:
        return True, DSC_score, new_parttt
    else:
        return False, DSC_score, new_parttt
    
def find_actual_causes(Gra, Endog_set, contingency, parttt, thresh): 
    lst = []
    for node in contingency.keys():
        e = Endog_set[str(node)][0]  # define fixed node e
        Gra.remove_edge(str(node), str(e))  # intervention of fixed node e (max reduction)
        max_dimensions = len(contingency[str(node)])
        list_causes = {}
        community_post=nodes_of_community(parttt,node)  # Precompute community nodes   
        
        for i in range(1, max_dimensions + 1):
            combinations = list(itertools.combinations(contingency[str(node)], i))  # produce all possible i-tuples
            for comb in combinations:  # `comb` is a tuple with `i` elements
                ebunch = [(str(node), str(n)) for n in comb]  # Create edges for all nodes in the combination
                inter, DSC, new_partittt = intervention(Gra, node, ebunch, parttt, community_post, thresh)
                    
                if inter:
                    NMI = nmi_score(parttt, new_partittt)  # Calculate NMI
                    resp = 1 / (1 + i)
                    list_causes = {
                        'Node': int(node),
                        'Actual cause': e,
                        'Contingency set': list(comb),  # Convert tuple to list
                        'DSC score': DSC,
                        'NMI score': NMI,
                        'Responsibility score': resp
                        }
                    lst.append(list_causes)
                    # Stop searching for causes for this node if at least one cause is found
                    break                
            if list_causes:  # If a cause is found, stop further combinations
                Gra.add_edge(str(node), str(e))  # Undo the intervention
                break

        # Undo the intervention
        if not list_causes:
            Gra.add_edge(str(node), str(e))
    
    return lst

# def to create a list with all the founded causes (counterfactual and actual) and a list with their NMI scores
def calc_causes_with_NMI_scores(list_cont,list_count):      
    list_causes=[]
    list_NMI_scores=[]
    for i in range(len(list_cont)): # iteration in actual causes set
        if list_cont[i]['Node'] not in list_causes:  
                list_causes.append(list_cont[i]['Node']) 
                list_NMI_scores.append(list_cont[i]['NMI score'])   
    for i in range(len(list_count)): # iteration in counterfactual causes set
        if list_count[i]['Node'] not in list_causes:  
                list_causes.append(list_count[i]['Node']) 
                list_NMI_scores.append(list_count[i]['NMI score'])   
    return list_causes,list_NMI_scores


def F1_calc_100(graph, total_causes, total_nmi, t1, t2):
    # Computation of F1: Success Rate(SR) and NMI
    Suc_R=len(total_causes)/100#  Success Rate(SR)
    NMI_sc=sum(total_nmi)/100 # Sum up NMI scores of actual causes and normalize it
    F1_score=2*Suc_R*NMI_sc/(Suc_R+NMI_sc)
    total_time = t1 + t2

    return F1_score,Suc_R,total_time

# Pick randomly 100 nodes from Graph
nodes_to_list=list(G.nodes)
nodes_100 = random.sample(nodes_to_list, 100)
E_e_100={}
for i in nodes_100:
    E_e_100[i]=E_e[i]

start_time = time.time()
Counterf_causes_100,Cont_sets_100,count_causes_100,comm_det_method_past=find_counterf_causes_100(G,part,E_e_100, threshold)
elapsed=time.time() - start_time
elapsed=timedelta(seconds=elapsed)
time_for_countf_causes=elapsed

# Delete missing values
Counterf_causes_100={k: v for k, v in Counterf_causes_100.items() if v}
Cont_sets_100={k: v for k, v in Cont_sets_100.items() if v}
    
# Create contingencies with l edges at most
Cont_sets_ldim_100=deepcopy(Cont_sets_100)
for key in Cont_sets_ldim_100:
    if len(Cont_sets_ldim_100[key])>l:
        del Cont_sets_ldim_100[key][l:]   
        
start_time = time.time()    
cont_all_100=find_actual_causes(G, E_e_100, Cont_sets_ldim_100, part, threshold)
elapsed = time.time() - start_time
elapsed = timedelta(seconds=elapsed)
time_for_actual_causes=elapsed
print("Execution time for finding actual causes:", str(elapsed))

Founded_causes,NMI_scores=calc_causes_with_NMI_scores(cont_all_100,count_causes_100)
F1,SR,tot_time=F1_calc_100(G,Founded_causes,NMI_scores,time_for_countf_causes,time_for_actual_causes)
print(f'l = {l}, tau = {threshold}, F1 = {F1}, SR = {SR} and the total time is: {tot_time}')

"FOR RUNNING 100 EXPERIMENTS WITH 100 RANDOM NODES EACH TIME TO GET F1,SR,TIME RATIOS"
# SR_list=[]
# F1_list=[]
# t_list=[]
# for i in range(1,101):  
#     # Pick randomly 100 nodes from Graph
#     nodes_to_list=list(G.nodes)
#     nodes_100 = random.sample(nodes_to_list, 100)
#     E_e_100={}
#     for i in nodes_100:
#         E_e_100[i]=E_e[i]
    
#     start_time = time.time()
#     Counterf_causes_100,Cont_sets_100,count_causes_100,comm_det_method_past=find_counterf_causes_100(G,part,E_e_100, threshold)
#     elapsed=time.time() - start_time
#     elapsed=timedelta(seconds=elapsed)
#     time_for_countf_causes=elapsed
    
#     # Delete missing values
#     Counterf_causes_100={k: v for k, v in Counterf_causes_100.items() if v}
#     Cont_sets_100={k: v for k, v in Cont_sets_100.items() if v}
        
#     # Create contingencies with l edges at most
#     Cont_sets_ldim_100=deepcopy(Cont_sets_100)
#     for key in Cont_sets_ldim_100:
#         if len(Cont_sets_ldim_100[key])>l:
#             del Cont_sets_ldim_100[key][l:]   
            
#     start_time = time.time()    
#     cont_all_100=find_actual_causes(G, E_e_100, Cont_sets_ldim_100, part, threshold)
#     elapsed = time.time() - start_time
#     elapsed = timedelta(seconds=elapsed)
#     time_for_actual_causes=elapsed
#     # print("Execution time for finding actual causes:", str(elapsed))
#     Founded_causes,NMI_scores=calc_causes_with_NMI_scores(cont_all_100,count_causes_100)
#     F1,SR,tot_time=F1_calc_100(G,Founded_causes,NMI_scores,time_for_countf_causes,time_for_actual_causes)
#     F1_list.append(F1)
#     SR_list.append(SR)
#     t_list.append(tot_time)

# F1_ratio=0 
# SR_ratio=0 
# t_ratio=timedelta(0)

# for i in range(0,100):
#     F1_ratio+=F1_list[i]
#     SR_ratio+=SR_list[i]
#     t_ratio+=t_list[i]
    
# F1_ratio,SR_ratio,t_ratio=F1_ratio/100,SR_ratio/100,t_ratio/100
# print(f'l = {l}, tau = {threshold}, F1_ratio = {F1_ratio}, SR_ratio = {SR_ratio} and the total time is: {t_ratio}')
