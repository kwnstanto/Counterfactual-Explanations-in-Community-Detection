# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:25:54 2025

@author: Konstanto
"""

import pandas as pd
import networkx as nx
import itertools
import csv
import igraph as ig
from community import community_louvain
from datetime import timedelta
import time
import json
import os
from copy import deepcopy
from sklearn.metrics.cluster import normalized_mutual_info_score

file_path = os.path.join(os.path.expanduser('~'), 'documents', 'python', '/home/pclab09/Downloads/Comparison/opsahl-powergrid(pow)/out.opsahl-powergrid')
column_names = ["node1", "node2"] # Skip the first two lines (header and metadata) and load the actual edge data into a DataFrame
edge_data = pd.read_csv(file_path, delim_whitespace=True, skiprows=2, names=column_names)

edge_data.to_csv(r'/home/pclab09/Downloads/Comparison/opsahl-powergrid(pow)\out.opsahl-powergrid.txt', header=None, index=None, sep=' ', quoting=csv.QUOTE_NONE, escapechar=' ') 
# Create a graph using NetworkX from the rescaled edge data
G = nx.read_edgelist(r'/home/pclab09/Downloads/Comparison/opsahl-powergrid(pow)\out.opsahl-powergrid.txt', create_using=nx.Graph())
## create file for Gephi
nx.write_gexf(G, r"/home/pclab09/Downloads/Comparison/opsahl-powergrid(pow)\out.opsahl-powergrid.gexf")

# Set the threshold τ={0.3,0.5,0.8}
threshold=0.8
# Set the constant l={μ/2,μ,2μ}
m=len(G.edges)/len(G.nodes)
l=round(2*m)-1 # l=[0,1,3]

# Perform community detection using the Louvain method
# comm_det_method= 'Louvain'

# # Open and read the JSON file
# with open(r'F:\Users\Xristos\Downloads\Math\PhD\Datasets\Unweighted\Comparison\vote\Partition-Louvain.json', 'r') as file:
#     part = json.load(file) 

# Perform community detection using the Greedy method
comm_det_method= 'Greedy'

# Open and read the JSON file
with open(r'/home/pclab09/Downloads/Comparison/opsahl-powergrid(pow)\Partition-Greedy.json', 'r') as file:
    part = json.load(file) 

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
    start_time = time.time() 
    com_det = G_ig.community_fastgreedy()
    elapsed = str(timedelta(seconds=(time.time() - start_time)))
    com_list = [c for c in com_det.as_clustering()] # converted to VertexClustering object
    communities=com_det_tranform_list_to_diction(com_list) # transform the list to proper dictionary
    communities_updated = {id_mapping[int(k)]: v for k, v in communities.items()}
    
    return communities_updated

# Function to perform community detection using the Louvain method
def Louvain(Graph):
    start_time = time.time() 
    communities = community_louvain.best_partition(Graph, randomize=False)
    elapsed = str(timedelta(seconds=(time.time() - start_time)))
    # print("Elapsed Time of Louvain method:", elapsed)
    return communities

def Louvain_ig(Graph):
    original_ids = list(Graph.nodes())
    id_mapping = {new_id:old_id  for new_id,old_id  in enumerate(original_ids)} # store the mapping between nodes
    G_ig = ig.Graph.from_networkx(Graph) # Tranform NetworkX graph to iGraph
    start_time = time.time() 
    com_det = G_ig.community_multilevel()
    elapsed = str(timedelta(seconds=(time.time() - start_time)))
    com_list = [c for c in com_det]
    communities=com_det_tranform_list_to_diction(com_list) # transform the list to proper dictionary
    communities_updated = {id_mapping[int(k)]: v for k, v in communities.items()}
    # print("Elapsed Time of Louvain method:", elapsed)
    return communities_updated

# Function to perform community detection using the Walktrap method
def Walktrap(Graph):
    
    original_ids = list(Graph.nodes())
    id_mapping = {new_id:old_id  for new_id,old_id  in enumerate(original_ids)} # store the mapping between nodes
    G_ig = ig.Graph.from_networkx(Graph) # Tranform NetworkX graph to iGraph
    start_time = time.time() 
    com_det = G_ig.community_walktrap()
    elapsed = str(timedelta(seconds=(time.time() - start_time)))
    com_list = [c for c in com_det.as_clustering()] # converted to VertexClustering object
    communities=com_det_tranform_list_to_diction(com_list) # transform the list to proper dictionary
    communities_updated = {id_mapping[int(k)]: v for k, v in communities.items()}
 
    return communities_updated
# Save dictionaries as JSON
def save_dict(d, filename,path):
   # Combine the path and filename into a full file path
   filepath = os.path.join(path, filename)
   with open(filepath, 'w') as f:
       json.dump(d, f, indent=4)
   print(f"Saved dictionary to {filepath}")
   
file_path = r"/home/pclab09/Downloads/Comparison/opsahl-powergrid(pow)"
    
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

# Open and read the JSON file
with open(r'/home/pclab09/Downloads/Comparison/opsahl-powergrid(pow)\t=0.5\Greedy to Greedy\Endogenous set-Greedy.json', 'r') as file:
    E_e = json.load(file) 
    
# Open and read the JSON file
with open(r'/home/pclab09/Downloads/Comparison/opsahl-powergrid(pow)\t=0.3\Greedy to Greedy\Counterf_causes.json', 'r') as file:
    Counterf_causes = json.load(file)   
    
# Open and read the JSON file
with open(r'/home/pclab09/Downloads/Comparison/opsahl-powergrid(pow)\t=0.3\Greedy to Greedy\Cont_sets.json', 'r') as file:
    Cont_sets = json.load(file)
    
# Open and read the JSON file
with open(r'/home/pclab09/Downloads/Comparison/opsahl-powergrid(pow)\t=0.3\Greedy to Greedy\Count_causes.json', 'r') as file:
    count_causes = json.load(file)        

## Create contingencies with l edges at most
Cont_sets_ldim=deepcopy(Cont_sets)
for key in Cont_sets_ldim:
    if len(Cont_sets_ldim[key])>l:
        del Cont_sets_ldim[key][l:]   

# Intervention Function
def intervention(Gra, node_1, edges_to_remove, partit, thresh):
    """
    Perform an intervention by removing an edge and recalculating communities.
    """
    Gra.remove_edges_from(edges_to_remove)
    
    # Louvain
    comm_det_past = 'Louvain'
    new_parttt=Louvain_ig(Gra)
    
    # Greedy
    # comm_det_method_past = 'Greedy'
    # new_parttt = Greedy(Gra)
    
    # # Walktrap
    # comm_det_method_past = 'Walktrap'
    # new_parttt = Walktrap(Gra)

    community_post=nodes_of_community(partit,node_1)
    community_past=nodes_of_community(new_parttt,node_1)
    
    # Calculate Sorensen-Dice coefficient(DSC)
    DSC_score=sorensen_dice(community_post,community_past)
    
    if DSC_score<=thresh:
        # Undo intervention
        Gra.add_edges_from(edges_to_remove)
        return True, DSC_score, new_parttt, comm_det_past
    else:
        # Undo intervention
        Gra.add_edges_from(edges_to_remove)
        return False, DSC_score, new_parttt, comm_det_past

def find_actual_causes(Gra, Endog_set, contingency, parttt, thresh): 
    lst = []
    for node in contingency.keys():
        e = Endog_set[str(node)][0]  # define fixed node e
        print("Finding causes for node", node, "with actual cause", e)
        Gra.remove_edge(str(node), str(e))  # intervention of fixed node e (max reduction)
        max_dimensions = len(contingency[str(node)])
        list_causes = {}
            
        for i in range(1, max_dimensions + 1):
            combinations = list(itertools.combinations(contingency[str(node)], i))  # produce all possible i-tuples
            for comb in combinations:  # `comb` is a tuple with `i` elements
                ebunch = [(str(node), str(n)) for n in comb]  # Create edges for all nodes in the combination
                inter, DSC, new_partittt, comm_detect_method_past = intervention(Gra, node, ebunch, parttt, thresh)
                    
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
    
    return lst, comm_detect_method_past

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

# Computation of F1: Success Rate(SR) and NMI
def F1_calc(graph, total_causes, total_nmi, c_det_method_past, t):
    # Computation of F1: Success Rate(SR) and NMI
    Suc_R=len(total_causes)/len(graph.nodes) # Success Rate(SR)
    NMI_sc=sum(total_nmi)/len(total_nmi) # Sum up NMI scores of actual causes and normalize it
    F1_score=2*Suc_R*NMI_sc/(Suc_R+NMI_sc)
    print(f'Testing the setup Greedy to {c_det_method_past}. The F1 score for threshold t={threshold} and constant l={l}, is: F1 = {F1_score}. The Success Rate is: SR = {Suc_R}. '
          f'The total time for finding causes of every node of the graph is: Time = {str(t)}'
                  )  
start_time = time.time()    
cont_all,c_det_method_past=find_actual_causes(G, E_e, Cont_sets_ldim, part, threshold)
elapsed = time.time() - start_time
elapsed = timedelta(seconds=elapsed)
time_for_actual_causes=elapsed
print("Execution time for finding actual causes:", str(elapsed))
Founded_causes,NMI_scores=calc_causes_with_NMI_scores(cont_all,count_causes)
F1=F1_calc(G,Founded_causes,NMI_scores,c_det_method_past,time_for_actual_causes)
save_dict(cont_all, "Actual_Causes.json", file_path)