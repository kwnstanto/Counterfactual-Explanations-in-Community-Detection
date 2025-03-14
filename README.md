# Counterfactual Explanations in Community Detection

## Requirements

- Python 3
- pandas
- networkx 
- igraph
- matplotlib
- community (for community_louvain)
- scikit-learn

## Directory Structure

-- Figures folder: contains all the images presented in the paper.

-- Datasets folder: contains all six datasets used for experiments
	- adjnoun_adjacency(words): out.adjnoun_adjacency_adjacency is the edge list file for unweighted words dataset
	- Facebook(fb-75): socfb-American75 is the edge list file for unweighted df-75 dataset
	- opsahl-powergrid(pow): out.opsahl-powergrid is the edge list file for unweighted power dataset
	- Primary school – cumulative networks: Primary School-Day 1(Duration) is the edge list file for weighted Primary school dataset. Metadata (Class and Gender) contains the ground truth data for the class and the gender of each kid
	- ucidata-zachary: out is the edge list file for weighted karate dataset. meta file contains the ground truth data.
	- vote: soc-wiki-Vote.mtx is the edge list file for unweighted vote dataset.

-- adjnoun_adjacency(words) folder: 
	-json_files folder: contains the saved dictionaries:
		- Partition: community structure found from greedy method
		- Neighbors: every each connections for each node (the neighborhood of each node)
		- Endogenous_set: the Ee set for each node. Contains every intra-community edge connection for each node.

	- t=0.3 folder: our results (for every node in the graph) for threshold τ=0.3 for every value l.
	- t=0.5 folder: our results (for every node in the graph) for threshold τ=0.5 for every value l.
	- t=0.8 folder: our results (for every node in the graph) for threshold τ=0.8 for every value l.

	words.py: the main python file of our code for word dataset. Testing for every node.
	
-- Facebook(fb-75) folder:
	-json_files folder: contains the saved dictionaries:
		- Partition: community structure found from greedy method
		- Neighbors: every each connections for each node (the neighborhood of each node)
		- Endogenous_set: the Ee set for each node. Contains every intra-community edge connection for each node.

	fb-75_100.py: the main python file of our code for fb-75 dataset. Testing for 100 random nodes. 

-- opsahl-powergrid(pow) folder:
	-json_files folder: contains the saved dictionaries:
		- Partition: community structure found from greedy method
		- Neighbors: every each connections for each node (the neighborhood of each node)
		- Endogenous_set: the Ee set for each node. Contains every intra-community edge connection for each node.

	- t=0.3 folder: our results (for every node in the graph) for threshold τ=0.3 for every value l.
	- t=0.5 folder: our results (for every node in the graph) for threshold τ=0.5 for every value l.
	- t=0.8 folder: our results (for every node in the graph) for threshold τ=0.8 for every value l.

	pow.py: the main python file of our code for power dataset. Testing every node.
	pow_100.py: the main python file of our code for power dataset. Testing for 100 random nodes.

-- Primary school – cumulative networks folder:
	-json_files folder: contains the saved dictionaries:
		- Partition: community structure found from Louvain method		
		- Neighbors: every each connections for each node (the neighborhood of each node)
		- Endogenous_set: the Ee set for each node. Contains every intra-community edge connection for each node.

	- t=0.3 folder: our results (for every node in the graph) for threshold τ=0.3 for every value l.
	- t=0.5 folder: our results (for every node in the graph) for threshold τ=0.5 for every value l.
	- t=0.8 folder: our results (for every node in the graph) for threshold τ=0.8 for every value l.
	
	- Remapped_Nodes: the resulted edge list after the remap of nodes.

	- Primary School-Day 1(Duration)_rescaled.txt: the edge_list with rescaled weights and remapped ids.

	- Primary_school.py: the main python file of our code for word dataset. Testing for every node.

	- Remap_metadata.py: python file to remap the ids of metadata file.

	- Remapped_Metadata: the resulted metadata file with remapped ids.

-- ucidata-zachary(kar) folder: 
 	-json_files folder: contains the saved dictionaries:
		- Partition: community structure found from greedy method
		- Neighbors: every each connections for each node (the neighborhood of each node)
		- Endogenous_set: the Ee set for each node. Contains every intra-community edge connection for each node.

	- t=0.3 folder: our results (for every node in the graph) for threshold τ=0.3 for every value l.
	- t=0.5 folder: our results (for every node in the graph) for threshold τ=0.5 for every value l.
	- t=0.8 folder: our results (for every node in the graph) for threshold τ=0.8 for every value l.

	Karate-unweighted.py: the main python file of our code for Karate(unweighted) dataset. Testing every node.

-- ucidata-zachary(weighted) folder: 
	-json_files folder: contains the saved dictionaries:
		- Partition: community structure found from Louvain method
		- Neighbors: every each connections for each node (the neighborhood of each node)
		- Endogenous_set: the Ee set for each node. Contains every intra-community edge connection for each node.

	- t=0.3 folder: our results (for every node in the graph) for threshold τ=0.3 for every value l.
	- t=0.5 folder: our results (for every node in the graph) for threshold τ=0.5 for every value l.
	- t=0.8 folder: our results (for every node in the graph) for threshold τ=0.8 for every value l.

	Karate-weighted.py: the main python file of our code for Karate(weigthed) dataset. Testing every node.

-- vote folder:
	-json_files folder: contains the saved dictionaries:
		- Partition: community structure found from Louvain method
		- Neighbors: every each connections for each node (the neighborhood of each node)
		- Endogenous_set: the Ee set for each node. Contains every intra-community edge connection for each node.
		- Weights: contains the weights of every neighbor for every node of the network. It's sorted in descending order.

	- t=0.3 folder: our results (for every node in the graph) for threshold τ=0.3 for every value l.
	- t=0.5 folder: our results (for every node in the graph) for threshold τ=0.5 for every value l.
	- t=0.8 folder: our results (for every node in the graph) for threshold τ=0.8 for every value l.

	vote.py: the main python file of our code for vote dataset. Testing every node.
	vote_100.py: the main python file of our code for vote dataset. Testing for 100 random nodes.



## Run the code

To experiment with a dataset, just open the python file in the the appropriate folder, fix the paths and set the diserid parameters for threshold τ, interventions β and error epsilon (only in weigthed case) and hit run! 

