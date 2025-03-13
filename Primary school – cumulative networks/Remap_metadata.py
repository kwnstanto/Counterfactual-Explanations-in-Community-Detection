# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 13:15:50 2025

@author: Konstanto
"""

# File paths
original_edges_path = "./Datasets/Primary school – cumulative networks/Primary School-Day 1(Duration).txt"
remapped_edges_path  = "./Datasets/Primary school – cumulative networks/Primary school – cumulative networks/Remapped_Nodes.txt"
metadata_path = "./Datasets/Primary school – cumulative networks/Primary school – cumulative networks/Metadata (Class and Gender).txt"
output_metadata_path = "./Datasets/Primary school – cumulative networks/Primary school – cumulative networks/Remapped_Metadata.txt"

# Step 1: Extract unique node IDs from original edge list
original_nodes = set()
with open(original_edges_path, "r") as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) == 3:  # Ensure it's an edge line
            node1, node2, _ = map(int, parts)
            original_nodes.update([node1, node2])

# Step 2: Extract unique node IDs from remapped edge list
remapped_nodes = set()
with open(remapped_edges_path, "r") as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) == 3:
            node1, node2, _ = map(int, parts)
            remapped_nodes.update([node1, node2])

# Step 3: Create a mapping from sorted original IDs to sorted remapped IDs
sorted_original_nodes = sorted(original_nodes)
sorted_remapped_nodes = sorted(remapped_nodes)
node_mapping = {old: new for old, new in zip(sorted_original_nodes, sorted_remapped_nodes)}

# Step 4: Read and remap metadata
remapped_metadata = []
with open(metadata_path, "r") as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) == 3:
            old_id, class_info, gender = parts
            old_id = int(old_id)
            if old_id in node_mapping:
                new_id = node_mapping[old_id]
                remapped_metadata.append(f"{new_id}\t{class_info}\t{gender}\n")

# Step 5: Save the remapped metadata to a new file
with open(output_metadata_path, "w") as file:
    file.writelines(remapped_metadata)

print(f"Remapped metadata saved to {output_metadata_path}")