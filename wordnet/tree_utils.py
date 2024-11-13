import os
import sys
import pickle
import numpy as np
import networkx as nx
from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')
import matplotlib.pyplot as plt
import random
import time
from tqdm import tqdm
from datetime import datetime
from adjustText import adjust_text
import copy

def load_semantic_tree(file_path):
    """Load the semantic tree from a specified file path."""
    return nx.read_gexf(file_path)

def to_offset_label(node):
    """Convert a synset name to offsets (i.e., ImageNet file ID)."""
    try:
        # synset = wn.synset(node.split("'")[1])
        first_quote_index = node.find("'")
        last_quote_index = node.rfind("'")
        if first_quote_index == -1 or last_quote_index == -1 or first_quote_index == last_quote_index:
            raise ValueError("Invalid node format")
        synset_name = node[first_quote_index + 1:last_quote_index]
        synset = wn.synset(synset_name)
        return synset.pos() + str(synset.offset()).zfill(8)
    except ValueError as e:
        print(f'Error processing node: {node}')
        print(e)
        return None
    
def search_pkl_files_all_model(children):
    """Search for corresponding pkl files based on a list of children nodes."""
    not_found_children = []
    pkl_files = [f for f in os.listdir('/n/holylabs/LABS/sompolinsky_lab/Everyone/xupan/hierarchy_manifold/image_score/') if f.endswith(".pkl")]

    for child in children:
        found = False
        for pkl_file in pkl_files:
            if child in pkl_file:
                found = True
        if not found:
            not_found_children.append(child)
    
    return not_found_children

def search_pkl_files(children, pkl_files):
    """Search for corresponding pkl files based on a list of children nodes."""
    matching_files = {}
    not_found_children = []

    for child in children:
        found = False
        for pkl_file in pkl_files:
            if child in pkl_file:
                matching_files.setdefault(child, []).append(pkl_file)
                found = True
        if not found:
            not_found_children.append(child)
    
    return matching_files, not_found_children

def find_children(node, semantic_tree):
    """Find the children of a specified node in the semantic tree."""
    return list(semantic_tree.successors(node))

def check_combined_data_size(combined_data):
    """Check the size of the combined data."""
    for key, value in combined_data.items():
        print(f"Layer: {key}, Shape: {value.shape}")

def draw_and_save_tree(semantic_tree, subgraph_nodes, subgraph_edges, save_dir, node_colors, timestamp):
    """Draw and save the semantic hierarchy tree."""
    subgraph = semantic_tree.subgraph(subgraph_nodes).copy()
    subgraph.add_edges_from(subgraph_edges)
    
    plt.figure(figsize=(12, 6))
    pos = nx.nx_agraph.graphviz_layout(subgraph, prog="dot")
    labels = {node: node.split("'")[1].split('.')[0].replace('_', ' ') for node in subgraph.nodes()}
    colors = [node_colors.get(node, "skyblue") for node in subgraph.nodes()]
    nx.draw(subgraph, pos, node_size=200, node_color=colors, font_size=2, font_weight="bold", edge_color="gray")
    
    texts = [plt.text(x, y, labels[node], fontsize=10, ha='center', va='center', rotation=90) for node, (x, y) in pos.items()]
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray'))
    
    plt.title("Binary Semantic Hierarchy Tree", fontsize=16, fontweight='bold')
    plt.figtext(0.5, 0.01, "\n The yellow nodes are the last leaf nodes used for data combination.\n The blue nodes are used as classes in manifold calculation.", wrap=True, horizontalalignment='center', fontsize=8)
    plt.savefig(os.path.join(save_dir, f"generate_semantic_hierarchy_tree_{timestamp}.svg"))
    

def shuffle_children(all_children, permutation_strength, s, l, n, b):
    """Shuffle the children nodes based on a specified permutation strength and shuffle layers."""
    if s >= l:
        raise ValueError("Shuffle layers (s) must be less than tree depth (l)")

    num_to_shuffle = int(len(all_children) * permutation_strength)
    if num_to_shuffle < b:
        print('Permutation strength is too low to shuffle children nodes')
        return all_children, 0

    # Calculate the number of partitions
    if s == 0:
        num_partitions = 1
    else:
        num_partitions = n * (b ** (s - 1))
    
    partition_size = len(all_children) // num_partitions

    # Shuffle within each partition
    shuffled_children = []
    # 2024-9-16: Changed the shuffle method for s==1 to shuffle all children directly
    if s == 0:
        # Shuffle all children directly
        shuffled_children = all_children.copy()
        indices_to_shuffle = random.sample(range(len(all_children)), num_to_shuffle)
        children_to_shuffle = [all_children[i] for i in indices_to_shuffle]
        random.shuffle(children_to_shuffle)
        for idx, child in zip(indices_to_shuffle, children_to_shuffle):
            shuffled_children[idx] = child
    else:
        for i in range(num_partitions):
            start_idx = i * partition_size
            end_idx = (i + 1) * partition_size if i < num_partitions - 1 else len(all_children)
            partition = all_children[start_idx:end_idx]

            indices_to_shuffle = random.sample(range(len(partition)), min(num_to_shuffle, len(partition)))
            children_to_shuffle = [partition[i] for i in indices_to_shuffle]
            random.shuffle(children_to_shuffle)
            for idx, child in zip(indices_to_shuffle, children_to_shuffle):
                partition[idx] = child

            shuffled_children.extend(partition)

    return shuffled_children, permutation_strength

def shuffle_across_keys(all_children, pkl_files, pkl_dir, num_im):
    """Combine data from pkl files, shuffle row vectors within each key, and reassign them back."""
    combined_data = {}
    original_sizes = {}

    # print('all_children:', len(all_children), all_children)
    # Combine data from pkl files
    for child in all_children:
        child_label = to_offset_label(child)
        for pkl_file in pkl_files:
            if child_label in pkl_file:
                with open(os.path.join(pkl_dir, pkl_file), 'rb') as f:
                    data = pickle.load(f)
                    for key, value in data.items():
                        combined_data.setdefault(key, []).append(value[:num_im])
                        original_sizes.setdefault(key, []).append(value[:num_im].shape[0])

    # print('Original sizes:', len(original_sizes))
    # print('Combined data keys:', combined_data.keys())
    # for key in combined_data:
    #     print(f'Key: {key}, Shape: {len(combined_data[key]), combined_data[key][0].shape}')
    
    # Concatenate values for each key
    for key, value in combined_data.items():
        combined_data[key] = np.concatenate(value, axis=0)
    
    # Shuffle row vectors within each key
    for key in combined_data:
        rows = list(combined_data[key])
        random.shuffle(rows)
        combined_data[key] = np.array(rows)
    # print('Combined shuffle data keys:', combined_data.keys())
    # for key in combined_data:
    #     print(f'Key: {key}, Shape: {len(combined_data[key]), combined_data[key][0].shape}')
    
    # Split combined data back to original sizes
    split_data = {}
    for key, value in combined_data.items():
        split_data[key] = []
        start = 0
        for size in original_sizes[key]:
            split_data[key].append(value[start:start + size])
            start += size
            
    # # 2024-9-16: check split_data dimension
    # print('check split_data:', len(split_data), split_data.keys(), len(split_data['features1']), split_data['features1'][0].shape)
    return {key: np.concatenate(split_data[key], axis=0) for key in split_data}

def generate_binary_tree(semantic_tree, n=5, l=1, branch=2, draw_only=False):
    """Generate binary tree structures from the semantic tree based on given parameters."""
    non_leaf_nodes = [node for node in semantic_tree.nodes() if semantic_tree.out_degree(node) > 0]
    valid_fathers = []
    subgraph_nodes = []
    subgraph_edges = []
    father_leaf_counts = {}
    node_colors = {}
    all_final_children = []
    second_last_layer_nodes = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with tqdm(total=n, desc="Generating binary trees") as pbar:
        while len(valid_fathers) < n:
            father = random.choice(non_leaf_nodes)
            
            # check if the father node has been generated
            if father in valid_fathers or father in subgraph_nodes:
                continue
            
            # 2024-11-2: check if the father node is child of existing father
            is_child_of_existing_father = False
            for existing_father in valid_fathers:
                if nx.has_path(semantic_tree, existing_father, father):
                    is_child_of_existing_father = True
                    break

            if is_child_of_existing_father:
                continue
            
            # Check if the father node can generate a valid binary tree of depth l
            current_layer = [father]
            valid_tree = True
            temp_subgraph_nodes = []
            temp_subgraph_edges = []

            # Condition 1: Check if every layer has enough children
            for depth in range(l):
                next_layer = []
                for node in current_layer:
                    if depth == l - 1:
                        b = branch
                    else:
                        b = 2
                    # # 2024-11-2 use n-branched tree
                    # b = branch
                    children = find_children(node, semantic_tree)
                    if len(children) < b:
                        valid_tree = False
                        break
                    children = children[:b]  # Select the first two children
                    next_layer.extend(children)
                    for child in children:
                        if child not in temp_subgraph_nodes:
                            temp_subgraph_nodes.append(child)
                            node_colors[child] = "gray"  # General nodes are gray
                    for child in children:
                        temp_subgraph_edges.append((node, child))
                if not valid_tree:  # this is necessary to break the loop
                    break
                current_layer = next_layer
            
            if not valid_tree:
                continue
            
            # For the last layer, perform random walk until leaf nodes are found
            final_layer = current_layer
            random_walk_depths = {}
            final_children = []
            for node in final_layer:
                walk_depth = 0
                original_node = node
                while semantic_tree.out_degree(node) > 0:
                    children = list(semantic_tree.successors(node))
                    node = random.choice(children)
                    walk_depth += 1
                random_walk_depths[original_node] = walk_depth
                final_children.append(node)
                # if original_node != node:
                #     node_colors[node] = "yellow"  # Change color for nodes replaced by random walk
            print('    final children length:', len(final_children))
            print('    All the nodes have been generated for the father node ', father)
            
            # Collect all final children for shuffling later
            all_final_children.extend(final_children)
            print('    All final children length:', len(all_final_children))
            valid_fathers.append(father)
            for node in temp_subgraph_nodes:
                if node not in subgraph_nodes:
                    subgraph_nodes.append(node)
            for child in final_children:
                if child not in subgraph_nodes:
                    subgraph_nodes.append(child)
            subgraph_edges.extend(temp_subgraph_edges)
            subgraph_edges.extend([(final_layer[i], final_children[i]) for i in range(len(final_layer)) if final_layer[i] != final_children[i]])
            
            # Get second last layer nodes for combining and saving data
            for node in final_layer:
                for parent in semantic_tree.predecessors(node):
                    if parent not in second_last_layer_nodes:
                        second_last_layer_nodes.append(parent)
            pbar.update(1)
    
    for father in valid_fathers:
        node_colors[father] = "red"  # Father nodes are red
        
    # Mark all final children as yellow
    for child in all_final_children:
        node_colors[child] = "yellow"
    
    # Mark second last layer nodes as blue
    for node in second_last_layer_nodes:
        node_colors[node] = "lightblue"
    
    print(f"Second last layer nodes count: {len(second_last_layer_nodes)}")
    print('all final children:', len(all_final_children), all_final_children)
    
    return valid_fathers, subgraph_nodes, subgraph_edges, father_leaf_counts, node_colors, timestamp, all_final_children, second_last_layer_nodes

def combine_and_save_data(matching_files, pkl_dir, father, trail, permutation_strength, s, l, n, num_im, combined_data=None):
    """Combine data from pkl files and save the combined data."""
    if combined_data is None:
        combined_data = {}
        for child, files in matching_files.items():
            for file in files:
                with open(os.path.join(pkl_dir, file), 'rb') as f:
                    data = pickle.load(f)
                    for key, value in data.items():
                        combined_data.setdefault(key, []).append(value[:num_im]) # 2024-11-1 add this to control num of images
        print('data is combined:', combined_data.keys())
        
        for key, value in combined_data.items():
            combined_data[key] = np.concatenate(value, axis=0)
            # 2024-11-2 add shuffle among images
            np.random.shuffle(combined_data[key])
    
    check_combined_data_size(combined_data)

    
    return combined_data

def shuffle_and_save_data(second_last_layer_nodes, all_final_children, pkl_files, pkl_dir, mode, permutation_strength, trail, s, l, b, n, num_im):
    """Shuffle all final children and save the data based on the specified mode."""
    activations = {}
    if mode == 'baseline_shuffle':
        combined_data = shuffle_across_keys(all_final_children, pkl_files, pkl_dir, num_im)
        num_fathers = len(second_last_layer_nodes)
        # 2024-9-16: check baseline shuffle dimension
        # print('num_fathers in baseline shuffle:', num_fathers)
        for i, father in enumerate(second_last_layer_nodes):
            father_data = {key: combined_data[key][i::num_fathers] for key in combined_data}
            activation = combine_and_save_data(None, pkl_dir, father, trail, permutation_strength, s, l, n, num_im, father_data)
            for key, value in activation.items():
                if key not in activations:
                    activations[key] = []
                # Add a new dimension to the value
                value = [value]
                # Append the value to the corresponding key's list
                activations[key].append(value)
        for key in activations.keys():
            activations[key] = np.concatenate(activations[key], axis=0)
        effective_permutation_strength = 0
    elif mode == 'children_shuffle':
        print('Children shuffle, all_final_children:', len(all_final_children), all_final_children)
        shuffle_final_children, effective_permutation_strength = shuffle_children(all_final_children, permutation_strength, s, l, n, b)
        print('    shuffle_final_children:', len(shuffle_final_children), shuffle_final_children)
        for father in second_last_layer_nodes:
            selected_children = shuffle_final_children[:b]
            print(f'        selected children for father node {father}:', selected_children)
            shuffle_final_children = shuffle_final_children[b:]
            shuffle_final_children_label = [to_offset_label(c) for c in selected_children]
            matching_files, _ = search_pkl_files(shuffle_final_children_label, pkl_files)
            activation = combine_and_save_data(matching_files, pkl_dir, father, trail, permutation_strength, s, l, n, num_im)
            for key, value in activation.items():
                if key not in activations:
                    activations[key] = []
                # Add a new dimension to the value
                value = [value]
                # Append the value to the corresponding key's list
                activations[key].append(value)
        for key in activations.keys():
            activations[key] = np.concatenate(activations[key], axis=0)
    else:
        for father in second_last_layer_nodes:
            father_children = all_final_children[:b]
            print(f'        selected children for father node {father}:', father_children)
            all_final_children = all_final_children[b:]
            father_children_label = [to_offset_label(c) for c in father_children]
            matching_files, _ = search_pkl_files(father_children_label, pkl_files)
            print('    matching_files:', matching_files)
            activation = combine_and_save_data(matching_files, pkl_dir, father, trail, permutation_strength, s, l, n, num_im)
            for key, value in activation.items():
                if key not in activations:
                    activations[key] = []
                # Add a new dimension to the value
                value = [value]
                # Append the value to the corresponding key's list
                activations[key].append(value)
        for key in activations.keys():
            activations[key] = np.concatenate(activations[key], axis=0)
        effective_permutation_strength = 0
        
    # print('check activation_dict length:', len(activations))
    # for key, value in activations.items():
    #     print(f"activations {key}: {type(value)}, shape: {value.shape}, value: {value}")
        
    return activations, effective_permutation_strength

    
import random
import networkx as nx

def d_radius_tree(tree, node_num, d_mean, d_range):
    """
    Generate a list of nodes and their distances from a randomly chosen node within a specified distance range.
    
    Parameters:
    - tree: The input tree (networkx graph).
    - node_num: Number of nodes to sample.
    - d_mean: Mean distance.
    - d_range: Range around the mean distance.
    
    Returns:
    - node_list: List of sampled nodes.
    - d_list: List of distances corresponding to the sampled nodes.
    """
    d_min = d_mean - d_range
    d_max = d_mean + d_range
    node_list = []
    
    print(f"Mean distance: {d_mean}, Distance range: {d_range}")
    
    # 2024-10-16 correct this node_0 sample error
    while len(node_list) < node_num:
        node_0 = random.choice([node for node in tree.nodes() if tree.out_degree(node) == 0])
        node_list = [node_0]
        d_list = []
        print(f"    Resample node 0: {node_0}")
        
        start_time = time.time()
        with tqdm(total=node_num - 1, desc="Selecting nodes", unit="node") as pbar:
            while len(node_list) < node_num:
                if time.time() - start_time > 10: # if exceed 10 sec (1 min is too long), then resample node_0
                    print(f"     Find {len(node_list)} node. Exceeded time limit, resampling node_0")
                    break
                
                node = random.choice([node for node in tree.nodes() if tree.out_degree(node) == 0])
                flag = True
                for n in node_list:
                    d = compute_distance_tree(tree, n, node)
                    if d > d_max or d < d_min: # 2024-10-15 correct this distance metric
                        flag = False
                        break
                if flag:
                    node_list.append(node)
                    d_list.append(d)
                    print(f"        Selected node: {node}, Distance: {d}")
                    pbar.update(1)
    
    return node_list, d_list

#2024-10-16 add this
def d_radius_tree_v2(tree, node_num, d_mean, d_range):
    """
    Generate a list of nodes and their distances from a randomly chosen node within a specified distance range.
    
    Parameters:
    - tree: The input tree (networkx graph).
    - node_num: Number of nodes to sample.
    - d_mean: Mean distance.
    - d_range: Range around the mean distance.
    
    Returns:
    - node_list: List of sampled nodes.
    - d_list: List of distances corresponding to the sampled nodes.
    """
    d_min = d_mean - d_range
    d_max = d_mean + d_range
    node_list = []
    
    print(f"Mean distance: {d_mean}, Distance range: {d_range}")
    
    while len(node_list) < node_num:
        node_0 = random.choice([node for node in tree.nodes() if tree.out_degree(node) == 0])
        node_list = [node_0]
        d_list = []
        
        start_time = time.time()
        with tqdm(total=node_num - 1, desc="Selecting nodes", unit="node") as pbar:
            while len(node_list) < node_num:
                if time.time() - start_time > 120:
                    print("Exceeded time limit, resampling node_0")
                    break
                
                node = random.choice([node for node in tree.nodes() if tree.out_degree(node) == 0])
                flag = True
                for n in node_list:
                    d = compute_distance_tree(tree, n, node)
                    if d > d_max:
                        flag = False
                        break
                if flag:
                    node_list.append(node)
                    d_list.append(d)
                    print(f"    Selected node: {node}, Distance: {d}")
                    pbar.update(1)
    
    return node_list, d_list

def compute_distance_tree(tree, node1, node2):
    """
    Compute the distance between two nodes in the tree.
    
    Parameters:
    - tree: The input tree (networkx graph).
    - node1: The first node.
    - node2: The second node.
    
    Returns:
    - distance: The shortest path length between node1 and node2.
    """
    # try:
    #     return nx.shortest_path_length(tree, node1, node2)
    # except nx.NetworkXNoPath:
    #     print(f"No path between {node1} and {node2}.")
    #     return 0
    # return min(nx.shortest_path_length(tree, node1, node2), nx.shortest_path_length(tree, node1, node2))
    undirected_tree = tree.to_undirected()
    try:
        return nx.shortest_path_length(undirected_tree, node1, node2)
    except nx.NetworkXNoPath:
        print(f"No path between {node1} and {node2}.")
        return 0

def n_l_level_tree(tree, l, node_num):
    """
    Generate a list of nodes from a specified level in the tree and find 2 leaf nodes for each node.
    
    Parameters:
    - tree: The input tree (networkx graph).
    - l: The level in the tree.
    - node_num: Number of nodes to sample.
    
    Returns:
    - node_list: List of sampled nodes from the specified level.
    - leaf_node_list: List of leaf nodes found for each node in node_list, stored in order.
    """
    # Find the root node
    root_node = find_root(tree)
    
    level_nodes = [node for node, depth in nx.single_source_shortest_path_length(tree, root_node).items() if depth == l]
    if len(level_nodes) < node_num:
        raise ValueError(f"Not enough nodes at level {l}. Required: {node_num}, Available: {len(level_nodes)}")
    
    # --------
    node_list = []
    leaf_node_list = []
    
    while len(node_list) < node_num:
        # Sample a node from the level_nodes
        node = random.choice(level_nodes)
        
        # Find leaf nodes under the sampled node
        leaf_nodes = [n for n in nx.descendants(tree, node) if tree.out_degree(n) == 0]
        
        if len(leaf_nodes) >= 2:
            node_list.append(node)
            leaf_node_list.extend(random.sample(leaf_nodes, 2))
    
    return node_list, leaf_node_list

import networkx as nx

def find_root(tree):
    """
    Find the root node containing 'entity' in its name and not containing '_'.
    
    Parameters:
    - tree: The input tree (networkx graph).
    
    Returns:
    - root_node: The found root node.
    
    Raises:
    - ValueError: If no such node is found.
    """
    root_node = None
    for node in tree.nodes():
        if 'entity' in node and '_' not in node:
            root_node = node
            break

    if root_node is None:
        raise ValueError("No node containing 'entity' and not containing '_' found in the graph.")

    print(f"Root node found: {root_node}")
    return root_node

def single_l_level_tree(tree, l, node_num):
    """
    Generate a list of leaf nodes under a randomly selected node from a specified level in the tree.
    
    Parameters:
    - tree: The input tree (networkx graph).
    - l: The level in the tree.
    - node_num: Number of leaf nodes to sample.
    
    Returns:
    - leaf_nodes: List of sampled leaf nodes under the selected node.
    """
    # Find the root node
    root_node = find_root(tree)
    
    # Find nodes at level l
    level_nodes = [node for node, depth in nx.single_source_shortest_path_length(tree, root_node).items() if depth == l]
    
    if not level_nodes:
        raise ValueError(f"No nodes found at level {l}.")
    
    for node in level_nodes:
        # Find leaf nodes under the sampled node
        leaf_nodes = [n for n in nx.descendants(tree, node) if tree.out_degree(n) == 0]
        
        if len(leaf_nodes) >= node_num:
            sampled_leaf_nodes = random.sample(leaf_nodes, node_num)
            print(f"Selected Node: {node}, Leaf nodes: {sampled_leaf_nodes}")
            return sampled_leaf_nodes
    
    raise ValueError(f"Not enough leaf nodes under any node at level {l} to sample {node_num} nodes.")



# TODO: draw the trees for d_distance_tree and n_l_level_tree