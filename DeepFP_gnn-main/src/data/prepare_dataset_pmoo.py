# This code is based on the prepare_dataset.py written by Karim Hadidane
# This code is changed by Weiran Wang.
# For any questions or problems, please contact the author of the code at (weiran.wang@epfl.ch)

import os
import torch
import pickle
import argparse
import re
import sys
from pbzlib import open_pbz
from torch_geometric.data import Data, DataLoader
from scipy.sparse import coo_matrix
from tqdm import tqdm
sys.path.insert(0, "../")
from data.graph_transformer import *

# The inputs are the graph G, unique identifiers for each node, prolongation nodes if its for training
def graph2torch_pmoo(G, node_ids):
    """

    :param G: networkx graph object of the network with the prolongations
    :param node_ids:
    :return: a torch geometric Data object of the dataset
    """

    # node features : the paper used 11 input dimension (corrected after asking the author). We use 9 features for
    # training but we add one more case in the feature vector to store the flow corresponding to each prolongation node
    x = torch.zeros((len(G.nodes), 10))

    # a placebo, won't be used as we use more than one target
    y = torch.zeros(G.number_of_nodes(), dtype=torch.float)

    # As the loss function depends only on prolongation nodes, we use this to mask the other nodes
    prolongation_mask = torch.zeros(G.number_of_nodes(), dtype=torch.bool)

    for node, data in G.nodes(data=True):
        node_idx = node_ids[node]
        node_type = data["ntype"]

        # Adding node features
        # Adding the type in 1-hot encoding
        x[node_idx, node_type - 1] = 1

        # ADDED: A flow of interest is also a flow
        if node_type == NodeType.Flow_oi:
            x[node_idx, 1] = 1
            # y[node_idx] = 1 if worth_prolonging else 0

        # Adding server rates and latencies : position 4, 5
        if node_type == NodeType.Server:
            x[node_idx, 4] = data["rate_server"]
            x[node_idx, 5] = data["latency"]

        # Adding flow rates and bursts : position 6, 7
        if node_type == NodeType.Flow or node_type == NodeType.Flow_oi:
            x[node_idx, 6] = data["rate_flow"]
            x[node_idx, 7] = data["burst"]

        # Identify the prolongation nodes in the mask
        if node_type == NodeType.Prolong:
            prolongation_mask[node_idx] = 1
            x[node_idx, 8] = data["hop"]
            # index of the corresponding flow
            f = node_ids["f_" + node[1:node.index("_")]]
            x[node_idx, 9] = f

    # Each edge is encoded twice
    edge_index = torch.zeros((2, G.number_of_edges() * 2), dtype=torch.long)

    i = 0
    for src, dst in G.edges():
        # Each edge from the undirected graph G is encoded as two directed edges
        edge_index[0, i] = node_ids[src]
        edge_index[1, i] = node_ids[dst]
        i += 1
        edge_index[0, i] = node_ids[dst]
        edge_index[1, i] = node_ids[src]
        i += 1

    # creating a data object
    graph = Data(x=x, y=y, edge_index=edge_index, mask=prolongation_mask)

    return graph


# Returns best combination in a list of dictionaries, each dictionary is a mapping between key: flow and value: its sink
def get_best_pmoofp_combination(flow):
    """
    A method that returns the best combinations i.e the combinations with minimum delay bounds
    :param flow: the flow of interest to explore
    :return: best combinations of flow prolongations for the flow of interest flow, list of combinations,
    each combination is in the form of dictionary
    """
    # the minimum delay bound found in all the combinations
    min_delay_bound = flow.pmoofp.delay_bound

    # Iterate over all explored combinations for PMOO FP exhaustive search
    min_combinations = [dict(comb.flows_prolongation) for comb in flow.pmoofp.explored_combination \
                        if  comb.delay_bound == min_delay_bound ]

    return min_combinations


# a method to read the network models and extract the graphs , and save them in pickle files
def prepare_dataset_pmoo(path, tp, to_pickle=True):
    """
    :param path: path to the dataset raw file
    :param tp: type (but the type is a key word in Python) to show wether it's a trainning dataset or testing dataset
    :param to_pickle: boolean to indicate if the processed data is stored in serializable format
    :return: the graphs in the dataset (torch geometric data object) and the targets (torch tensors)
    """
    graphs = []
    targets = []

    if tp == "attack":
        topo_id, foi_id = re.findall(r"\d+\.?\d*", file)[0], re.findall(r"\d+\.?\d*", file)[1][:-1]

    # For each network in the file
    for network in open_pbz(path):
        # Get the base graph i.e server nodes, flow nodes, and links between them
        G, flow_paths = net2basegraph(network)

        for flow in network.flow:

            # If the flow has been explored using pmoo FP
            if flow.HasField("pmoofp"):
                # create version of a graph where the current flow is the flow of interest (Algorithm 2 of the paper)
                G_f, pro_dict, node_ids = prolong_graph(G, flow.id, flow_paths)

                # best combinations are the ones that yield minimum delay bound, multiple optimal combinations can exist
                best_combinations = get_best_pmoofp_combination(flow)

                flows_that_can_be_prolonged = set(pro_dict.keys())

                grph = graph2torch_pmoo(G_f, node_ids=node_ids)

                # Append the created graph to the dataset
                graphs.append(grph)

                # worth prolonging if pmoo FP gives tighter delay bound than Pmoo
                worth_prolonging = flow.pmoofp.delay_bound < flow.pmoo.delay_bound
                foi_index = node_ids["f_" + str(flow.id)]

                possible_targets = []

                # Equally optimal targets can exist: this is implementing Equation 11 in the paper
                for comb in best_combinations:
                    # Prolongation nodes to activate
                    comb_nodes = ["p" + str(k) + "_" + str(v) for k, v in comb.items()]

                    # Some prolongation nodes that need to be activated (for flows that will not be prolonged)
                    # are not included in the mapping given in the dataset
                    # These lines are to mitigate this problem
                    prolonged = set(comb.keys())
                    k = flows_that_can_be_prolonged.difference(prolonged)
                    comb_nodes.extend(list(map(lambda x: "p" + str(x) + "_" + str(flow_paths[x][-1]), k)))

                    y = torch.zeros(G_f.number_of_nodes(), dtype=torch.float)
                    y[foi_index] = worth_prolonging

                    prolongation_nodes_indices = torch.tensor([node_ids[pro_node] for pro_node in comb_nodes])
                    y.index_fill_(dim=0, index=prolongation_nodes_indices, value=1)
                    possible_targets.append(y)

                targets.append(possible_targets)

    # save graphs and targets in serializable format according to different types
    if to_pickle:
        if tp == "train":
            file_name_graphs = "train_graphs.pickle"
            file_name_targets = "train_targets.pickle"
        if tp == "test":
            file_name_graphs = "test_graphs.pickle"
            file_name_targets = "test_targets.pickle"
        # Save the attack pickle files to the Network_Information_and_Analysis folder
        if tp == "attack":
            nia = "../../../Network_Information_and_Analysis/original_topology/before_fp/pickle/"
            file_name_graphs = nia + "graphs/attack_graphs_" + str(topo_id) + "_" + str(foi_id) + ".pickle"
            file_name_targets = nia + "targets/attack_targets_" + str(topo_id) + "_" + str(foi_id) + ".pickle"

        # Saving the training graphs in a pickle format
        outfile = open(file_name_graphs, 'wb')
        pickle.dump(graphs, outfile)
        outfile.close()

        # Saving the training targets in a pickle format
        outfile = open(file_name_targets, 'wb')
        pickle.dump(targets, outfile)
        outfile.close()

    return graphs, targets


if __name__ == "__main__":
    
    p = argparse.ArgumentParser()
    p.add_argument("dataset_folder")
    p.add_argument("tp")
    args = p.parse_args()
    dataset_folder = args.dataset_folder
    tp = args.tp

    if tp == "attack":
        files = os.listdir(dataset_folder)
        if ".DS_Store" in files:
            files.remove(".DS_Store")
        files.sort(key = lambda x: int(re.findall(r"\d+\.?\d*", x)[0]))

        for file in tqdm(files):
            topo_id = re.findall(r"\d+\.?\d*", file)[0]
            prepare_dataset_pmoo(dataset_folder + file, tp)

    else:
        prepare_dataset_pmoo(dataset_folder, tp)
