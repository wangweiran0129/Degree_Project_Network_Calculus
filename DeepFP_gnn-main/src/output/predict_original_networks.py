# This code is written by Weiran Wang
# For any questions or problems, please contact the author of code at (weiran.wang@epfl.ch)

import torch
import csv
import argparse
import os
import re
from tqdm import tqdm
import sys
sys.path.append("../")
from output.predict_model import *
from pbzlib import open_pbz

def predict_original_network(model, original_network, topology_id, foi_id):

    """
    Output the network topology after flow prolongation, but before the adversarial attack, in .pbz format.
    Output a prediction csv file where two prediction values are stored inside.
    This .pbz file will help us find the potential attack targets
    :param model: the pre-trained model
    :param original_network: the original network before the adversarial attack and before the GNN flow prolongation prediction, in .pbz format
    :param topology_id: the id of this network topology, i.e., the network id in the pbz file
    :param foi_id: the flow of interest id
    """

    # The original_G is the network topology before the flow prolongation
    original_G, flow_paths = net2basegraph(original_network)
    num_server = len(original_network.server)

    # Find the foi (Though in my newly-generated network, there is only one foi)
    print("foi path : ", original_network.flow[foi_id].path)

    # GNN prediction path for original topologies
    original_fp_path = "../../../Network_Information_and_Analysis/"

    # Prolong the flow based on the foi
    print("----- topology id : ", topology_id, "-----")
    print("----- foi : ", foi_id, "-----")

    # Transoform the original network to get the matrix for the prolonged nodes information
    
    # prolonged_G is the network topology after the flow prolongation
    prolonged_G, pro_dict, node_ids = prolong_graph(original_G, foi_id, flow_paths)
    # prolonged_graph is a torch matrix
    prolonged_graph = graph2torch_pmoo(prolonged_G, node_ids)
    
    # Find the start servers and sink servers
    cross_flows = []
    cross_flow_combination_start_servers = []
    cross_flow_cominbation_sink_servers = []

    for cross_flow in prolonged_graph.x[original_G.number_of_nodes():]:
        flow_id = (int)(cross_flow[-1].item() - num_server)
        cross_flows.append(flow_id)
        cross_flow_combination_start_servers.append(flow_paths[flow_id][0])
        cross_flow_cominbation_sink_servers.append(flow_paths[foi_id][(int)(cross_flow[-2].item() - 1)])
    
    # Predict the ORIGINAL network, i.e., the network topology before the adversarial attack
    original_network_after_fp = original_fp_path + 'original_topology/after_fp/original_' + str(topology_id) + '_' + str(foi_id) + '.pbz'
    _, pred1_before_attack, pred2_before_attack = predict_network(original_network, foi_id, model, original_network_after_fp)

    # GNN prediction pred1 & pred2 analysis folder
    prediction_file_path = "../../../Network_Information_and_Analysis/prediction_value/"
    prediction_file_name = "prediction_" + str(topology_id) + "_" + str(foi_id) + ".csv"

    # Create a new csv file to store the prediction results for a new network topology
    # For attacked dataset, there is only one foi in a topology
    print("GNN prediction file : ", prediction_file_path + prediction_file_name)
    with open(prediction_file_path + prediction_file_name, "w") as csvfile:
        pred_csv = csv.writer(csvfile)
        pred_csv.writerow(["foi", "start server", "sink server", "PRED1 before attack"])
        pred_csv.writerow([foi_id, flow_paths[foi_id][0], flow_paths[foi_id][-1], pred1_before_attack[num_server+foi_id].item()])
        pred_csv.writerow(["flow id", "start server", "sink server", "PRED2 before attack "]) 
        for line in range(original_G.number_of_nodes(), prolonged_G.number_of_nodes()):
            fid = line - original_G.number_of_nodes()
            pred_csv.writerow([cross_flows[fid], cross_flow_combination_start_servers[fid], cross_flow_cominbation_sink_servers[fid], pred2_before_attack[line].item()])


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("path")
    args = p.parse_args()

    # Load the pre-trained PMOO model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device : ", device)
    model = torch.load(args.model)
    
    # Original network path
    original_network_path = args.path

    files = os.listdir(original_network_path)
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    files.sort(key = lambda x: int(re.findall(r"\d+\.?\d*", x)[0]))

    for file in files:
        topo_id, foi_id = re.findall(r"\d+\.?\d*", file)[0], re.findall(r"\d+\.?\d*", file)[1][:-1]
        original_network = next(open_pbz(original_network_path + file))
        predict_original_network(model, original_network, int(topo_id), int(foi_id))
