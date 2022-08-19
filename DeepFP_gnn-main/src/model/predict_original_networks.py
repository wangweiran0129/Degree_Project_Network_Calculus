# This code is written by Weiran Wang
# For any questions or problems, please contact the author of code at (weiran.wang@epfl.ch)

import torch
import sys
sys.path.append("../")
from output.predict_model import *
import pbzlib
import csv
import os

def predict_original_network(model, original_network, topology_id):

    """
    Output the network topology after flow prolongation, but before the adversarial attack, in .pbz format.
    Output a prediction csv file where two prediction values are stored inside.
    This .pbz file will help us find the potential attack targets
    :param model: the pre-trained model
    :param original_network: the original network before the adversarial attack and before the GNN flow prolongation prediction, in .pbz format
    :param topology_id: the id of this network topology, i.e., the network id in the pbz file
    """

    # The main path of this project / Github repo
    main_path = "/Users/wangweiran/Desktop/MasterDegreeProject/Degree_Project_Network_Calculus/"

    # The original_G is the network topology before the flow prolongation
    original_G, flow_paths = net2basegraph(original_network)
    num_server = len(original_network.server)

    # Find the foi (Though in my newly-generated network, there is only one foi)
    fois = []
    for f in original_network.flow:
        if f.HasField('pmoofp'):
            fois.append(f.id)

    print("foi path : ", original_network.flow[fois[0]].path)

    # GNN prediction path for original topologies
    original_fp_path = main_path + 'Network_Information_and_Analysis/'

    # Prolong the flow based on the foi
    for foi in fois:
        print("----- topology id : ", topology_id, "-----")
        print("----- foi : ", foi, "-----")

        # Transoform the original network to get the matrix for the prolonged nodes information
        
        # prolonged_G is the network topology after the flow prolongation
        prolonged_G, pro_dict, node_ids = prolong_graph(original_G, foi, flow_paths)
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
            cross_flow_cominbation_sink_servers.append(flow_paths[foi][(int)(cross_flow[-2].item() - 1)])
        
        # Predict the ORIGINAL network, i.e., the network topology before the adversarial attack
        original_network_after_fp = original_fp_path + 'Original_Topology/after_fp/original_' + str(topology_id) + '_' + str(foi) + '.pbz'
        _, pred1_before_attack, pred2_before_attack = predict_network(original_network, foi, model, original_network_after_fp)

        # GNN prediction pred1 & pred2 analysis folder
        prediction_file_path = main_path + "Network_Information_and_Analysis/Prediction_Value/"
        prediction_file_name = "prediction_" + str(topology_id) + ".csv"

        # Create a new csv file to store the prediction results for a new network topology
        # For attacked dataset, there is only one foi in a topology
        print("GNN prediction file : ", prediction_file_path + prediction_file_name)
        with open(prediction_file_path + prediction_file_name, "w") as csvfile:
            pred_csv = csv.writer(csvfile)
            pred_csv.writerow(["foi", "start server", "sink server", "PRED1 before attack"])
            pred_csv.writerow([foi, flow_paths[foi][0], flow_paths[foi][-1], pred1_before_attack[num_server+foi].item()])
            pred_csv.writerow(["flow id", "start server", "sink server", "PRED2 before attack "]) 
            for line in range(original_G.number_of_nodes(), prolonged_G.number_of_nodes()):
                fid = line - original_G.number_of_nodes()
                pred_csv.writerow([cross_flows[fid], cross_flow_combination_start_servers[fid], cross_flow_cominbation_sink_servers[fid], pred2_before_attack[line].item()])

def predict_attacked_network(model, attacked_network, topology_id, eps):

    """
    Output the network topology after flow prolongation, and after the adversarial attack, in a .pbz format.
    :param model: the pre-trained model
    :param attacked_network: the attacked network before the GNN fp prediction, but after the adversarial attack, in a .pbz format
    :param topology_id: the id of this network topology, i.e., the network id in the pbz file
    :param eps: the eps value
    """

    # The main path of this project / Github repo
    main_path = "/Users/wangweiran/Desktop/MasterDegreeProject/Degree_Project_Network_Calculus/"

    # GNN prediction path for the attacked topologies
    attacked_network_path = main_path + "Network_Information_and_Analysis/Attacked/after_fp/"
    
    for f in attacked_network.flow:
        if f.HasField("pmoo"):
            foi = f.id

    # Predict the flow prolongation
    attacked_network_file = "attacked_" + str(eps) + "_" + str(topology_id) + "_" + str(foi) + ".pbz"
    predict_network(attacked_network, foi, model, attacked_network_path + attacked_network_file)
    print("attacked prediction file : ", attacked_network_file)


def main():

    # The main path of this project / Github repo
    main_path = "/Users/wangweiran/Desktop/MasterDegreeProject/Degree_Project_Network_Calculus/"

    # Pre-trained PMOO model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(main_path + "DeepFP_gnn-main/src/model/ggnn_pmoo.pt", map_location=torch.device(device))
    
    # Original network path
    original_network_features_path = main_path + "Network_Information_and_Analysis/Original_Topology/before_fp/dataset-attack-large.pbz"

    for original_network in pbzlib.open_pbz(original_network_features_path):
        topology_id = original_network.id
        predict_original_network(model, original_network, topology_id)

    """
    # Attacked network path
    attacked_network_path = main_path + "Network_Information_and_Analysis/Attacked/before_fp/"
    files = os.listdir(attacked_network_path)
    if ".DS_Store" in files:
        files.remove(".DS_Store")

    for file in files:
        print("file : ", file)
        eps, topo_id = re.findall(r"\d+\.?\d*", file)[0], re.findall(r"\d+\.?\d*", file)[1]
        predict_attacked_network(model, next(pbzlib.open_pbz(attacked_network_path+file)), topo_id, eps)
    """


if __name__ == "__main__":
    main()