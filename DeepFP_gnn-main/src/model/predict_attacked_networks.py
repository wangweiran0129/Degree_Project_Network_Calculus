# This code is written by Weiran Wang
# For any questions or problems, please contact the author of code at (weiran.wang@epfl.ch)

import torch
import sys
sys.path.append("../")
from output.predict_model import *
import pbzlib
import os

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
    attacked_network_path = main_path + "Network_Information_and_Analysis/Attacked_Topology/After_FP/"
    
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
    
    # Attacked network path
    attacked_network_path = main_path + "Network_Information_and_Analysis/Attacked_Topology/Before_FP/"
    files = os.listdir(attacked_network_path)
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    print("# attacked pbz : ", len(files))

    for file in files:
        print("file : ", file)
        eps, topo_id = re.findall(r"\d+\.?\d*", file)[0], re.findall(r"\d+\.?\d*", file)[1]
        predict_attacked_network(model, next(pbzlib.open_pbz(attacked_network_path+file)), topo_id, eps)


if __name__ == "__main__":
    main()