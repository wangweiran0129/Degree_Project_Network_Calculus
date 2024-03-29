# This code is written by Weiran Wang
# For any questions or problems, please contact the author of code at (weiran.wang@epfl.ch)

import torch
import tqdm
import os
import argparse
from pbzlib import open_pbz
import sys
sys.path.append("../")
from predict.predict_model import *

def predict_attacked_network(model, attacked_network, topology_id, foi, eps):
    
    """
    Output the network topology after flow prolongation, and after the adversarial attack, in a .pbz format.
    
    :param model: the pre-trained model
    :param attacked_network: the attacked network before the GNN fp prediction, but after the adversarial attack, in a .pbz format
    :param topology_id: the id of this network topology, i.e., the network id in the pbz file
    :param foi: the flow of interest id
    :param eps: the eps value
    """

    # GNN prediction path for the attacked topologies
    attacked_network_path = "../../../../Network_Information_and_Analysis/attacked_topology/after_fp/"

    # Predict the flow prolongation
    attacked_network_file = "aafp_" + str(topology_id) + "_" + str(foi) + "_" + str(eps) + ".pbz"
    predict_network(attacked_network, int(foi), model, attacked_network_path + attacked_network_file)
    print("attacked prediction file : ", attacked_network_file)


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("attack_before_fp_path")
    args = p.parse_args()

    # Pre-trained PMOO model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model)
    
    # Attacked network path
    attacked_network_path = args.attack_before_fp_path
    files = os.listdir(attacked_network_path)
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    files.sort(key = lambda x: int(re.findall(r"\d+\.?\d*", x)[0]))

    for file in tqdm(files):

        topo_id, foi, eps = re.findall(r"\d+\.?\d*", file)[0], re.findall(r"\d+\.?\d*", file)[1], re.findall(r"\d+\.?\d*", file)[2]
        if int(topo_id) in range(0, 8000):
            continue
        elif int(topo_id) in range(8000, 9000):
            predict_attacked_network(model, next(open_pbz(attacked_network_path+file)), topo_id, foi, eps)
        else:
            break
