import os
import argparse
import re
from pbzlib import open_pbz
from tqdm import tqdm


def attack_analysis(original_topology, topo_id, foi_id):

    """
    The function will analyze the results of the delay bounds after the attack and/or after the GNN flow prolongation prediction.
    Please feel free to change the condition in this code so that different results can be obtained.

    :param original_topology_path: The .pbz file path for the original network topology, i.e., the topology before the GNN predictiona and before the adversarial attack
    :param topo_id: the id of the topology
    :param foi_id: the id of the flow of interest
    """

    num_tight_nw = 0

    update_max_norm = [0.001, 0.002, 0.003, 0.004, 0.005]

    """
    For the convenience of naming, the following abbreviations are used:
        ---------------------------------
    | delay bound name | FP? | FGSM? |
    |------------------|------|-------|
    |  delay_bound_1   | No   |  No   |
    |  delay_bound_2   | Yes  |  No   |
    |  delay_bound_3   | No   |  Yes  |
    |  delay_bound_4   | Yes  |  Yes  |
        ---------------------------------
    """
        
    # ----- Before FP, Before Attack -----
    nw1 = next(open_pbz(original_topology))
    delay_bound1 = nw1.flow[foi_id].pmoo.delay_bound
                
    # ------ After GNN, Before Attack ------
    nw2_path = "../../../Network_Information_and_Analysis/original_topology/after_fp/original_" + str(topo_id) + "_" + str(foi_id) + ".pbz"
    if os.path.exists(nw2_path):
        nw2 = next(open_pbz(nw2_path))
        delay_bound2 = nw2.flow[foi_id].pmoo.delay_bounds
        
    """
    # How many delay bounds are tightened after the flow prolongation
    if delay_bound2 < delay_bound1:
        num_tight_nw = num_tight_nw + 1
        print("network id", nw1.id)
    """
    
    success1 = 0
    success2 = 0
    success3 = 0
    success4 = 0
    success5 = 0

    for eps in update_max_norm:

        delay_bound3 = 0
        delay_bound4 = 0
        
        attack_db_dif_before_GNN = 0
        attack_db_dif_after_GNN = 0
    
        # ------ Before GNN, After Attack ------
        nw3_path = "../../../Network_Information_and_Analysis/attacked_topology/before_fp/attacked_" + str(topo_id) + "_" + str(foi_id) + "_" + str(eps) + ".pbz"
        if os.path.exists(nw3_path):
            nw3 = next(open_pbz(nw3_path))
            delay_bound3 = nw3.flow[foi_id].pmoo.delay_bound
            # print("delay bound 3 : ", delay_bound3)
        else:
            continue

        # ------ After GNN, After Attack ------
        nw4_path = "../../../Network_Information_and_Analysis/attacked_topology/after_fp/attacked_" + str(eps) + "_" + str(topo_id) + "_" + str(foi) + ".pbz"
        if os.path.exists(nw4_path):
            nw4 = next(open_pbz(nw4_path))
            delay_bound4 = f.pmoo.delay_bound
            # print("delay bound 4 : ", delay_bound4)
        else:
            continue
        
        # Find whether there are some successful attacked examples
        attack_db_dif_before_GNN = abs(delay_bound3 - delay_bound1)
        attack_db_dif_after_GNN = abs(delay_bound4 - delay_bound2)
        
        """
        if attack_db_dif_after_GNN > attack_db_dif_before_GNN*5:
            print("*********************************************************")
            print("network id : ", network_id)
            print("eps : ", eps)
            print("Before GNN, Before Attack : ", delay_bound1)
            print("After GNN, Before Attack : ", delay_bound2)
            print("Before GNN, After Attack : ", delay_bound3)
            print("After GNN, After Attack : ", delay_bound4)
            print("attack difference before GNN : ", attack_db_dif_before_GNN)
            print("attack difference after GNN : ", attack_db_dif_after_GNN)
            print("")
        """

        # The delay bound after FP is tighter before the attack, and looser after the attack
        if (delay_bound2 < delay_bound1) and (delay_bound4 > delay_bound3) and (attack_db_dif_after_GNN > attack_db_dif_before_GNN*3):
            print("*********************************************************")
            print("network id : ", topo_id)
            print("eps : ", eps)
            print("Before GNN, Before Attack : ", delay_bound1)
            print("After GNN, Before Attack : ", delay_bound2)
            print("Before GNN, After Attack : ", delay_bound3)
            print("After GNN, After Attack : ", delay_bound4)
            print("attack difference before GNN : ", attack_db_dif_before_GNN)
            print("attack difference after GNN : ", attack_db_dif_after_GNN)
            print("")
            if (eps == 0.001):
                success1 = success1 + 1
            if (eps == 0.002):
                success2 = success2 + 1
            if (eps == 0.003):
                success3 = success3 + 1
            if (eps == 0.004):
                success4 = success4 + 1
            if (eps == 0.005):
                success5 = success5 + 1
        
    print("success attack ratio for eps 0.001 : ", float(success1/754))
    print("success attack ratio for eps 0.002 : ", float(success2/754))
    print("success attack ratio for eps 0.003 : ", float(success3/754))
    print("success attack ratio for eps 0.004 : ", float(success4/754))
    print("success attack ratio for eps 0.005 : ", float(success5/754))


if __name__ == "__main__":

    # Give the original network topology path
    # The programme will automatically find the corresponding three network topologies

    p = argparse.ArgumentParser()
    p.add_argument("original_topo_path")
    args = p.parse_args()

    original_topology_path = args.original_topo_path

    files = os.listdir(original_topology_path)
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    files.sort(key = lambda x: int(re.findall(r"\d+\.?\d*", x)[0]))

    for file in tqdm(files):
        print("original network name : ", file)
        topo_id, foi_id = int(re.findall(r"\d+\.?\d*", file)[0]), int(re.findall(r"\d+\.?\d*", file)[1])
        attack_analysis(original_topology_path + file, topo_id, foi_id)
