import os
import argparse
from pbzlib import open_pbz


def attack_analysis(original_topology_path):

    """
    The function will analyze the results of the delay bounds after the attack and/or after the GNN flow prolongation prediction.
    Please feel free to change the condition in this code so that different results can be obtained.

    :param original_topology_path: The .pbz file path for the original network topology, i.e., the topology before the GNN predictiona and before the adversarial attack
    """

    num_tight_nw = 0

    update_max_norm = [0.0001, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005]

    for nw1 in open_pbz(original_topology_path):

        """
        For the convenience of naming, the following abbreviations are used:
         ---------------------------------
        | delay bound name | GNN? | FGSM? |
        |------------------|------|-------|
        |  delay_bound_1   | No   |  No   |
        |  delay_bound_2   | Yes  |  No   |
        |  delay_bound_3   | No   |  Yes  |
        |  delay_bound_4   | Yes  |  Yes  |
         ---------------------------------
        """
        
        delay_bound1 = 0
        delay_bound2 = 0
        
        network_id = nw1.id
        
        # ----- Before GNN, Before Attack -----
        foi = -1
        for f in nw1.flow:
            if f.HasField("pmoo"):
                foi = f.id
                delay_bound1 = f.pmoo.delay_bound
                # print("delay bound 1 : ", delay_bound1)
                
        # ------ After GNN, Before Attack ------
        nw2_path = "../../../Network_Information_and_Analysis/original_topology/after_fp/original_" + str(network_id) + "_" + str(foi) + ".pbz"
        if os.path.exists(nw2_path):
            nw2 = next(open_pbz(nw2_path))
            for f in nw2.flow:
                if f.HasField("pmoo"):
                    delay_bound2 = f.pmoo.delay_bound
                    # print("delay bound 2 : ", delay_bound2)
        else:
            continue
        
        """
        # How many delay bounds are tightened after the flow prolongation
        if delay_bound2 < delay_bound1:
            num_tight_nw = num_tight_nw + 1
            print("network id", nw1.id)
        """
        
        for eps in update_max_norm:

            delay_bound3 = 0
            delay_bound4 = 0
            
            attack_db_dif_before_GNN = 0
            attack_db_dif_after_GNN = 0
        
            # ------ Before GNN, After Attack ------
            # print("Which file not found?", "Network_Information_and_Analysis/attacked_topology/before_fp/attacked_" + str(eps) + "_" + str(network_id) + "_" + str(foi) + ".pbz")
            nw3_path = "../../../Network_Information_and_Analysis/attacked_topology/before_fp/attacked_" + str(eps) + "_" + str(network_id) + "_" + str(foi) + ".pbz"
            if os.path.exists(nw3_path):
                nw3 = next(open_pbz(nw3_path))
                for f in nw3.flow:
                    if f.HasField("pmoo"):
                        delay_bound3 = f.pmoo.delay_bound
                        # print("delay bound 3 : ", delay_bound3)
            else:
                continue

            # ------ After GNN, After Attack ------
            nw4_path = "../../../Network_Information_and_Analysis/attacked_topology/after_fp/attacked_" + str(eps) + "_" + str(network_id) + "_" + str(foi) + ".pbz"
            if os.path.exists(nw4_path):
                nw4 = next(open_pbz(nw4_path))
                for f in nw4.flow:
                    if f.HasField("pmoo"):
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

            # some order depending like this: attack_db_dif_after_GNN - attack_db_dif_before_GNN

            if (delay_bound2 < delay_bound1) and (delay_bound4 > delay_bound3):
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



if __name__ == "__main__":

    # Give the original network topology path
    # The programme will automatically find the corresponding three network topologies

    p = argparse.ArgumentParser()
    p.add_argument("original_topo_path")
    args = p.parse_args()

    original_topology_path = args.original_topo_path
    
    attack_analysis(original_topology_path)
