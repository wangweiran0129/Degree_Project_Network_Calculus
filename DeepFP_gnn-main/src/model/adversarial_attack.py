# This code is written by Weiran Wang
# For any questions or problems, please contact the author of code at (weiran.wang@epfl.ch)

import sys
sys.path.insert(0, "../")
import pickle
import torch
import csv
from pbzlib import write_pbz, open_pbz
from model.train_model import *
from output.write_attacked_network import *
from data.prepare_dataset_deborah import *
from data.prepare_dataset_pmoo import *

torch.set_printoptions(profile="full")


def fgsm_update(feature_matrix, feature_matrix_grad, eps, flow_rate):
    """
    Use Fast Gradient Sign Attack (FGSM) to add adversarial attack on server/flow feature matrix
    It mainly follows the function: x = x + eps * sign(x.grad)
    :param feature_matrix: the feature matrix of the server/flow, i.e., server latency, server rate, flow rate or flow burst
    :param feature_matrix_grad: the gradient of server/flow feature matrix
    :param eps: epsilons value in the attack
    :param flow_rate: a boolen variable, because the flow rate is very sensitive to the eps
    :return: the server/flow feature matrix after the adversarial attack
    """
    
    # Collect the element-wise sign of the data gradient
    sign_feature_matrix_grad = feature_matrix_grad.sign()

    if flow_rate == True:
        eps = eps / 40

    # Create the perturbed features
    perturbed_feature_matrix = feature_matrix + eps * sign_feature_matrix_grad

    # Adding clipping to maintain [0,1]
    perturbed_feature_matrix = torch.clamp(perturbed_feature_matrix, 0, 1)

    return perturbed_feature_matrix


def evaluate_attack(model, device, potential_attack_target_topology_id):
    """
    Evaluate the attack perfomance with different eps values
    Output the pbz file after the attack
    :param model: the pre-trained pmoo or ludb model
    :param device: cpu or gpu
    :param potential_attack_target_topology_id: the topology id where attack might happen
    """

    # Define the epsilon
    update_max_norm = [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0012, 0.0014, 0.0016, 0.0018, 0.002]
    
    main_path = "/Users/wangweiran/Desktop/MasterDegreeProject/Degree_Project_Network_Calculus/"
    attack_dataset = "Network_Information_and_Analysis/Original_Topology/before_fp/dataset-attack-large.pbz"

    for eps in update_max_norm:

        # Load the dataset
        attack_graphs_path = main_path + "Network_Information_and_Analysis/Original_Topology/before_fp/attack_graphs.pickle"
        attack_targets_path = main_path + "Network_Information_and_Analysis/Original_Topology/before_fp/attack_targets.pickle"
        
        outfile = open(attack_graphs_path, 'rb')
        attack_graphs = pickle.load(outfile)
        outfile.close()

        outfile = open(attack_targets_path, 'rb')
        attack_targets = pickle.load(outfile)
        outfile.close()

        dataset = list(zip(attack_graphs, attack_targets))
    
        # Loop over all examples in test set in batches
        for graph, targets in dataset:

            print("eps : ", eps)

            # The indices of servers
            server_index = graph.x[:,0]
            server_start_index = 0
            server_end_index = server_index.nonzero()[-1]
            server_feature = graph.x[server_start_index:server_end_index+1, 4:6]

            # The indices of flows
            flow_index = graph.x[:,1]
            flow_index_start_index = flow_index.nonzero()[0]
            flow_index_end_index = flow_index.nonzero()[-1]
            flow_feature = graph.x[flow_index_start_index:flow_index_end_index+1, 6:8]

            # Confirm and get the network topology id
            for original_network in open_pbz(main_path + attack_dataset):
                if original_network.server[0].rate == graph.x[0][4] and original_network.server[0].latency == graph.x[0][5]:
                    topology_id = original_network.id
                    break
            
            print("----- topology id : ", topology_id, " -----")

            if topology_id in potential_attack_target_topology_id:
            
                # Create adjacency matrix
                adjacency = prepare_adjacency_matrix(graph)

                # Indicate that we want PyTorch to compute a gradient with respect to the input batch
                graph.x.requires_grad = True

                # Feed forward pass
                # original model feed forward
                pred1, pred2 = model(graph.to(device), adjacency.to(device))

                # Identify nodes of flow of interests
                foi_idx = torch.where(graph.x[:, 2])[0]
                foi = foi_idx.item() - len(server_feature)

                # Calculate the loss of feed forward
                # The gradient of input network feature matrix highly depends on this loss
                losses = []
                for t in targets:
                    criterion1 = nn.BCELoss()
                    criterion2 = nn.BCELoss()

                    target_foi = torch.index_select(t.to(device), 0, foi_idx)
                    output_foi = torch.index_select(pred1.view(-1), 0, foi_idx)

                    # indices of prolongation nodes
                    idxmask = torch.where(graph.mask)[0]

                    # prolongatins nodes: target values
                    target_prolongations = torch.index_select(t.to(device), 0, idxmask)
                    output_prolongations = torch.index_select(pred2.view(-1), 0, idxmask)

                    loss = criterion1(output_prolongations, target_prolongations) + criterion2(output_foi, target_foi)
                    losses.append(loss)

                # Zero all existing gradient
                min_loss = losses[np.argmin(list(map(lambda x: x.item(), losses)))]
                model.zero_grad()
                min_loss.backward()

                # Find the min/max server rate and server latency
                min_server_rate = graph.x[0:len(server_feature), 4].min()
                min_server_latency = graph.x[0:len(server_feature), 5].min()
                max_server_rate = graph.x[0:len(server_feature), 4].max()
                max_server_latency = graph.x[0:len(server_feature), 5].max()
                
                # Find the min/max flow rate and burst
                min_flow_rate = graph.x[len(server_feature):len(server_feature)+len(flow_feature), 6].min()
                min_flow_burst = graph.x[len(server_feature):len(server_feature)+len(flow_feature), 7].min()
                max_flow_rate = graph.x[len(server_feature):len(server_feature)+len(flow_feature), 6].max()
                max_flow_burst = graph.x[len(server_feature):len(server_feature)+len(flow_feature), 7].max()

                # Attack the whole graph excapt for the flow rate
                perturbed_graph_feature = fgsm_update(graph.x, graph.x.grad, eps, flow_rate = False)
                
                # Recover the original value for the server rate and latency
                x_hat = graph.x.detach()
                x_hat[0:len(server_feature), 4:6] = perturbed_graph_feature[0:len(server_feature), 4:6]
                # Replace 0 with the minimum value
                x_hat[0:len(server_feature), 4] = torch.where(x_hat[0:len(server_feature), 4]==0, min_server_rate, x_hat[0:len(server_feature), 4])
                x_hat[0:len(server_feature), 5] = torch.where(x_hat[0:len(server_feature), 5]==0, min_server_latency, x_hat[0:len(server_feature), 5])
                # Replace 1 with the maximum value
                x_hat[0:len(server_feature), 4] = torch.where(x_hat[0:len(server_feature), 4]==1, max_server_rate, x_hat[0:len(server_feature), 4])
                x_hat[0:len(server_feature), 5] = torch.where(x_hat[0:len(server_feature), 5]==1, max_server_latency, x_hat[0:len(server_feature), 5])

                # Recover the original value for the flow burst
                x_hat[len(server_feature):len(server_feature)+len(flow_feature), 7] = perturbed_graph_feature[len(server_feature):len(server_feature)+len(flow_feature), 7]
                # Replace 0 with the minimum value
                x_hat[len(server_feature):len(server_feature)+len(flow_feature), 7] = torch.where(x_hat[len(server_feature):len(server_feature)+len(flow_feature), 7]==0, min_flow_burst, x_hat[len(server_feature):len(server_feature)+len(flow_feature), 7])
                # Replace 1 with the maximum value
                x_hat[len(server_feature):len(server_feature)+len(flow_feature), 7] = torch.where(x_hat[len(server_feature):len(server_feature)+len(flow_feature), 7]==1, max_flow_burst, x_hat[len(server_feature):len(server_feature)+len(flow_feature), 7])
                
                # Attack the whole graph for the flow rate
                perturbed_graph_feature_flow_rate = fgsm_update(graph.x, graph.x.grad, eps, flow_rate = True)
                # Update the attacked flow rates
                x_hat[len(server_feature):len(server_feature)+len(flow_feature), 6] = perturbed_graph_feature_flow_rate[len(server_feature):len(server_feature)+len(flow_feature), 6]
                # Replace 0 with the minimum value
                x_hat[len(server_feature):len(server_feature)+len(flow_feature), 6] = torch.where(x_hat[len(server_feature):len(server_feature)+len(flow_feature), 6]==0, min_flow_rate, x_hat[len(server_feature):len(server_feature)+len(flow_feature), 6])
                # Replace 1 with the maximum value
                x_hat[len(server_feature):len(server_feature)+len(flow_feature), 6] = torch.where(x_hat[len(server_feature):len(server_feature)+len(flow_feature), 6]==1, max_flow_rate, x_hat[len(server_feature):len(server_feature)+len(flow_feature), 6])

                # Write the changes to a new .pbz file
                attacked_network_path = main_path + "Network_Information_and_Analysis/Attacked_Topology/Before_FP/"
                attacked_file_name = "attacked_" + str(eps) + "_" +str(topology_id) + "_" + str(foi) + ".pbz"
                print("attacked network file name : ", attacked_network_path + attacked_file_name)
                write_attacked_network(original_network, x_hat, foi, attacked_network_path+attacked_file_name)
            
            else:
                continue


def get_potential_attack_topology_id(potential_attack_file):
    """
    Grab the potential topology id
    :param potential_attack_file: the file name of the potential attack target
    :return: the potential attack topology id
    """
    
    topology_id_line = []
    topology_id = []
    
    # Navigate to the topology id chunk
    file = csv.reader(open(potential_attack_file))
    for line, content in enumerate(file):
        if content[0] == "topology id":
            topology_id_line.append(line+1)
    
    # Grab the topology id
    file = csv.reader(open(potential_attack_file))
    for line, content in enumerate(file):
        if line in topology_id_line:
            topology_id.append(int(content[0]))
    
    return topology_id


def main():

    main_path = "/Users/wangweiran/Desktop/MasterDegreeProject/Degree_Project_Network_Calculus/"
    potential_attack_file = main_path + "Network_Information_and_Analysis/potential_attack_target2.csv"

    # Load the pretrained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(main_path + "DeepFP_gnn-main/src/model/ggnn_pmoo.pt", map_location=torch.device(device))
    potential_attack_target_topology_id = get_potential_attack_topology_id(potential_attack_file)
    print("potential attack target topology id : ", potential_attack_target_topology_id)

    # Do the adversarial attack
    evaluate_attack(model, device, potential_attack_target_topology_id)


if __name__ == "__main__":
    main()