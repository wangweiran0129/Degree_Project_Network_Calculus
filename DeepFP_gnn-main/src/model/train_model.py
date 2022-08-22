# This code is written by Karim Hadidane, and modified by Weiran Wang.
# For any questions or problems, please contact the author of the code at (weiran.wang@epfl.ch)

import pickle
import random
import numpy as np
import argparse
from tqdm import tqdm
from scipy.sparse import coo_matrix
import sys
sys.path.insert(0, "../")
from model.gnn import *


def train_model(model, train_graphs, train_targets, test_graphs, test_targets, learning_rate, epochs, dropout=0.2):
    """
    This method is used to train the model
    :param model: the model to train
    :param train_graphs: serializable file for training graphs
    :param train_targets: serializable file for training graphs targets
    :param test_graphs: serializable file for test graphs
    :param test_targets: serializable file for test graphs targets
    :param learning_rate: the learning rate of the model
    :param epochs: number of epochs to train
    :param dropout: dropout probability parameter , default: 0.2
    :return: the model trained
    :return: the losses per epoch
    """
    # train_data = DataLoader(train_dataset, batch_size=batch_size)
    outfile = open(train_graphs, 'rb')
    train_graphs = pickle.load(outfile)
    outfile.close()
    outfile = open(train_targets, 'rb')
    train_targets = pickle.load(outfile)
    outfile.close()

    # train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Added: test data
    outfile = open(test_graphs, 'rb')
    test_graphs = pickle.load(outfile)
    outfile.close()

    outfile = open(test_targets, 'rb')
    test_targets = pickle.load(outfile)
    outfile.close()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model = model.to(device)

    losses_per_epoch = []
    test_accuracies_deepfp = []
    test_accuracies_deepfp4 = []

    # Tells the model that we are training
    model.train()
    for epoch in tqdm(range(epochs)):

        # Reshuffle
        train_data = random.sample(list(zip(train_graphs, train_targets)), len(train_graphs))
        test_data = random.sample(list(zip(test_graphs, test_targets)), len(test_graphs))

        losses_train = []
        for graph, targets in train_data:
            # Set gradients to zero
            opt.zero_grad()

            # create adjacency matrix
            adjacency = prepare_adjacency_matrix(graph)

            # pred1 is for the first classification task, classify if its worth prolonging
            # pred2 is for the second classification task, predict prolongations
            pred1, pred2 = model(graph.to(device), adjacency.to(device))

            # Identify nodes of flow of interests
            foi_idx = torch.where(graph.x[:, 2])[0]

            losses = []

            # Implementing equation 11 of the paper: dynamically choosing the target vector from equally good
            # combinations ===> we backward only the minimum loss
            for t in targets:
                # Define the loss function to use: here we use Cross entropy for both classification problems
                criterion1 = nn.BCELoss()
                criterion2 = nn.BCELoss()

                target_foi = torch.index_select(t.to(device), 0, foi_idx)
                output_foi = torch.index_select(pred1.view(-1), 0, foi_idx)

                #indices of prolongation nodes
                idxmask = torch.where(graph.mask)[0]

                # prolongations nodes : target values and
                target_prolongations = torch.index_select(t.to(device), 0, idxmask)
                output_prolongations = torch.index_select(pred2.view(-1), 0, idxmask)

                loss = criterion1(output_prolongations, target_prolongations) + criterion2(output_foi,
                                                                                           target_foi)

                losses.append(loss)

            min_loss = losses[np.argmin(list(map(lambda x: x.item(), losses)))]
            min_loss.backward()
            losses_train.append(min_loss.item())

            # update the parameters
            opt.step()

        losses_per_epoch.append(np.mean(losses_train))

        acc_deepfp, acc_deepfp4 = evaluate(model, test_data, device)
        test_accuracies_deepfp.append(acc_deepfp)
        test_accuracies_deepfp4.append(acc_deepfp4)

        tqdm.write(
            f"{epoch:3d} | loss={np.mean(losses_train):.2e}  | test_accuracy_deepfp= {acc_deepfp} | test_accuracy_deepfp4= {acc_deepfp4}")

    return model, losses_per_epoch


def evaluate(model, test_data, device):
    """
    A method to calculate the accuracy of the model in deepFp setting and deepFP4 setting

    :param model: the model parameters
    :param test_data: the test data
    :param device:
    :return: accuracy for deepFP, accuracy for deepFP4
    """
    # Tells the model that we are evaluating
    model.eval()

    correct_predicted_deepfp = 0
    correct_predicted_deepfp4 = 0

    for graph, targets in test_data:
        with torch.no_grad():
            adj = prepare_adjacency_matrix(graph)
            out1, out2 = model(graph.to(device), adj.to(device))

        accurate_deepfp, accurate_deepfp4 = accurate_deepFP_deepFP4(out1, out2, graph, targets)
        correct_predicted_deepfp = correct_predicted_deepfp + accurate_deepfp
        correct_predicted_deepfp4 = correct_predicted_deepfp4 + accurate_deepfp4

    # for each graph in the dataset, a correct classification means that the model outputs the correct flow
    # prolongations for all cross flows
    accuracy_deepfp = correct_predicted_deepfp / len(test_data)
    accuracy_deepfp4 = correct_predicted_deepfp4 / len(test_data)

    return accuracy_deepfp, accuracy_deepfp4


def accurate_deepFP_deepFP4(classification_foi_output, classification_pro_output, graph, targets):
    """
    check if the graph nodes are well classified by the model
    :param classification_foi_output: the output of the model (foi node) for the first classification task: is it worth prolonging
    :param classification_pro_output: the output of the model (prolong nodes) for the second classification task: where to prolong
    :param graph: the graph data
    :param targets: the targets vectors
    :return: True if the model predicts every possible prolongation
    """
    # Doing every thing on cpu
    classification_foi_output = classification_foi_output.cpu()
    classification_pro_output = classification_pro_output.cpu()
    graph = graph.cpu()

    foi_idx = torch.where(graph.x[:, 2])[0]
    output_foi = torch.index_select(classification_foi_output.view(-1), 0, foi_idx)

    # we use a cutoff of 0.5
    predicted_label = 1 if output_foi.item() >= 0.5 else 0

    # the indices of the prolongation nodes
    idxmask = torch.where(graph.mask)[0]
    output_prolongations = torch.index_select(classification_pro_output.view(-1), 0, idxmask)

    pro_nodes = graph.x[graph.mask]

    prolongations_deepfp = create_output_vector(pro_nodes, output_prolongations, 1)
    prolongations_deepfp4 = create_output_vector(pro_nodes, output_prolongations, 4)

    worth_prolonging = torch.index_select(targets[0], 0, foi_idx).item() == 1

    if worth_prolonging and predicted_label == 1:
        correct_predicted_targets_deepfp = 0
        correct_predicted_targets_deepfp4 = 0

        for t in targets:
            target_prolongations = torch.index_select(t, 0, idxmask).cpu()
            correct_predicted_targets_deepfp = correct_predicted_targets_deepfp + compute_correct_solutions(
                prolongations_deepfp, target_prolongations)
            correct_predicted_targets_deepfp4 = correct_predicted_targets_deepfp4 + compute_correct_solutions(
                prolongations_deepfp4, target_prolongations)

        return correct_predicted_targets_deepfp > 0, correct_predicted_targets_deepfp4 > 0
    elif (not worth_prolonging) and predicted_label == 0:
        return True, True

    return False, False


def compute_correct_solutions(output_prolongations, target_prolongations):
    """
    compute the number of correct solutions from the flow prolongation combinations returned by deepFP or deepFP_k

    :param output_prolongations:
    :param target_prolongations:
    :return: the number of combinations that matches with the targets
    """
    s = 0

    for alternative in output_prolongations:
        s = s + (torch.sum(target_prolongations == alternative.cpu()).item() == len(target_prolongations))

    return s


def create_output_vector(pro_nodes, out_prol, number_of_combinations=1):
    """

    :param pro_nodes:
    :param out_prol: the output of prolongation nodes
    :param number_of_combinations: the number of predictions to generate, e.g 4 for DeepFP4
    :return: combinations generated from output predictions of the neural network
    """
    target_prolo = torch.zeros(out_prol.shape[0])
    combinations = []

    flows = set(map(lambda x: int(x.item()), pro_nodes[:, 9]))

    if number_of_combinations == 1:
        for f in flows:
            # Select prolongation nodes related to the flow
            node_indx = torch.nonzero(pro_nodes[:, 9] == f)
            # Select the node with the highest score to be the sink of the prolongation
            r = torch.index_select(out_prol, 0, node_indx.squeeze())

            ind = node_indx[torch.argmax(r, dim=0)].item()

            # Create the output vector
            target_prolo[ind] = 1

        combinations.append(target_prolo)

    else:
        for i in range(number_of_combinations):
            target_prolo = torch.zeros(out_prol.shape[0])
            for f in flows:
                # Select prolongation nodes related to the flow
                node_indx = torch.nonzero(pro_nodes[:, 9] == f)

                # Select the node with the highest score to be the sink of the prolongation
                r = torch.index_select(out_prol, 0, node_indx.squeeze())

                # Generating probabilities from prediction vector output of the model (use softmax here)
                probs = np.array(r.view(len(r)).cpu())
                probs /= np.sum(probs)

                ind = np.random.choice(a=list(range(len(probs))), p=probs)
                # Create the output vector
                target_prolo[ind + node_indx[0].item()] = 1

            combinations.append(target_prolo)

    # Combinations are the predictions that we generate from the vector of gnn predictions
    return combinations


def prepare_adjacency_matrix(batch):
    """
    a method to construct the adjacency matrix for the edge attention layer
    :param batch:
    :return: adjacency matrix FloatTensor object
    """
    nodes_number = batch.x.shape[0]
    adj = coo_matrix(
        (np.ones(batch.edge_index.shape[1]), (batch.edge_index[0].cpu(), batch.edge_index[1].cpu())),
        shape=(nodes_number, nodes_number))
    adj = torch.FloatTensor(np.array(adj.todense())) + torch.eye(nodes_number)

    return adj


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("train_graphs")
    p.add_argument("train_targets")
    p.add_argument("test_graphs")
    p.add_argument("test_targets")
    p.add_argument("learning_rate")
    p.add_argument("epochs")
    args = p.parse_args()

    model = GGNN(9, 96, unrolls=2)
    train_graphs_path = args.train_graphs
    train_targets_path = args.train_targets
    test_graphs_path = args.test_graphs
    test_targets_path = args.test_targets
    
    model, losses_per_epoch = train_model(model=model, train_graphs=train_graphs_path, train_targets=train_targets_path,\
           test_graphs=test_graphs_path, test_targets=test_targets_path, learning_rate=args.learning_rate, epochs=args.epochs)

    print("model : ", model)
    print("losses_per_epoch : ", losses_per_epoch)
    
    torch.save(model, "deepfpPMOO.pt")
