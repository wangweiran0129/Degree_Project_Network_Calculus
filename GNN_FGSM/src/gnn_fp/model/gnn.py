# This code is written by Karim Hadidane.
# For any questions or problems, please contact the author of the code at (karim.hadidane@epfl.ch)


import torch
import torch_geometric.nn as pyg_nn
import torch.nn as nn


# This class implements the gated graph neural network model that will be used for training and prediction
class GGNN(nn.Module):
    """
    A class of the Gated Graph neural network model of DeepFP
    """

    # unrolls is the number of recurrence unrolls to perform
    def __init__(self, input_dim, hidden_dim, unrolls, dropout=0):
        """
        initialize the model
        :param input_dim: input dimension, number of features of each node (in the paper used is 9)
        :param hidden_dim: the hidden feature representation dimensions (in the paper 96)
        :param unrolls: the number of unrolls of the gated graph convolution layer
        :param dropout: dropout probability
        """
        super(GGNN, self).__init__()

        self.unrolls = unrolls

        # Init
        self.init_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.LeakyReLU())

        # # Memory Unit
        self.gru = pyg_nn.GatedGraphConv(hidden_dim, 1, aggr="mean")
        #
        # # Edge attention
        self.edge_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=True),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 1, bias=True),
            nn.Sigmoid()
        )
        #
        # # Output Hidden layers
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU()
        )

        # Output layer 1 : for node classification to predict flow prolongations
        self.out1 = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
            nn.Sigmoid()
        )

        # Output layer 2 : for node classification to predict if worth prolonging
        self.out2 = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, data, adj):
        """
        Forward pass of the model
        :param data: the graph data
        :param adj: the adjacency matrix of the graph
        :return: vector of predictions for each node in the graph for each of the two classification tasks
        """
        # unpacking the data object
        x, edge_index = data.x[:, :-1], data.edge_index

        x = self.init_layer(x)

        for i in range(self.unrolls):
            x = self.gru(x, edge_index)
            # Prepare the input for the edge attention mechanism
            d = prepare_edge_attention_input(x)

            #    Run the edge attention layer and squeeze to remove the last dimension
            coeffs = self.edge_attention(d).squeeze()
            zero_vec = torch.zeros_like(coeffs)
            coeffs = torch.where(adj == 1, coeffs, zero_vec)
            x = coeffs @ x

        x = self.out(x)
        out1 = self.out1(x)
        out2 = self.out2(x)
        return out1, out2


def prepare_edge_attention_input(feature_matrix):
    """
        Method to prepare the input for the edge attention layer by concatenating neighboring nodes feature vectors
        :param feature_matrix: torch tensor of the data of shape N x D
        :return: torch tensor of shape N x N x (2*D)
        """
    # Number of nodes
    N = feature_matrix.shape[0]

    # Number of features
    D = feature_matrix.shape[1]

    # This is a way to create every possible combination of nodes in the graph
    # Selecting the right nodes i.e neighbors is done later using the adjacency matrix
    t1 = feature_matrix.repeat_interleave(N, dim=0)
    t2 = feature_matrix.repeat(N, 1)
    new = torch.cat([t1, t2], dim=1)

    return new.view(N, N, 2 * D)
