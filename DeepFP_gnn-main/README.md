# DeepFP GNN
This repository contains the code for reproducing the results of the paper ["Tightening Network Calculus Delay Bounds by
Predicting Flow Prolongations in the FIFO Analysis"](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9470481) published at RTAS 2021 conference by authors Fabien Geyer, Steffen Bondorf and Alexander Scheffler.

## Dataset
The dataset used is the same used by the paper and can be downloaded [here](https://github.com/fabgeyer/dataset-rtas2021).


## Data Preprocessing
The dataset consists in networks with different sizes and topologies. For some flows of a network, combinations of flow prolongations were explored. 
For each pair of network (servers and flows) and a flow of interest, a graph transformation is to be done to translate it to a graph. See Algorithm 1.

The training dataset is `dataserv.ub.tum.de/dataset-train.pbz` and the test dataset is `dataserv.ub.tum.de/dataset-evaluation.pbz`. 

The code for preparing the dataset (graph transformation, selecting the best target combinations, creating a Data object for the graph etc) is found in `prepare_dataset.py` and `graph_transformer.py` (prepare_dataset method).

We encourage to use the `prepare_dataset` method for both the training dataset and test dataset files and save the outputs in `pickle` format (see to_pickle parameter in the method). You can then use the pickle files in the training process.

## Training
The code for training and testing is in `gnn.py`. 

We get good results using SGD optimizer with a learning rate `0.005`.
We don't use batching here i.e gradients are updated after seeing one graph in the training. 

## Prediction for new graphs
After training the graph, you can use the model to predict the prolongations for a new graph. 
The method `predict_network` takes as input a network object (same format as in the dataset i.e protobuf object), the flow of interest identifier and the model to use. This will do the graph transformation and run the model on the resulting graph and generate the flow prolongations predicted. 
The results will be written in an output file in the same format of the input dataset, with a slightly different structure. See the structure of the output file in `output.proto`.


