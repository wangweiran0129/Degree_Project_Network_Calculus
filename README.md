# Master Degree Project - Analysis of Flow Prolongation Using Graph Neural Network (GNN) in FIFO Multiplexing System

## Environment and Prerequisites
The whole project is run on my Macbook Pro (Intel Chip) and the EPFL servers. It's worth mentioning that for EPFL servers, only Izar supports NVIDIA GPUs for GNN model training and the adversarial attack.
- IDE: Visual Studio Code
- Python: 3.8.5
- Java: 16.0.2
- [NetCal/DNC](https://github.com/NetCal/DNC): v2.8.0
- [IBM ILOG CPLEX Optimization Studio](https://www.ibm.com/docs/en/icos/20.1.0?topic=cplex-setting-up-gnulinuxmacos): 20.1.0 （This is used only for LUDB-FF calculation）
- Python packages information can be found at [requirements.txt](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/requirements.txt)

## EPFL SCITAS Server Configuration
Considering students are not the admins of EPFL servers, it is therefore a good idea to put Java configuration into a user-defined file under the home path, i.e., add the following three lines into the .bashrc file.  
```
export JAVA_HOME=/home/<your_epfl_account>/jdk-16.0.2
export PATH=$JAVA_HOME/bin:$PATH
export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
```

For python environment, it is recommended to read the SCITAS Documentation on [Python Virtual Environment](https://scitas-data.epfl.ch/confluence/display/DOC/Python+Virtual+Environments).

For the writing of .sh script used for running the code on EPFL servers, please refer to one example [here](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/data/large_network_generation/netgen.sh).

## Codes Description
### DeepFP_gnn-main
- [netcal_analysis.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/analysis/netcal_analysis.py) : Call the ```NetCal4Python.java``` to calculate the delay bound of a given topology (The topology is stored in a .pbz format).
- [potential_attack_target.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/analysis/potential_attack_target.py) :  Find the potential attack target(s) based on the two prediction values of GNN. For the first prediction value, it is mainly based on the flow of interest, and tells whether other flows of this topology are worth prolonging (the criteria value is 0.5). For the second prediction values, they are based on the possibile prolonging flows whose criteria is the highest values.
- [graph_transformer.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/data/graph_transformer.py) : Transform a human reading-friendly graph to a GNN-based recognized graph.
- [prepare_dataset_pmoo.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/data/prepare_dataset_pmoo.py) :  Transform .pbz format dataset to a .pickle format dataset based on the NetCal method pmoo. The .pickle format dataset is based on matrices and will be used for the trainning of GNN and the FGSM adversarial attack.
- [prepare_dataset_deborah.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/data/prepare_dataset_deborah.py) : Same above while the NetCal method is changed to deborah.
- [large_network_generation_pbz.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/data/large_network_generation/large_network_generation_pbz.py): Generate a large-scale network dataset for adversarial attack, i.e., the number of servers and flows are large.
- [large_network.proto](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/data/large_network_generation/large_network/large_network.proto): Define the dataset structure for large-scale network. It is mainly used by ```large_network_generation_pbz.py```.
- [large_network.descr](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/data/large_network_generation/large_network/large_network.descr) and [large_network_pb2.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/data/large_network_generation/large_network/large_network_pb2.py): A systematic generated file after compilation from ```large_network.proto```. The following read and write opeartions on large network dataset structure will mainly based on these two.
- [netgen.sh](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/data/large_network_generation/netgen.sh) : A script needed for the EPFL server. This script is for the large-scale network dataset generation. For running operations on other codes, please change the correspoding content in this script.
- [adversarial_attack.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/model/adversarial_attack.py) : The adversarial attack using the FGSM to change the server rate, server latency, flow rate and flow burst.
- [ggnn_pmoo.pt](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/model/ggnn_pmoo.pt) : A pre-trained GNN model based on pmoo NetCal method. You are also welcomed to train the model on your own.
- [gnn.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/model/gnn.py) : Define the neural network structure of GNN.
- [train_model.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/model/train_model.py) : The process of training the GNN model, and evaluation of the model accuracy.
- [predict_multiple_networks.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/model/predict_multiple_networks.py) : Predict the flow prolongation by using the ```ggnn_pmoo.pt``` model. The prediction can be done for the topologies before/after the adversarial attack. This code highly depends on the ```predict_model.py```.
- [network_writer.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/output/network_writer.py) : Plot the network topology stored in .pbz.
- [predict_model.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/output/predict_model.py) : Use the pre-trained GNN model to predict the flow prolongation on a new network topology configuration, and store the result in a .pbz format.
- [write_attacked_network.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/output/write_attacked_network.py) : Write the network after the adversarial attack into a .pbz file according to the attack.descr description and also calculate the delay bound for this network.

### DNC
- [NetCal4Python.java](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DNC/src/main/java/org/networkcalculus/dnc/degree_project/NetCal4Python.java) : Calculate the delay bounds on a given network setting.
 - [NetCal.jar](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/NetCal.jar) : The .jar of this NetCal/DNC repo, for the convenience of running the NetCal/DNC on the EPFL server by command ```java -jar NetCal.jar```.