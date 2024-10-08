# Master Degree Project - Analysis of Flow Prolongation Using Graph Neural Network (GNN) in FIFO Multiplexing System

## Environment and Prerequisites
The whole project is run on my Macbook Pro (Intel Chip) and the EPFL servers. It's worth mentioning that for EPFL servers, only Izar supports NVIDIA GPUs for GNN model training and the adversarial attack.
- IDE: Visual Studio Code
- Python: 3.8.5 / 2.7.5(EPFL Server Python Version)
- Java: 16.0.2
- [NetCal/DNC](https://github.com/NetCal/DNC): v2.8.0
- [IBM ILOG CPLEX Optimization Studio](https://www.ibm.com/docs/en/icos/20.1.0?topic=cplex-setting-up-gnulinuxmacos): 20.1.0 （This is used only for LUDB-FF calculation）
- Python packages information can be found at [requirements.txt](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/requirements.txt)

## EPFL SCITAS Server Configuration
Java is mainly used for NetCal/DNC. If there is no Java config. in your computer, please download it from the official website. Considering students are not the admins of EPFL servers, it is therefore a good idea to put Java configuration into a user-defined file under the home path, i.e., add the following three lines into the .bashrc file.
```
$ export JAVA_HOME=/home/<your_epfl_account>/jdk-16.0.2
$ export PATH=$JAVA_HOME/bin:$PATH
$ export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
```
Python is used for GNN training and the subsequent Adversarial Attack. It is highly recommended to use Python in a user-defined virtual environment on EPFL servers.

To create a python virtual environemnt:  
```
$ virtualenv --system-site-packages venvs/<env_name>
```

To activate and use the virtual environment:  
```
$ source venvs/<env_name>/bin/activate
```

Confirm python3 is the default python version instead of python2:  
```
$ module spider python
$ module spider python/<python_version>
$ module load gcc/<gcc_version> python/<python_version>
```

Install the required python packages: [requirements.txt](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/requirements.txt). Please pay great attention to torch-related packages. There are several cuda versions available on the IZAR server. However, as far as to our test, ```cuda/10.2.89``` is the only one which meets all required dependencies for both PyTorch and PyG. Therefore, for torch-related packages installation on the server, please follow the steps:
```
$ module load cuda/10.2.89
$ pip3 install torch
$ python -c "import torch; print(torch.__version__)" # confirm the torch version
$ python -c "import torch; print(torch.version.cuda)" # confirt the cuda version
$ pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
$ pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
$ pip install torch-geometric
```

It is also worth mentioning that if you want to import ```torch_geometric```, please also import ```scipy``` at the same time.

For more information on Python environment, it is recommended to read the SCITAS Documentation on [Python Virtual Environment](https://scitas-data.epfl.ch/confluence/display/DOC/Python+Virtual+Environments).

For the writing of .sh script used for running the code on EPFL servers, please refer to [First steps on the clusters](https://scitas-data.epfl.ch/confluence/display/DOC/First+steps+on+the+clusters). [Here](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DeepFP_gnn-main/src/data/large_network_generation/netgen.sh) is also one example.

If the documentation of SCITAS website cannot be openned, please turn on the EPFL VPN. If more information about the server configuration is needed, please contact the EPFL SCITAS service desk 1234@epfl.ch

## Codes Description
### DeepFP_gnn-main
- [potential_attack_target.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/fgsm_adversarial_attack/potential_attack_target.py):  Find the potential attack target(s) based on the two prediction values of GNN. For the first prediction value, it is mainly based on the flow of interest, and tells whether other flows of this topology are worth prolonging (the criteria value is 0.5). For the second prediction values, they are based on the possibile prolonging flows whose criteria is the highest values.
- [adversarial_attack.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/fgsm_adversarial_attack/adversarial_attack.py): The adversarial attack using the FGSM to change the server rate, server latency, flow rate and flow burst.
- [graph_transformer.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/gnn_fp/preprocess/graph_transformer.py): Transform a human reading-friendly graph to a GNN-based recognized graph.
- [prepare_dataset_pmoo.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/gnn_fp/preprocess/prepare_dataset_pmoo.py):  Transform .pbz format dataset to a .pickle format dataset based on the NetCal method pmoo. The .pickle format dataset is based on matrices and will be used for the trainning of GNN and the FGSM adversarial attack. One graph.pickle file is the network feature matrices, and anther target.pickle file is the correct flow prolongation matrices.
- [prepare_dataset_deborah.py]https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/gnn_fp/preprocess/prepare_dataset_deborah.py): Same above while the NetCal method is changed to deborah.
- [large_network_generation_pbz.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/dataset_creation/large_network_generation/large_network_generation_pbz.py): Generate a large-scale network dataset for adversarial attack, i.e., the number of servers and flows are large.
- [large_network.proto](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/dataset_creation/large_network_generation/network_structure/network_structure.proto): Define the dataset structure for large-scale network. It is mainly used by ```large_network_generation_pbz.py```.
- [large_network.descr](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/dataset_creation/large_network_generation/network_structure/network_structure.descr) and [large_network_pb2.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/dataset_creation/large_network_generation/network_structure/network_structure_pb2.py): A systematic generated file after compilation from ```large_network.proto```. The following read and write opeartions on large network dataset structure will mainly based on these two.
- [ggnn_pmoo.pt](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/gnn_fp/model/ggnn_pmoo.pt): A pre-trained GNN model based on PMOO NetCal method. You are also welcomed to train the model on your own. It is trained on CPU.
- [deepfpPMOO.pt](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/gnn_fp/model/deepfpPMOO.pt) The same GNN model based on the PMOO NetCal method. It is trained on GPU. As far as I know, a CPU trained model cannot be loaded on a GPU machine, and vice versa.
- [gnn.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/gnn_fp/model/gnn.py): Define the neural network structure of GNN.
- [train_model.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/gnn_fp/model/train_model.py): The process of training the GNN model, and evaluation of the model accuracy.
- [predict_model.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/gnn_fp/predict/predict_model.py): Use the pre-trained GNN model to predict the flow prolongation on a new network topology configuration, and store the result in a .pbz format.
- [predict_original_networks.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/gnn_fp/predict/predict_original_networks.py): Predict the flow prolongation by using the GNN model for a new network topology. The prediction can be done for the topologies before the adversarial attack. This code highly depends on the ```predict_model.py```.
- [predict_attacked_networks.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/gnn_fp/predict/predict_attacked_networks.py): Similiar to ```predict_original_networks.py``` above, but predict the flow prolongations on the network after the adversarial attack.
- [network_writer.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/gnn_fp/predict/network_writer.py): Plot the network topology stored in .pbz.
- [write_attacked_network.py](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/gnn_fp/predict/write_attacked_network.py): Write the network after the adversarial attack into a .pbz file according to the attack.descr description and also calculate the delay bound for this network.

### DNC
- [NetCal4Python.java](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/DNC/src/main/java/org/networkcalculus/dnc/degree_project/NetCal4Python.java) : Calculate the delay bounds on a given network setting.
 - [NetCal.jar](https://github.com/wangweiran0129/Degree_Project_Network_Calculus/blob/master/GNN_FGSM/src/NetCal.jar) : The .jar of this NetCal/DNC repo, for the convenience of running the NetCal/DNC on the EPFL server by command ```java -jar NetCal.jar```.

## Reproduction and Usage Specification
For the storage of analysis, it is highly recommended to create a directory for the result analysis like this:
```
.
|-- DeepFP_gnn-main
|-- DNC
└-- Network_Information_and_Analysis
    └-- original_topology
    |   |-- after_fp
    |   └-- before_fp
    |       |-- pbz
    |       └-- pickle
    |           |-- graphs
    |           └-- targets
    |-- after_topology
    |   |-- after_fp
    |   └-- before_fp
    └-- prediction_value
```
### Train the GNN Model
1. Please download the dataset by followoing the this [link](https://github.com/fabgeyer/dataset-rtas2021).
2. Depending on the Network Calculus methods, choose the corresponding method python file. It will change the dataset stored in .pbz format to a .pickle format where network topologies are changed to network matrices. The matrices will then be used for the trainning process of the model/

    For the pickle files used for training:
    ```
    [data]$ python3 -m prepare_dataset_pmoo/deborah "<dataset-train.pbz file path>" "True"
    ```
    For the pickle files used for testing:
    ```
    [data]$ python3 -m prepare_dataset_pmoo/deborah "<dataset-evaluation.pbz file path>" "False"
    ```
3. Train the model:
    ```
    [model]$ python3 -m train_model "<train_graphs_pmoo.pickle file path>" "<train_targets_pmoo.pickle file path>" "<test_graphs_pmoo.pickle file path>" "<test_targets_pmoo.pickle file path>" <learning rate> <epoch number>
    ```
    On IZAR server, run the script:
    ```
    [<account@izar> model]$ sbatch train.sh
    ```
4. Use the trained model to predict the flow prolongations on a brand new network topology.
    ```
    [output]$ python3 -m predict_network <new network topology>, <foi id>, <pre-trained model>, <output file name>)
    ```

### Do the Adversarial Attack on the Network Topologies
1. Compile under the data/large_network_generation/network_structure folder if you cannot find the ```network_structure_pb2.py``` and ```network_strcture.descr```. This will compile and generate two necessary data structure files used for the creation of a new dataset. Furthermore, pmoo will be the main NetCal method used in the Adversarial Attack.
    ```
    [data/large_network_generation/network_structure]$ Make
    ```
2. Generate a larger size of dataset. This is mainly because the network size (# servers, # flows) is small in the existing [dataset](https://github.com/fabgeyer/dataset-rtas2021). This will output ```dataset-attack-large.pbz```. Please start up the NetCal.jar beforehand. There is also a ```temp4pred.pbz``` file in this folder, but ignore it. This is just an intermediate .pbz file for the proloned flow PMOO delay bound calculation. When the ```dataset-attack-large.pbz``` is created, please move it to ```Network_Information_and_Analysis/original_topology/before_fp/``` folder.
    ```
    [data/large_network_generation]$ java -jar NetCal.jar
    [data/large_network_generation]$ python3 -m dataset_network_generation_pbz <topo id start> <topo id end> <deepfpPMOO.pt/ggnn_pmoo.pt model path>
    ```
    On IZAR server, run the script
    ```
    [<account@izar> data/large_network_generation]$ sbatch netgen.sh
    ```
3. Transfer the pbz format file to .pickle format. The .pickle files are usually used as an input to the GNN model due to the matrix characteristic. Basically, it the two output files are named after ```test_graphs.pickle``` and ```test_targets.pickle```. Please re-name them to ```attack_graphs.pickle``` and ```attack_targets.pickle```. Besides, please move them to ```Network_Information_and_Analysis/original_topology/before_fp/```.
    ```
    [data]$ python3 -m prepare_dataset_pmoo "<dataset-attack-large.pbz file path>"
    ```
    On IZAR server, run the script 
    ```
    [<account@izar> data]$ sbatch pbz2pickle.sh
    ```
4. Make the prediction on the original topologies (The topologies before the attack and before the GNN prediction). This step will output two files: One is ```prediction_<topo_id>.csv``` where two prediction values are stored inside, in ```Network_Information_and_Analysis/prediction_value/```). Another is ```original_<topo_id>_<foi_id>.pbz``` which is the flow prolongation for the original topology (the topology before the Adversarial Attack), in (```Network_Information_and_Analysis/original_topology/after_fp/```).  Please be careful that in the prolongation of the original network, the delay bound after the flow prolongation will also be calculated, so please let the NetCal.jar run beforehand.
    ```
    [output]$ python3 -m predict_original_networks "<deepfpPMOO.pt/ggnn_pmoo.pt model path>" "<dataset-attack-large.pbz file path>"
    ```
    On IZAR server, run the script
    ```
    [<account@izar> output]$ sbatch pdognw.sh
    ```
5. Find the potential attack targets. Given the input of ```prediction_<topo_id>.csv```, it will output two files for the potential attack targets (```potential_attack_target1.csv``` and ```potential_attack_target2.csv```) based on the two predition values of GNN.
    ```
    [analysis]$ python3 -m potential_attack_target "</prediction_value/ folder path>"
    ```
    On IZAR server, run the script
    ```
    [<account@izar> analysis]$ sbatch poattar.sh
    ```
6. Use the [Fast Gradient Sign Method (FGSM)](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html) to do the adversarial attack on the network features. It will output the topologies after the attack (.pbz) into the ```Network_Information_and_Analysis/attacked_topology/before_fp``` folder. Besides, PMOO delay bounds will also be calculated for the network after the attack, so please set on the NetCal.jar
    ```
    [analysis]$ python -m adversarial_attack "<model path>" "<potential attack target path (the .csv file)>" "<dataset-attack-large.pbz file path>" "<attack_graphs.pickle file path>" "<attack_targets.pickle file path>"
    ````
    On IZAR server, run the script
    ```
    [<account@izar> analysis]$ sbatch fgsm.sh
    ```
7. Make the prediction on the flow prolongation of the networks after the adversarial attack. It will output the topologies into the ```Network_Information_and_Analysis/attacked_topologies/after_fp``` folder. PMOO delay bounds will also be calculated for the networks which are after the attack and after the flow prolongation, so please set on the NetCal.jar.
    ```
    [output]$ python3 -m predict_attacked_networks "<model path>" "<the folder path of the network topologies after the attack, but before the flow prolongation>"
    ```
    On IZAR server, run the script
    ```
    [<account@izar> output]$ sbatch pdatnw.sh
    ```

## Disclaimer and Special Acknowledgement
- Project Student: Weiran Wang (weiran.wang@epfl.ch / weiranw@kth.se)
- Ph.D. Advisor: Hossein Tabatabaee (hossein.tabatabaee@epfl.ch)
- Supervisor: Prof. Jean-Yves Le Boudec (jean-yves.leboudec@epfl.ch)
- Special Acknowledgement to:   
    - Bondorf Steffen (Bondorf.Steffen@ruhr-uni-bochum.de) and Alexander Scheffler (Alexander.Scheffler@ruhr-uni-bochum.de) for the support of the usage of NetCal/DNC tool.
    - Etienne Orliac (etienne.orliac@epfl.ch) for the support of the configuration of torch enviroment on EPFL IZAR cluster.
    - Hadidane Karim (karim.hadidane@epfl.ch) for providing the base code of GNN reproduction.