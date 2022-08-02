import sys
sys.path.insert(0, "../src/")
from data.graph_transformer import *
from data.prepare_dataset_pmoo import *
from data.prepare_dataset_deborah import *
from model.train_model import *
from output.output_pb2 import *
from pbzlib import write_pbz, open_pbz


def predict_network(network, foi_id, model, output_file="output.pbz"):
    """
    A method that uses the model trained to predict new network configuration and write it in proto file format
    :param network: network parameters
    :param foi_id: the flow of interest
    :param model: the model
    :param output_file: output file to generate
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create a base graph
    G, flows_path = net2basegraph(network)

    # prolong the graph with respect to the foi
    G_f, pro_dict, node_ids = prolong_graph(G, foi_id, flows_path)

    graph = graph2torch_pmoo(G_f, node_ids=node_ids)

    adj = prepare_adjacency_matrix(graph)

    out1, out2 = model(graph.to(device), adj.to(device))

    # HERE a dictionary flow: (start, sink)
    start_sink_dict = {k: [flows_path[k][0], flows_path[k][-1]] for k in flows_path.keys()}

    foi_idx = torch.where(graph.x[:, 2])[0]
    output_foi = torch.index_select(out1.view(-1), 0, foi_idx)
    predicted_label = 1 if output_foi.item() >= 0.5 else 0

    # If the prediction is that FP is not worth then write the same network
    if not predicted_label:
        write_network(network, start_sink_dict, output_file)
        return

    idxmask = torch.where(graph.mask)[0]
    output_prolongations = torch.index_select(out2.view(-1), 0, idxmask)

    pro_nodes = graph.x[graph.mask]

    prolongations_deepfp = create_output_vector(pro_nodes, output_prolongations, 1)
    sinks = idxmask[torch.where(prolongations_deepfp[0])[0]]

    # Create a dictionary nodeids and flow id
    inv_map = {v: k for k, v in node_ids.items()}

    z = [inv_map[x] for x in np.array(sinks)]
    to_be_prolonged = {get_flowid_from_prolongation_node_name(k): get_serverid_from_prolongation_node_name(k) for k in
                       z}

    for flow, server in to_be_prolonged.items():
        start_sink_dict[flow][1] = server

    write_network(network, start_sink_dict, output_file)

    return graph, out1, out2


def write_network(network, flows_start_sink, filename):
    """
    A method that writes the network generated into a protobuf file according to the output.descr description file
    :param network: the network parameters
    :param flows_start_sink: the flows path
    :param filename: output filename
    """
    objs = [Network(id=1)]

    for s in network.server:
        p = objs[0].server.add()
        p.id = s.id
        p.rate = s.rate
        p.latency = s.latency

    for f in network.flow:
        p = objs[0].flow.add()
        p.id = f.id
        p.rate = f.rate
        p.burst = f.burst
        p.start = flows_start_sink[f.id][0]
        p.sink = flows_start_sink[f.id][1]

    with write_pbz(filename, "/Users/wangweiran/Desktop/MasterDegreeProject/Degree_Project_Network_Calculus/DeepFP_gnn-main/src/output/output.descr") as w:
        for obj in objs:
            w.write(obj)



def get_flowid_from_prolongation_node_name(s):
    flow = int(s[s.index("_") - 1])
    return flow


def get_serverid_from_prolongation_node_name(s):
    server = int(s[s.index("_") + 1])
    return server
