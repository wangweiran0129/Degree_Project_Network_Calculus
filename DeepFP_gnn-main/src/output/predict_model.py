# This code is written by Karim Hadidane, and changed by Weiran Wang.
# For any questions or problems, please contact the author of the code at (weiran.wang@epfl.ch)


import sys, re, os
sys.path.insert(0, "../")
from data.graph_transformer import *
from data.prepare_dataset_pmoo import *
from data.prepare_dataset_deborah import *
from data.large_network_generation.network_structure.network_structure_pb2 import *
from model.train_model import *
from pbzlib import write_pbz
from py4j.java_gateway import JavaGateway


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
    sinks = idxmask[torch.where(prolongations_deepfp[0])[0]].cpu()

    # Create a dictionary nodeids and flow id
    inv_map = {v: k for k, v in node_ids.items()}

    z = [inv_map[x] for x in np.array(sinks)]
    to_be_prolonged = {get_flowid_from_prolongation_node_name(k): get_serverid_from_prolongation_node_name(k) for k in
                       z}

    for flow, server in to_be_prolonged.items():
        start_sink_dict[flow][1] = server

    write_network(network, start_sink_dict, foi_id, output_file)

    return graph, out1, out2, start_sink_dict


def predict_sink_sever(network, foi_id, model):
    """
    A method that uses the pre-trained model trained to predict the sink server.
    This function is mainly used in the process of creating a larger dataset
    :param network: a network topology stored in the .pbz file
    :param foi_id: the flow of interest
    :param model: the pre-trained model
    :return: the start sink server dictionary
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
        return

    idxmask = torch.where(graph.mask)[0]
    output_prolongations = torch.index_select(out2.view(-1), 0, idxmask)

    pro_nodes = graph.x[graph.mask]

    prolongations_deepfp = create_output_vector(pro_nodes, output_prolongations, 1)
    sinks = idxmask[torch.where(prolongations_deepfp[0])[0]].cpu()

    # Create a dictionary nodeids and flow id
    inv_map = {v: k for k, v in node_ids.items()}

    z = [inv_map[x] for x in np.array(sinks)]
    to_be_prolonged = {get_flowid_from_prolongation_node_name(k): get_serverid_from_prolongation_node_name(k) for k in z}

    for flow, server in to_be_prolonged.items():
        start_sink_dict[flow][1] = server

    return start_sink_dict


def write_network(network, flows_start_sink, foi, filename):
    """
    A method that writes the network generated into a protobuf file according to the attack.descr description file
    :param network: the network parameters
    :param flows_start_sink: the flows path
    :param foi: the foi id
    :param filename: output filename
    """

    # Connect to the JVM
    gateway = JavaGateway()
    double_class = gateway.jvm.double
    int_class = gateway.jvm.int

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
        flow_src = flows_start_sink[f.id][0]
        flow_dest = flows_start_sink[f.id][1]
        if flow_src == flow_dest:
            p.path.append(flow_src)
        else:
            for flow_path in range(flow_src, flow_dest+1):
                p.path.append(flow_path)

    num_server = len(network.server)
    num_flow = len(network.flow)
    
    # Calculate the delay bound
    # Collect the network features which will pass to the NetCal4Python.java
    server_rate_java = gateway.new_array(double_class, num_server)
    server_latency_java = gateway.new_array(double_class, num_server)
    flow_rate_java = gateway.new_array(double_class, num_flow)
    flow_burst_java = gateway.new_array(double_class, num_flow)
    flow_src_java = gateway.new_array(int_class, num_flow)
    flow_dest_java = gateway.new_array(int_class, num_flow)

    # Fill in the network features
    for s in objs[0].server:
        server_rate_java[s.id] = s.rate
        server_latency_java[s.id] = s.latency
        
    for f in objs[0].flow:
        flow_rate_java[f.id] = f.rate
        flow_burst_java[f.id] = f.burst
        flow_src_java[f.id] = f.path[0]
        flow_dest_java[f.id] = f.path[-1]

    # Get the network topology instance and call the delayBoundCalculation Java method
    network_topology = gateway.entry_point
    delay_bound = network_topology.delayBoundCalculation4OneFoi(server_rate_java, server_latency_java, flow_rate_java, flow_burst_java, flow_src_java, flow_dest_java, foi)
    print("pmoo delay bound: ", delay_bound, "\n")
    objs[0].flow[foi].pmoo.delay_bound = delay_bound

    print("path : ", os.getcwd())

    with write_pbz(filename, "../data/large_network_generation/network_structure/network_structure.descr") as w:
        for obj in objs:
            w.write(obj)


def get_flowid_from_prolongation_node_name(s):
    flow = int(re.search(r"\d+", s).group())
    # flow = int(s[s.index("_") - 1])
    return flow


def get_serverid_from_prolongation_node_name(s):
    server_temp = re.search(r"_\d+", s).group()
    server = int(re.search(r"\d+", server_temp).group())
    # server = int(s[s.index("_") + 1])
    return server
