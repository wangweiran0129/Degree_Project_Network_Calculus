# This code is written by Karim Hadidane .
# For any questions or problems, please contact the author of the code at (karim.hadidane@epfl.ch)

# !/usr/bin/env python3

import pbzlib
import argparse
import enum
import networkx as nx
from copy import deepcopy
from collections import defaultdict

# Maybe won't need it for now
NodeType= enum.IntEnum("NodeType", [
    "Server", "Flow", "Flow_oi", "Prolong"
])


# Returns the base graph i.e does not depend on the flow of interest, without prolongation nodes
# A network is given as the proto structure
def net2basegraph(net):
    """
    A method that returns the base graph as a networkx graph object.
    The nodes are given prefixes to indicate their types: servers, flows or prolongations, for visualization purposes
    :param net: the network considered (proto object)
    :return: the base graph of the form of networkx graph object, a dictionary of flow paths of the form flow_id: [servers]
    """
    # Initialize empty graph
    G = nx.Graph()

    # Initialize nodes features dictionaries
    servers_rates = {}
    servers_latencies = {}
    flows_rates = {}
    flows_bursts = {}

    # Dictionary to save the path of each flow
    flow_paths = {}

    servers_list = []
    servers_edges = []

    # Add server nodes to the graph
    for server in net.server:
        server_id = "s_" + str(server.id)

        # Add edges between servers nodes
        if len(servers_list) != 0:
            servers_edges.append((servers_list[-1], server_id))
        servers_list.append(server_id)

        G.add_node(server_id, ntype=NodeType.Server)
        servers_rates[server_id] = server.rate
        servers_latencies[server_id] = server.latency

    G.add_edges_from(servers_edges)

    # Add flow nodes
    for flow in net.flow:
        flow_id = "f_" + str(flow.id)
        G.add_node(flow_id, ntype=NodeType.Flow)
        flows_rates[flow_id] = flow.rate
        flows_bursts[flow_id] = flow.burst
        servers_in_path = list(map(lambda x: (flow_id, "s_" + str(x)), list(flow.path)))
        G.add_edges_from(servers_in_path)
        flow_paths[flow.id] = list(flow.path)

    # Add attributes of the nodes i.e service curves and arrival curves parameters
    nx.set_node_attributes(G, servers_rates, name="rate_server")
    nx.set_node_attributes(G, flows_rates, name="rate_flow")
    nx.set_node_attributes(G, servers_latencies, name="latency")
    nx.set_node_attributes(G, flows_bursts, name="burst")

    return G, flow_paths


def valid_prolongation(flow, s, flow_paths, foi_id):
    """
    A method to check if a prolongation is valid, the check depends on the flow to prolong, the server and the foi
    :param flow: the flow to prolong
    :param s: server to prolong to
    :param flow_paths: the paths of flows in the network
    :param foi_id: the flow of interest
    :return: True if valid prolongation otherwise False
    """
    path_foi = flow_paths[foi_id]
    path = flow_paths[flow]

    last_server_in_flow_path = path[-1]

    # Ordering of the servers in a flow : Desc for descending and Asc for ascending
    ordering = "Asc" if path_foi[-1] > path_foi[-2] else "Desc"

    # We add prolongation nodes if it is the last server in the flow path 
    condition1 = last_server_in_flow_path == s
    # Or it follows the ordering of the paths in the network
    condition2 = s > last_server_in_flow_path if ordering == "Asc" else s < last_server_in_flow_path
    condition3 = last_server_in_flow_path != path_foi[-1]
    condition4 = len(set(path_foi) & set(path)) > 0
    return (condition1 or condition2) and condition3 and condition4

def prolong_graph(G_in, foi_id, flow_paths):
    """

    :param G_in: the base graph, networkx graph object
    :param foi_id: the id of the flow of interest
    :param flow_paths: the paths of flows in the network
    :return: networkx graph object graph with prolongation nodes added,
    :return: a dictionary of the prolongations flow: prolongation nodes
    """
    G = deepcopy(G_in)

    foi_node_id = "f_" + str(foi_id)

    # Change the type of the flow of interest
    nx.set_node_attributes(G, name="ntype", values={foi_node_id: NodeType.Flow_oi})

    # all flows excluding the flow of interest
    all_other_flows = [x for x in G.nodes if (x.startswith("f_") and x != foi_node_id)]

    # dictionary for flow: prolongation nodes
    prolongations = defaultdict(list)

    for flow in all_other_flows:
        for s_idx, s in enumerate(flow_paths[foi_id]):
            flow_id = int(flow[flow.index('_') + 1:])
            if valid_prolongation(flow_id, s, flow_paths, foi_id):
                prolong_id = "p" + str(flow_id) + "_" + str(s)
                G.add_node(prolong_id, ntype=NodeType.Prolong, hop=s_idx + 1)
                G.add_edge(prolong_id, flow)
                G.add_edge("s_" + str(s), prolong_id)
                prolongations[flow_id].append(prolong_id)

    # identifier for nodes
    ids = dict(zip(G.nodes(), range(G.number_of_nodes())))

    return G, prolongations, ids
