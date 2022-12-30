# This code is written by Weiran Wang
# For any questions, please contact the author of the code at (weiran.wang@epfl.ch)

from pbzlib import write_pbz
from py4j.java_gateway import JavaGateway
import sys
sys.path.insert(0, "../")
from data.large_network_generation.network_structure.network_structure_pb2 import *


def write_attacked_network(network, perturbed_graph, foi, filename):

    """
    A method that writes the network after the adversarial attack
    into a protobuf file according to the attack.descr description file
    and also calculate the delay bound for this network
    
    :param network: the original network before the attack
    :perturbed_graph: the network matrix after the attack
    :param foi: the foi id
    :param filename: the output filename
    """

    # Connect to the JVM
    gateway = JavaGateway()
    double_class = gateway.jvm.double
    int_class = gateway.jvm.int
    
    objs = [Network(id=0)]
    
    for s in network.server:
        p = objs[0].server.add()
        p.id = s.id
        p.rate = perturbed_graph[s.id][4]
        p.latency = perturbed_graph[s.id][5]
    
    for f in network.flow:
        p = objs[0].flow.add()
        p.id = f.id
        p.rate = perturbed_graph[f.id + len(network.server)][6]
        p.burst = perturbed_graph[f.id + len(network.server)][7]
        # The server order is ascending
        if f.path[0] < f.path[-1]:
            for flow_path in range(f.path[0], f.path[-1]+1):
                p.path.append(flow_path)
        # The server order is descending
        if f.path[0] > f.path[-1]:
            for flow_path in range(f.path[0], f.path[-1]-1, -1):
                p.path.append(flow_path)
        # There is only server in this flow
        if f.path[0] == f.path[-1]:
            p.path.append(f.path[0])

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

    # Write the network topology
    with write_pbz(filename, "../data/large_network_generation/network_structure/network_structure.descr") as w:
        for obj in objs:
            w.write(obj)
