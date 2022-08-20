# This code is written by Weiran Wang
# For any questions, please contact the author of this code at (weiran.wang@epfl.ch)

import random
import collections
import torch
from tqdm import tqdm
from pbzlib import open_pbz, write_pbz
from scipy.sparse import coo_matrix
from py4j.java_gateway import JavaGateway
import sys
sys.path.insert(0, "../../")
from output.predict_model import *


def large_network_random_search(num_topo):
    """
    This function will generate a large-scale network dataset for adversarial attack
    The number of network topologies depends on the parameter passing 'num_topology',
    and will save the network results in a pbz format. The explored combination of
    flow prolongation is found by a random search, similar to exhaustive search, but
    with fewer numbers of search, and randomly choose the flows to prolong.

    :param num_topo: the number of topology in this dataset 
    """

    # Connect to the JVM
    gateway = JavaGateway()
    double_class = gateway.jvm.double
    int_class = gateway.jvm.int

    # objs store the network information
    objs = []

    for topo_id in range(num_topo):

        print("----- topo id : ", topo_id, " -----")

        num_server = random.randint(30, 40)
        num_flow = random.randint(300, 400)

        # To simplify the calculation in the adverseari attack process
        # There will be only one foi in each topology
        foi = random.randint(0, num_flow-1)

        objs.append(Network(id = topo_id))

        # Add server information
        for i in range(num_server):
            server = objs[topo_id].server.add()
            server.id = i
            server.rate = random.uniform(0.05, 1)
            server.latency = random.uniform(0.01, 1)
        
        # Add flow information
        for i in range(num_flow):
            flow = objs[topo_id].flow.add()
            flow.id = i
            flow.rate = random.uniform(0.00001, 0.00005)
            flow.burst = random.uniform(0.01, 1)

            # I want the foi to cross nearly the whole topology due to the NetCal characteristics
            if flow.id == foi:
                flow_src = random.randint(0, 5)
                flow_dest = num_server-1
            else:
                flow_src = random.randint(0, num_server-1)
                flow_dest = random.randint(flow_src, num_server-1)

            # Add servers to the flow path
            if flow_src == flow_dest:
                flow.path.append(flow_src)
            else:
                for flow_path in range(flow_src, flow_dest+1):
                    flow.path.append(flow_path)

        # Create a list to find which flows are on each server
        flows_in_servers_temp = collections.defaultdict(list)
        for s in objs[topo_id].server:
            for f in objs[topo_id].flow:
                if s.id in f.path:
                    flows_in_servers_temp[s.id].append(f.id)

        # Find the first server where there is a flow passing by
        first_useful_server = 0
        for s in objs[topo_id].server:
            if len(flows_in_servers_temp[s.id]) != 0:
                first_useful_server = s.id
                break

        # Delete the server where there are no flows passing by
        useless_server = []
        for s in objs[topo_id].server:
            if len(flows_in_servers_temp[s.id]) == 0:
                useless_server.append(s.id)
        useless_server.reverse()
        for sid in useless_server:
            del objs[topo_id].server[sid]
        num_server = num_server - len(useless_server)

        # Re-number the server id
        for s in objs[topo_id].server:
            s.id = s.id - first_useful_server
        
        # Update the flow path
        for f in objs[topo_id].flow:
            for p in range(len(f.path)):
                f.path[p] = f.path[p] - first_useful_server
        
        # Update the flows_in_servers
        flows_in_servers = collections.defaultdict(list)
        for s in objs[topo_id].server:
            for f in objs[topo_id].flow:
                if s.id in f.path:
                    flows_in_servers[s.id].append(f.id)

        # Modify the server rates to guarantee the flows on each server are within the server's capacity
        for sid in flows_in_servers:
            aggregated_flow_rate = 0
            for fid in flows_in_servers[sid]:
                aggregated_flow_rate = aggregated_flow_rate + objs[topo_id].flow[fid].rate
            if aggregated_flow_rate >= objs[topo_id].server[sid].rate:
                objs[topo_id].server[sid].rate = aggregated_flow_rate + 0.01

        # Collect the network features which will pass to the NetCal4Python.java
        server_rate_java = gateway.new_array(double_class, num_server)
        server_latency_java = gateway.new_array(double_class, num_server)
        flow_rate_java = gateway.new_array(double_class, num_flow)
        flow_burst_java = gateway.new_array(double_class, num_flow)
        flow_src_java = gateway.new_array(int_class, num_flow)
        flow_dest_java = gateway.new_array(int_class, num_flow)

        # Fill in network features
        for s in objs[topo_id].server:
            server_rate_java[s.id] = s.rate
            server_latency_java[s.id] = s.latency
        
        for f in objs[topo_id].flow:
            flow_rate_java[f.id] = f.rate
            flow_burst_java[f.id] = f.burst
            flow_src_java[f.id] = f.path[0]
            flow_dest_java[f.id] = f.path[-1]

        # Print some key parameters in this topology
        print("# server : ", num_server)
        print("# flow : ", num_flow)
        print("foi : ", foi, ", foi path : ", objs[topo_id].flow[foi].path[0], "->", objs[topo_id].flow[foi].path[-1])

        # Get the network topology instance and call the delayBoundCalculation Java method
        network_topology = gateway.entry_point
        original_delay_bound = network_topology.delayBoundCalculation4OneFoi(server_rate_java, server_latency_java, flow_rate_java, flow_burst_java, flow_src_java, flow_dest_java, foi)
        print("pmoo original delay bound: ", original_delay_bound, "\n")
        objs[topo_id].flow[foi].pmoo.delay_bound = original_delay_bound

        # Find the possible flows which can be prolonged
        # i.e., the flow sink/destination server id < foi sink/destination server id
        foi_sink_server = objs[topo_id].flow[foi].path[-1]
        possbile_flow_to_be_prolonged = []
        for f in objs[topo_id].flow:
            if f.path[-1] < foi_sink_server:
                possbile_flow_to_be_prolonged.append(f.id)
        length_possible_flow_to_be_prolonged = len(possbile_flow_to_be_prolonged)
        print("# possible flows ", length_possible_flow_to_be_prolonged)

        # Add the explore combination (potential flow prolongation)
        num_explore_combination = random.randint(num_flow*500, num_flow*1000)
        real_num_explore_combination = 0
        print("# explored combination ", num_explore_combination, "\n")

        for i in tqdm(range(num_explore_combination)):

            flow_to_be_prolonged = random.sample(possbile_flow_to_be_prolonged, random.randint(1, length_possible_flow_to_be_prolonged))
            flow_to_be_prolonged.sort()

            # Backup the original flow destination servers
            flow_prolonged_dest_java = gateway.new_array(int_class, num_flow)
            for fid in range(num_flow):
                flow_prolonged_dest_java[fid] = flow_dest_java[fid]
            
            # Prolong the flows
            for fid in flow_to_be_prolonged:
                original_sink_server = objs[topo_id].flow[fid].path[-1]
                prolonged_sink_server = random.randint(original_sink_server, foi_sink_server)
                flow_prolonged_dest_java[fid] = prolonged_sink_server
            
            # Compute the delay bound after the flow prolongation
            delay_bound_after_prolongation = network_topology.delayBoundCalculation4OneFoi(server_rate_java, server_latency_java, flow_rate_java, flow_burst_java, flow_src_java, flow_prolonged_dest_java, foi)
            
            # To reduce the size of dataset, only the tigher delay bounds are recorded
            if delay_bound_after_prolongation <= original_delay_bound:
                objs[topo_id].flow[foi].pmoofp.explored_combination.add()
                for fid in flow_to_be_prolonged:
                    objs[topo_id].flow[foi].pmoofp.explored_combination[real_num_explore_combination].flows_prolongation[fid] = flow_prolonged_dest_java[fid]
                objs[topo_id].flow[foi].pmoofp.explored_combination[real_num_explore_combination].delay_bound = delay_bound_after_prolongation
                real_num_explore_combination = real_num_explore_combination + 1
                print("BINGO! One tigher delay bound is found!")
                print("Iteration : ", i)
                print("# real explore combination : ", real_num_explore_combination)
        
        # The pmoofp.delay_bound is the tightest value among all the explored combinations
        if real_num_explore_combination != 0:
            min_fp_delay_bound = objs[topo_id].flow[foi].pmoofp.explored_combination[0].delay_bound
            for ex_com_idx in range(real_num_explore_combination):
                if objs[topo_id].flow[foi].pmoofp.explored_combination[ex_com_idx].delay_bound < min_fp_delay_bound:
                    min_fp_delay_bound = objs[topo_id].flow[foi].pmoofp.explored_combination[ex_com_idx].delay_bound
            print("min pmoofp delay bound : ", min_fp_delay_bound)
            objs[topo_id].flow[foi].pmoofp.delay_bound = min_fp_delay_bound
        print("")

    # Write the network topology into the pbz file
    with write_pbz("dataset-attack-large.pbz", "network_structure/network_structure.descr") as w:
        for obj in objs:
            w.write(obj)


def large_network_gnn_prediction(num_topo, model):
    """
    This function will generate a large-scale network dataset for adversarial attack
    The number of network topologies depends on the parameter passing 'num_topology',
    and will save the network results in a pbz format. The pmoo delay bound after flow
    prolongation is predicted by GNN. This way of dataset generation will save a huge
    running time. The gnn.py is copied and pasted here for the purpose of loading the
    gnn model.

    :param num_topo: the number of topology in this dataset 
    :param model: the pre-trained model based on pmoo
    """

    # Connect to the JVM
    gateway = JavaGateway()
    double_class = gateway.jvm.double
    int_class = gateway.jvm.int

    # objs store the network information
    objs = []

    for topo_id in tqdm(range(num_topo)):

        print("----- topo id : ", topo_id, " -----")

        num_server = random.randint(20, 30)
        num_flow = random.randint(200, 300)

        # To simplify the calculation in the adversearial attack process
        # There will be only one foi in each topology
        foi = random.randint(0, num_flow-1)

        objs.append(Network(id = topo_id))

        # Add server information
        for i in range(num_server):
            server = objs[topo_id].server.add()
            server.id = i
            server.rate = random.uniform(0.05, 1)
            server.latency = random.uniform(0.01, 1)
        
        # Add flow information
        for i in range(num_flow):
            flow = objs[topo_id].flow.add()
            flow.id = i
            flow.rate = random.uniform(0.00001, 0.00005)
            flow.burst = random.uniform(0.01, 1)

            # I want the foi to cross nearly the whole topology due to the NetCal characteristics
            if flow.id == foi:
                flow_src = random.randint(0, 5)
                flow_dest = num_server-1
            else:
                flow_src = random.randint(0, num_server-1)
                flow_dest = random.randint(flow_src, num_server-1)

            # Add servers to the flow path
            if flow_src == flow_dest:
                flow.path.append(flow_src)
            else:
                for flow_path in range(flow_src, flow_dest+1):
                    flow.path.append(flow_path)

        # Create a list to find which flows are on each server
        flows_in_servers_temp = collections.defaultdict(list)
        for s in objs[topo_id].server:
            for f in objs[topo_id].flow:
                if s.id in f.path:
                    flows_in_servers_temp[s.id].append(f.id)

        # Find the first server where there is a flow passing by
        first_useful_server = 0
        for s in objs[topo_id].server:
            if len(flows_in_servers_temp[s.id]) != 0:
                first_useful_server = s.id
                break

        # Delete the server where there are no flows passing by
        useless_server = []
        for s in objs[topo_id].server:
            if len(flows_in_servers_temp[s.id]) == 0:
                useless_server.append(s.id)
        useless_server.reverse()
        for sid in useless_server:
            del objs[topo_id].server[sid]
        num_server = num_server - len(useless_server)

        # Re-number the server id
        for s in objs[topo_id].server:
            s.id = s.id - first_useful_server
        
        # Update the flow path
        for f in objs[topo_id].flow:
            for p in range(len(f.path)):
                f.path[p] = f.path[p] - first_useful_server
        
        # Update the flows_in_servers
        flows_in_servers = collections.defaultdict(list)
        for s in objs[topo_id].server:
            for f in objs[topo_id].flow:
                if s.id in f.path:
                    flows_in_servers[s.id].append(f.id)

        # Modify the server rates to guarantee the flows on each server are within the server's capacity
        for sid in flows_in_servers:
            aggregated_flow_rate = 0
            for fid in flows_in_servers[sid]:
                aggregated_flow_rate = aggregated_flow_rate + objs[topo_id].flow[fid].rate
            if aggregated_flow_rate >= objs[topo_id].server[sid].rate:
                objs[topo_id].server[sid].rate = aggregated_flow_rate + 0.01

        # Collect the network features which will pass to the NetCal4Python.java
        server_rate_java = gateway.new_array(double_class, num_server)
        server_latency_java = gateway.new_array(double_class, num_server)
        flow_rate_java = gateway.new_array(double_class, num_flow)
        flow_burst_java = gateway.new_array(double_class, num_flow)
        flow_src_java = gateway.new_array(int_class, num_flow)
        flow_dest_java = gateway.new_array(int_class, num_flow)

        # Fill in network features
        for s in objs[topo_id].server:
            server_rate_java[s.id] = s.rate
            server_latency_java[s.id] = s.latency
        
        for f in objs[topo_id].flow:
            flow_rate_java[f.id] = f.rate
            flow_burst_java[f.id] = f.burst
            flow_src_java[f.id] = f.path[0]
            flow_dest_java[f.id] = f.path[-1]

        # Print some key parameters in this topology
        print("# server : ", num_server)
        print("# flow : ", num_flow)
        print("foi : ", foi, ", foi path : ", objs[topo_id].flow[foi].path[0], "->", objs[topo_id].flow[foi].path[-1])

        # Get the network topology instance and call the delayBoundCalculation Java method
        network_topology = gateway.entry_point
        original_delay_bound = network_topology.delayBoundCalculation4OneFoi(server_rate_java, server_latency_java, flow_rate_java, flow_burst_java, flow_src_java, flow_dest_java, foi)
        print("pmoo original delay bound: ", original_delay_bound, "\n")
        objs[topo_id].flow[foi].pmoo.delay_bound = original_delay_bound

        # Write a temporary network file for the input of prediction
        write_pbz("temp4pred.pbz", "network_structure/network_structure.descr", objs[topo_id])

        # Use the GNN model to predict the flow prolongation
        start_sink_dict = predict_sink_sever(next(open_pbz("temp4pred.pbz")), foi, model)

        # Backup the original flow destination servers
        flow_prolonged_dest_java = gateway.new_array(int_class, num_flow)
        for fid in range(num_flow):
            flow_prolonged_dest_java[fid] = flow_dest_java[fid]

        # Add the pmoofp block
        objs[topo_id].flow[foi].pmoofp.explored_combination.add()
        # Change the sink servers
        for f in start_sink_dict:
            if objs[topo_id].flow[f].path[-1] != start_sink_dict[f][1]:
                objs[topo_id].flow[foi].pmoofp.explored_combination[0].flows_prolongation[f] = start_sink_dict[f][1]
                flow_prolonged_dest_java[f] = start_sink_dict[f][1]

        # Compute the delay bound after the flow prolongation
        delay_bound_after_prolongation = network_topology.delayBoundCalculation4OneFoi(server_rate_java, server_latency_java, flow_rate_java, flow_burst_java, flow_src_java, flow_prolonged_dest_java, foi)
        if delay_bound_after_prolongation > original_delay_bound:
            print("The delay bound after fp is looser")
        objs[topo_id].flow[foi].pmoofp.explored_combination[0].delay_bound = delay_bound_after_prolongation
        objs[topo_id].flow[foi].pmoofp.delay_bound = delay_bound_after_prolongation
        print("pmoo delay bound after fp : ", delay_bound_after_prolongation)
        print("")
        
    # Write the network topology into the pbz file
    with write_pbz("dataset-attack-large.pbz", "network_structure/network_structure.descr") as w:
        for obj in objs:
            w.write(obj)


def main():

    model = torch.load("../../model/deepfpPMOO.pt")
    large_network_gnn_prediction(1000, model)

    # large_network_random_search(2)



if __name__ == "__main__":
    main()