# This code is written by Weiran Wang
# For any questions, please contact the author of this code at (weiran.wang@epfl.ch)

import random
import collections
import numpy as np
import csv
import time
import sys
sys.path.insert(0, "../")
from data.large_network_generation.network_structure.network_structure_pb2 import *
from tqdm import tqdm
from py4j.java_gateway import JavaGateway


def ex4tighterpmoofp():
   
    """
    This function will analyze the average iteration to find a tighter delay bound.
    This function will also analyze the average execution time to find a tighter delay bound.
    The # server will ranage from 10 to 30.
    In theory, the max # flow without redundance is #server*(#server+1)/2.
    But in this code, we choose #server*(#server+1)/4.
    Once there is a tighter delay bound found, the code will break, 
    and we will record the iteration and the exectuion. This function will be iterated for 10 times,
    and then we will take the average to avoid the outlier.
    """

    # Connect to the JVM
    gateway = JavaGateway()
    double_class = gateway.jvm.double
    int_class = gateway.jvm.int

    # objs store the network information
    objs = []

    iteration = []
    execution_time = []
    topo_id = 0

    for round in tqdm(range(2)):

        num_servers = np.arange(10, 12)
        
        for num_server in num_servers:

            num_flow = int((num_server+1)*num_server/4)

            # To simplify the calculation in the adverseari attack process, there will be only one foi in each topology
            foi = random.randint(0, num_flow-1)

            objs.append(Network(id = topo_id))

            # Add server information
            for i in range(num_server):
                server = objs[topo_id].server.add()
                server.id = i
                server.rate = random.uniform(0.05, 1)
                server.latency = random.uniform(0.01, 1)
        
            # Clarify the path for the foi
            foi_start_server = random.randint(0, 3)
            foi_sink_server = num_server - 1

            # Try to find all the possible flows so as to confirm that there are no redundant flows in the topology
            all_possible_flows = collections.defaultdict(list)
            fid = 0
            for s_src in range(num_server):
                for s_dest in range(s_src, num_server):
                    # Make sure the rest flows' paths are not the same with the foi's path
                    if s_src != foi_sink_server or s_dest != foi_sink_server:
                        if s_src != s_dest:
                            all_possible_flows[fid].append(s_src)
                            all_possible_flows[fid].append(s_dest)
                        else:
                            all_possible_flows[fid].append(s_src)
                        fid = fid + 1

            # Randomly pick up some flows from the all_possible_flows list
            selected_flows = random.sample(all_possible_flows.keys(), num_flow)
            selected_flow_index = 0

            # Add flow information
            for i in range(num_flow):
                flow = objs[topo_id].flow.add()
                flow.id = i
                flow.rate = random.uniform(0.0001, 0.0005)
                flow.burst = random.uniform(0.01, 1)

                # I want the foi to cross nearly the whole topology due to the NetCal characteristics
                if flow.id == foi:
                    flow_src = foi_start_server
                    flow_dest = foi_sink_server
                else:
                    flow_src = all_possible_flows[selected_flows[selected_flow_index]][0]
                    flow_dest = all_possible_flows[selected_flows[selected_flow_index]][-1]
                    selected_flow_index = selected_flow_index + 1

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

            # Add the explore combination (potential flow prolongation)
            num_explore_combination = random.randint(num_flow*500, num_flow*1000)
            real_num_explore_combination = 0
            print("# explored combination ", num_explore_combination, "\n")

            for i in range(num_explore_combination):

                # Now we begin to calculate the time
                start = time.time()

                flow_to_be_prolonged = random.sample(num_flow, random.randint(1, num_flow-1))
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
                    iteration.append(i)
                    print("We found a tighter delay bound at the iteration ", i)
                    
                    # Since we found a tighter delay bound, the time calculated will be stop
                    end = time.time()
                    exe_time = end - start
                    execution_time.append(exe_time)
                    print("execution time : ", exe_time)
                    
                    break
            
            topo_id = topo_id + 1

    # Write the results to a csv file
    with open("exhaustive_search_time_analysis.csv", "w") as csvfile:
        
        writer = csv.writer(csvfile)

        # Define the column name
        writer.writerow(["Round", "# Server", "Iteration", "Time"])
        
        # Write the contents
        num_server = 10
        for i in range(topo_id):
            writer.writerow([i, num_server, iteration[i], execution_time[i]])
            num_server = num_server + 1


if __name__ == "__main__":

    ex4tighterpmoofp()
