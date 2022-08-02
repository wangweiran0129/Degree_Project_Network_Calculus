from pbzlib import write_pbz, open_pbz
from output.adversaril_attack_graph_generation.attack_pb2 import *


def write_attacked_network(network, perturbed_graph, filename):
    
    objs = [Network(id=1)]
    
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

    with write_pbz(filename, "/Users/wangweiran/Desktop/MasterDegreeProject/Adversarial_Attack_GNN/DeepFP_gnn-main/src/output/adversaril_attack_graph_generation/attack.descr") as w:
        for obj in objs:
            w.write(obj)