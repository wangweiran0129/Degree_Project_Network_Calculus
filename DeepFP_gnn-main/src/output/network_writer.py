#!/usr/bin/env python3

from pbzlib import write_pbz, open_pbz
import messages_pb2
from graph_transformer import *
import networkx as nx


# def main():
#     objs = []
#     for i in range(1,2):
#         print(i)
#         objs.append(messages_pb2.Network(id=i))
#
#     # Method 1: Write messages incrementally
#     with write_pbz("output.pbz", "messages.descr") as w:
#         for obj in objs:
#             w.write(obj)
#
#     # Method 2: Write all messages at once
#     #write_pbz("output.pbz", "tests/messages.descr", *objs)
#
#     for msg in open_pbz("output.pbz"):
#         print(msg)
#
# # servers
def draw_net(servers, flows=None, flows_servers=None, foi= None):
    objs = [messages_pb2.Network(id=1)]

    for s in servers:
        p = objs[0].server.add()
        p.id = s

    for f in flows:
        p = objs[0].flow.add()
        p.id = f
        # p.path.append(flows_servers[f])

        for i in flows_servers[f]:
            p.path.append(i)

    with write_pbz("output.pbz", "messages.descr") as w:
        for obj in objs:
            w.write(obj)


    print("******************************")

    for network in open_pbz("output.pbz"):
        print("network id is " + str(network.id))
        G, fp = net2basegraph(network)
        G, _, _ = prolong_graph(G, foi, fp)

        for s in network.server:
            print(s.id)

        for f in network.flow:
            print(f.id)
            print(f.path)

        break

    color_map = []
    for node in G:
        if node.startswith("s"):
            color_map.append('blue')
        elif node.startswith("f"):
            color_map.append('green')
        else:
            color_map.append('red')

    nx.draw(G, with_labels=True, node_color=color_map)
    return G


if __name__ == "__main__":
    # main()
    l = {3: [1, 2]}
    print(l[3])
    build_net({1, 2}, {3, 4}, {3: [1, 2], 4: [1]})
