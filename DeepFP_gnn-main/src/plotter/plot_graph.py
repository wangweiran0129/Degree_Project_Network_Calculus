import networkx as nx


def plot_graph_network(G):
    """
    method to plot a graph of the network
    :param G: the graph
    """
    color_map = []
    for node in G:
        if node.startswith("s"):
            color_map.append('blue')
        elif node.startswith("f"):
            color_map.append('green')
        else:
            color_map.append('red')

    nx.draw(G, with_labels=True, node_color=color_map)
