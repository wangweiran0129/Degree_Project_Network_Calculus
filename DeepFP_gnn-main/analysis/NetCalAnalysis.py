# This code is written by Weiran Wang
# For any questions, please contact the author of the code at (weiran.wang@epfl.ch)

from py4j.java_gateway import JavaGateway
from pbzlib import open_pbz

def DelayBoundCalculation(dataset):
    """
    This code will calculate the delay bounds of a topology with a given flow of interest in a dataset
    
    In order to run this code, please first start up the NetCal4Python.java in the DNC directory.
    This will connect to the JVM.
    This python function will first read the network features, e.g., server rate and latency
    flow rate and burst, the source and destination/sink servers of a flow, and a assigned flow of interst
    from a given network topologies. Then this function will pass these parameters to NetCal4Python.java to
    calculate the delay bound results.

    :param dataset: The dataset which stores the network topologies in a pbz format
    :return: The calculated delay bounds after calculation
    """

    # Connect to the JVM
    gateway = JavaGateway()
    double_class = gateway.jvm.double
    int_class = gateway.jvm.int

    # Read the network features
    for network in open_pbz(dataset):

        server_number = len(network.server)
        flow_number = len(network.flow)

        # Define the parameters which will pass to the NetCal4Python.java
        server_rate = gateway.new_array(double_class, server_number)
        server_latency = gateway.new_array(double_class, server_number)
        flow_rate = gateway.new_array(double_class, flow_number)
        flow_burst = gateway.new_array(double_class, flow_number)
        flow_src = gateway.new_array(int_class, flow_number)
        flow_dest = gateway.new_array(int_class, flow_number)
        fois = []

        # Grasp the server features
        for s in network.server:
            server_rate[s.id] = s.rate
            server_latency[s.id] = s.latency
        
        for f in network.flow:
            flow_rate[f.id] = f.rate
            flow_burst[f.id] = f.burst
            flow_src[f.id] = f.path[0]
            flow_dest[f.id] = f.path[-1]
            if f.HasField("pmoofp"):
                fois.append(f.id)
        
        foi_id = gateway.new_array(int_class, len(fois))
        for foi_idx, foi in enumerate(fois):
            foi_id[foi_idx] = foi

        # Get the network topology instance and call the DelayBoundCalculation method
        network_topology = gateway.entry_point
        delay_bound = network_topology.DelayBoundCalculation(server_rate, server_latency, flow_rate, flow_burst, flow_src, flow_dest, foi_id)
        for i in delay_bound:
            print("delay bound of the first topology : ", i)
        break


def main():
    dataset = "/Users/wangweiran/Desktop/MasterDegreeProject/Degree_Project_Network_Calculus/dataset-rtas2021/dataserv.ub.tum.de/dataset-train.pbz"
    DelayBoundCalculation(dataset)


if __name__ == "__main__":
    main()

