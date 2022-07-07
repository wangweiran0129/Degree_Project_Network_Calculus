#!/usr/bin/env python3

import pbzlib
import argparse
import numpy as np


def main(filename):
    i=0
    # Open a .pbz file containing the networks
    for network in pbzlib.open_pbz(filename):

        
        #print("**************************************************************\n")
        # Print all the servers of the network
        for server in network.server:
            print(server.id)

        print("********************************")
        for flow in network.flow:
            print(flow.id)
            print(flow.path)

        print("********************************")

        # Print all the flows in the network
        #print("flows start here\n")
        for flow in network.flow:
            print(flow.id)
            print(flow.path)

            min_delay_bound = flow.deborahfp.delay_bound
  
            for explored_comb in flow.deborahfp.explored_combination:
                if(explored_comb.delay_bound == min_delay_bound):

                    print(explored_comb)

            break
        break

           
            

        # Stop the loop to only display the first network
        
        

    


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    args = p.parse_args()
    main(args.input)
