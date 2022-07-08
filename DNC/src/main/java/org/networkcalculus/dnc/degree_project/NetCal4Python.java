// This code is written by Weiran Wang
// For any questions, please contact the author of the code at
// (weiran.wang@epfl.ch)

package org.networkcalculus.dnc.degree_project;

import py4j.GatewayServer;
import org.networkcalculus.dnc.AnalysisConfig;
import org.networkcalculus.dnc.curves.ArrivalCurve;
import org.networkcalculus.dnc.curves.Curve;
import org.networkcalculus.dnc.curves.ServiceCurve;
import org.networkcalculus.dnc.network.server_graph.Flow;
import org.networkcalculus.dnc.network.server_graph.Server;
import org.networkcalculus.dnc.network.server_graph.ServerGraph;
import org.networkcalculus.dnc.tandem.analyses.FIFOTandemAnalysis;
import org.networkcalculus.dnc.tandem.analyses.PmooAnalysis;
import org.networkcalculus.num.Num;

public class NetCal4Python {

    public NetCal4Python() {

    }

    public double[] delayBoundCalculation(double[] server_rate, double[] server_latency, double[] flow_rate,
            double[] flow_burst, int[] flow_src, int[] flow_dest, int[] foi_id) {

        System.out.println("hello ?");

        /*
         * This method will calculate the delay bounds based on the network features and
         * the flow(s) of interest
         * 
         * @param server_rate: the rate of each server in this topology
         * 
         * @param server_latency: the latency of each server in this topology
         * 
         * @param flow_rate: the rate of each flow in this topology
         * 
         * @param flow_burst: the burst of each flow in this topology
         * 
         * @param flow_src: the source server of each flow
         * 
         * @param flow_dest: the destination/sink server of each flow
         * 
         * @param foi_id: the flow(s) of interest in this topology
         * 
         * @return: the delay bound based on the network features and the designated
         * flow of interest
         */

        int server_number = server_rate.length;
        int flow_number = flow_rate.length;
        System.out.println("# servers : " + server_number);
        System.out.println("# flows : " + flow_number);
        System.out.println();

        ServerGraph sg = new ServerGraph();
        AnalysisConfig configuration = new AnalysisConfig();
        configuration.enforceMaxSC(AnalysisConfig.MaxScEnforcement.GLOBALLY_ON);
        configuration.enforceMaxScOutputRate(AnalysisConfig.MaxScEnforcement.GLOBALLY_ON);

        // Define the service curve and add the service curves to the servers
        ServiceCurve[] service_curve = new ServiceCurve[server_number];
        Server[] servers = new Server[server_number];
        for (int i = 0; i < server_number; i++) {
            service_curve[i] = Curve.getFactory().createRateLatency(server_rate[i], server_latency[i]);
            servers[i] = sg.addServer(service_curve[i]);
        }

        // Figure out the server order
        String server_order = "";
        int order_increase_counter = 0;
        int order_decrease_counter = 0;
        for (int i = 0; i < flow_number; i++) {
            if (flow_src[i] <= flow_dest[i]) {
                order_increase_counter++;
            }
            if (flow_src[i] >= flow_dest[i]) {
                order_decrease_counter++;
            }
        }

        if (order_increase_counter == flow_number) {
            server_order = "increasing";
        }
        if (order_decrease_counter == flow_number) {
            server_order = "decreasing";
        }
        if (order_increase_counter != flow_number && order_decrease_counter != flow_number) {
            System.out.println("Chaotic Server Connections. Delay Bounds Cannot Be Calculated");
            System.exit(0);
        }

        System.out.println("The order of this topology : " + server_order);
        System.out.println();

        // Connect the servers
        try {
            if (server_order.equals("increasing")) {
                for (int i = 0; i < server_number - 1; i++) {
                    sg.addTurn(servers[i], servers[i + 1]);
                }
            }
            if (server_order.equals("decreasing")) {
                for (int i = server_number - 1; i > 0; i--) {
                    sg.addTurn(servers[i], servers[i - 1]);
                }
            }
        } catch (Exception e) {
            System.out.println("Server Connections Error");
            e.printStackTrace();
        }

        // Define the arrival curves
        ArrivalCurve[] arrival_curve = new ArrivalCurve[flow_number];
        try {
            for (int i = 0; i < flow_number; i++) {
                arrival_curve[i] = Curve.getFactory().createTokenBucket(flow_rate[i], flow_burst[i]);
            }
        } catch (Exception e) {
            System.out.println("Arrival Curve Adding Error");
            e.printStackTrace();
        }

        // Add the arrival curves to the flows
        try {
            for (int i = 0; i < flow_number; i++) {
                int src = flow_src[i];
                int dest = flow_dest[i];
                if (src == dest) {
                    sg.addFlow(arrival_curve[i], servers[src]);
                } else {
                    sg.addFlow(arrival_curve[i], servers[src], servers[dest]);
                }
            }
        } catch (Exception e) {
            System.out.println("Flow Adding Error");
            e.printStackTrace();
        }

        double[] pmoo_delay_bound = new double[foi_id.length];

        try {
            // Define the flow of interest
            for (int i = 0; i < foi_id.length; i++) {
                Flow flow_of_interest = sg.getFlow(foi_id[i]);
                System.out.println("flow of interest : " + flow_of_interest);
                System.out.println();

                // Analyze the network
                System.out.println("--- PMOO Analysis ---");
                System.out.println();
                PmooAnalysis pmoo = new PmooAnalysis(sg, configuration);
                pmoo.performAnalysis(flow_of_interest);
                Num pmoo_db = pmoo.getDelayBound();
                pmoo_delay_bound[i] = pmoo_db.doubleValue();
                System.out.println("delay bound Num : " + pmoo_db);
                System.out.println("delay bound double : " + pmoo_delay_bound[i]);
            }

        } catch (Exception e) {
            System.out.println("PMOO Analysis Failed");
            e.printStackTrace();
        }

        return pmoo_delay_bound;

    }

    public static void main(String[] args) {
        NetCal4Python network = new NetCal4Python();
        // network is now the gateway.entry_point
        GatewayServer server = new GatewayServer(network);
        server.start();
    }
}