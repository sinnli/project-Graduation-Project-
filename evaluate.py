
# Evaluate script

import numpy as np
import matplotlib.pyplot as plt
import adhoc_wireless_net
import agent
from system_parameters import *
import argparse
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
# 'DDQN_Q_Novel',
METHODS = ['Max Reward']#, 'Closest to Destination', 'Best Direction', "Least Interfered", 'Strongest Neighbor', 'Largest Data Rate', 'Destination Directly']
N_ROUNDS = 2
METHOD_PLOT_COLORS = cm.rainbow(np.linspace(1,0,len(METHODS)))
# select Plot type
# PLOT_TYPE = "SumRate" "Rate" "Reach" "Power"
PLOT_TYPE = "Power"

def method_caller(agent, method, visualize_axis=None):
    if method == 'DDQN_Q_Novel':
        agent.route_DDRQN(visualize_axis)
    elif method == 'Max Reward':
        agent.route_neighbor_with_largest_reward()
    elif method == 'DDQN Lowest Interference':
        agent.route_DDRQN_with_lowest_interference_band()
    elif method == 'Strongest Neighbor':
        agent.route_strongest_neighbor()
    elif method == 'Closest to Destination':
        agent.route_close_neighbor_closest_to_destination()
    elif method == 'Least Interfered':
        agent.route_close_neighbor_under_lowest_power()
    elif method == 'Largest Data Rate':
        agent.route_close_neighbor_with_largest_forward_rate()
    elif method == 'Best Direction':
        agent.route_close_neighbor_best_forwarding_direction()
    elif method == 'Destination Directly':
        agent.route_destination_directly()
    else:
        print("Shouldn't be here!")
        exit(1)
    return

# Perform a number of rounds of sequential routing
def sequential_routing(agents, method):
    # 1st round routing, just with normal order
    for agent in agents:
        assert len(agent.flow.get_links()) == 0, "Sequential routing should operate on fresh starts!"
        while not agent.flow.first_packet():
            method_caller(agent, method)
    for agent in agents:
        while not agent.flow.destination_reached():
            method_caller(agent, method)
    # compute bottleneck SINR to determine the routing for the sequential rounds
    for i in range(N_ROUNDS-1):
        bottleneck_rates = []
        for agent in agents:
            agent.process_links(memory=None)
            bottleneck_rates.append(agent.flow.bottleneck_rate)
        ordering = np.argsort(bottleneck_rates)[::-1]
        for agent_id in ordering:  # new round routing
            agent = agents[agent_id]
            agent.reset()
            while not agent.flow.first_packet():
                method_caller(agent, method)
        for agent_id in ordering:
            agent = agents[agent_id]
            while not agent.flow.destination_reached():
                method_caller(agent, method)
    for agent in agents:
        agent.process_links(memory=None)
    return

def evaluate_routing(adhocnet, agents, method, n_layouts):
    assert adhocnet.n_flows == len(agents)
    results = []
    for i in range(n_layouts):
        adhocnet.update_layout()
        sequential_routing(agents, method)
        for agent in agents:
            results.append([agent.flow.bottleneck_rate, len(agent.flow.get_links()),
                            agent.flow.get_number_of_reprobes(), agent.flow.number_reached_packets(),
                            agent.flow.tot_power, agent.adhocnet.used_bands]) #agent.capacity
        for agent in agents:
            agent.reset()
    results = np.array(results); assert np.shape(results)==(n_layouts*adhocnet.n_flows, 6)
    results = np.reshape(results, (n_layouts, adhocnet.n_flows, 6))
    return results

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--visualize', help='option to visualize the routing results by all methods', default=False)
    parser.add_argument('--step', help='option to visualize step selection and scores', default=False)
    args = parser.parse_args()
    adhocnet = adhoc_wireless_net.AdHoc_Wireless_Net()
    agents = [agent.Agent(adhocnet, i) for i in range(adhocnet.n_flows)]

    N_LAYOUTS_TEST = 3
    if args.visualize:
        N_LAYOUTS_TEST = 3

    if (not args.visualize) and (not args.step):
        all_results = dict()
       # for method in METHODS:
        #    print("Evaluating {}...".format(method))
        method = 'Max Reward'
        all_results[method] = evaluate_routing(adhocnet, agents, method, N_LAYOUTS_TEST)
        # plot Sum-Rate and Min-Rate CDF curve
        if PLOT_TYPE == "SumRate":
            xlabel_format = "SumRate"
        elif PLOT_TYPE == "Rate":
            xlabel_format = "Rate"
        elif PLOT_TYPE == "Reach":
            xlabel_format = "Reach In time Packets"
        elif PLOT_TYPE == "Power":
            xlabel_format = "Power"
        else:
            print(f"Invalid plot type {PLOT_TYPE}!")
            exit(1)
        plt.xlabel(xlabel_format)
        plt.ylabel("Cumulative Distribution over Test Adhoc Networks")
        plt.grid(linestyle="dotted")
        plot_upperbound = 0
        for i, (method, results) in enumerate(all_results.items()):
            rates, n_links, n_reprobes, packets_reach, total_power, used_bands = results[:, :, 0], results[:, :, 1], \
                                                                                 results[:, :, 2], results[:, :, 3], \
                                                                                 results[:, :, 4], results[:, :, 5]
            reached_packets = np.mean(packets_reach, axis=1) / num_packets * 100  # percentage of reach packets.
            sumrates, minrates = np.sum(rates, axis=1), np.min(rates, axis=1)
            print(
                "[{}] Avg SumRate: {:.3g}Mbps; Avg MinRate: {:.3g}Mbps; Avg Rate: {:.3g}Mbps; Avg # links per flow: {:.1f}; Avg # reprobes per flow: {:.2g}".format(
                    method, np.mean(sumrates) / 1e6, np.mean(minrates) / 1e6, np.mean(rates) / 1e6, np.mean(n_links),
                    np.mean(n_reprobes)))
            if PLOT_TYPE == "SumRate":
                plt.plot(np.sort(sumrates) / 1e6,
                         np.arange(1, N_LAYOUTS_TEST + 1) / (N_LAYOUTS_TEST),
                         c=METHOD_PLOT_COLORS[i], label=method)
                plot_upperbound = max(np.max(sumrates) / 1e6, plot_upperbound)
            elif PLOT_TYPE == "Rate":
                plt.plot(np.sort(rates.flatten()) / 1e6,
                         np.arange(1, N_LAYOUTS_TEST * adhocnet.n_flows + 1) / (N_LAYOUTS_TEST * adhocnet.n_flows),
                         c=METHOD_PLOT_COLORS[i], label=method)
                plot_upperbound = max(np.max(rates) / 1e6, plot_upperbound)
            elif PLOT_TYPE == "Reach":
                plt.plot(np.sort(reached_packets),
                         np.arange(1, N_LAYOUTS_TEST + 1) / (N_LAYOUTS_TEST),
                         c=METHOD_PLOT_COLORS[i], label=method)
                plot_upperbound = max(np.max(reached_packets), plot_upperbound)
            elif PLOT_TYPE == "Power":
                plt.plot(np.sort(total_power.flatten()) / 1e6,
                         np.arange(1, N_LAYOUTS_TEST * adhocnet.n_flows + 1) / (N_LAYOUTS_TEST * adhocnet.n_flows),
                         c=METHOD_PLOT_COLORS[i], label=method)
                plot_upperbound = max(np.max(total_power) / 1e6, plot_upperbound)
            else:
                print(f"Invalid plot type {PLOT_TYPE}!")
                exit(1)
        plt.legend()
        plt.show()
    elif args.visualize:
        METHODS = ["DDQN_Q_Novel", "Closest to Destination", "Largest Data Rate", "Best Direction"]
        for i in range(N_LAYOUTS_TEST):
            adhocnet.update_layout()
            fig, axes = plt.subplots(2, 2)
            axes = axes.flatten()
            gs = gridspec.GridSpec(2, 2)
            gs.update(wspace=0.05, hspace=0.05)
            for (j, method) in enumerate(METHODS):
                ax = axes[j]
                ax.set_title(method)
                ax.tick_params(axis=u'both', which=u'both', length=0)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                sequential_routing(agents, method)
                # start visualization plot
                adhocnet.visualize_layout(ax)
                for agent in agents:
                    agent.visualize_route(ax)
                    agent.reset()
            plt.show()
    elif args.step:
        METHODS = ["DDQN_Q_Novel", "Closest to Destination"]
        for method in METHODS:
            for i, agent in enumerate(agents):
                print("[Sequential Routing 1st round] Agent ", i)
                while not agent.flow.destination_reached():
                    ax = plt.gca()
                    adhocnet.visualize_layout(ax)
                    for agent_finished in agents[:i]:
                        agent_finished.visualize_route(ax)
                    # execute one step and plot
                    method_caller(agent, method, ax)
                    if agent.flow.destination_reached():
                        agent.visualize_route(ax)
                    plt.tight_layout()
                    plt.show()
            for i, agent in enumerate(agents):
                print("[Sequential Routing 2nd round] Agent ", i)
                agent.reset()
                while not agent.flow.destination_reached():
                    ax = plt.gca()
                    adhocnet.visualize_layout(ax)
                    for agent_finished in agents:
                        if agent_finished == agent:
                            continue
                        agent_finished.visualize_route(ax)
                    # execute one step and plot
                    method_caller(agent, method, ax)
                    if agent.flow.destination_reached():
                        agent.visualize_route(ax)
                    plt.tight_layout()
                    plt.show()
            for agent in agents:
                agent.reset()

    print("Evaluation Completed!")
