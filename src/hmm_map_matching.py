import pandas as pd
import time
import math
import networkx as nx

from data_process.dis_latlon import distance_meter


class HmmMapMatching(object):
    def __init__(self, segment):
        self.G = None
        self.edges = None
        self.nodes = None
        self.segment = segment

    def get_graph(self, segment):
        road_network = nx.DiGraph()

        for segment_id in segment:
            # print(segment[segment_id])
            road_network.add_edge(segment[segment_id]['node_a_id'], segment[segment_id]['node_b_id'],
                                  length=segment[segment_id]['length'],
                                  label=segment_id)
        self.G = road_network

    def get_node(self, segment):
        node_id = []
        node = []
        for segment_id in segment:
            node_id.append(segment[segment_id]['node_a_id'])
            node.append(segment[segment_id]['node_a'])
            node_id.append(segment[segment_id]['node_b_id'])
            node.append(segment[segment_id]['node_b'])

        node_dict = pd.DataFrame({
            'node_id': node_id,
            'node': node
        })
        # print(node_dict)
        node_dict.drop_duplicates(['node_id', 'node'], keep='first', inplace=True, ignore_index=True)
        # print(node_dict)
        nodes_dict = list()

        for i in range(len(node_dict)):
            nodes_dic = {
                'node_id': node_dict.node_id[i],
                'node': node_dict.node[i]
            }
            nodes_dict.append(nodes_dic)

        self.nodes = {node['node_id']: node for node in nodes_dict}

    def calc_sigma(self, distances_set):
        """
        Calculate sigma for emission probability
        :param distances_set:  distance set of all candidate segments of a point
        :return:
        """
        if len(distances_set) % 2 == 0:
            return 1.4826 * (distances_set[len(distances_set)//2] + distances_set[len(distances_set)//2-1])/2
        return 1.4826 * distances_set[len(distances_set)//2]

    def calc_emission_probabilities_v2(self, roads, sigma=None):
        """
        Calculate the emission probability of trajectory points
        :param sigma:
        :param roads: The set of candidate segments [[[osmid, start_id, end_id, Point(x,y), d], [], ..., []], [], ..., []]
        :param distance: search range distance
        :return: emission probability of candidate segments for points [[,,...,],[],[],...,[]]
        """
        start_time = time.time()
        probabilities = []
        for candidate_roads in roads:
            temp = []
            distances_set = []
            for road in candidate_roads:
                distances_set.append(road[-1])
            # print(distances_set)
            if sigma:
                pass
            else:
                sigma = self.calc_sigma(sorted(distances_set))
            for road in candidate_roads:
                # Calculate emission probability， Gaussian distribution function with an expectation of 0 and a standard deviation of sigma
                e = 1 / (math.sqrt(2 * math.pi) * sigma) * math.exp(-0.5 * ((road[-1] / sigma) ** 2))
                if e < 1e-20:
                    e = 0
                temp.append(e)
            probabilities.append(temp)
        # print("Calculate emission probability time： {:.2f} s".format(time.time() - start_time))
        return probabilities

    def calc_beta(self, distances_set):
        distances_set = [x for x in distances_set if math.isnan(x) is False]
        if distances_set:
            if len(distances_set) % 2 == 0:
                try:
                    beta = 1/math.log(2) * (distances_set[len(distances_set)//2] + distances_set[len(distances_set)//2-1])/2
                    return beta
                except IndexError:
                    print(distances_set)
            return 1/math.log(2) * distances_set[len(distances_set)//2]
        else:
            return 2

    def calc_road_distance_v2(self, road1, road2):
        """
        Calculate the road distance between the projected points on the two candidate segments
        :param road1: one of the candidate segments of the first point，（osmid, start_id, end_id, Point(x,y), d)
        :param road2: one of the candidate segments of the second point，（osmid, start_id, end_id, Point(x,y), d）
        :return: road distance
        """

        #  the closest point x1 of point node1 on its candidate road segment path1
        closet_point1 = road1[3]
        # the closest point x2 of point node2 on its candidate road segment path2
        closet_point2 = road2[3]
        # If two sections are the same, the route distance is the distance between the nearest points
        if road1[1] == road2[1] and road1[2] == road2[2]:
            return distance_meter(closet_point1, closet_point2)
        else:
            # have no path
            try:
                # Calculate the roads passed from the start of edge path1 to the end of edge path2
                a = nx.shortest_path(self.G, road1[2], road2[1], weight='length')
                # print(road1)
                # print(road2)
                # print(a)

                a_dis = 0
                for i in range(0, len(a)-1):
                    a_dis += self.G[a[i]][a[i+1]]['length']

            except nx.exception.NetworkXNoPath:
                # have no path
                return float('nan')

            route = distance_meter(self.nodes[a[0]]['node'], closet_point1)

            route += a_dis
            # route += nx.shortest_path_length(self.G, a[0], a[-1], weight='length')

            route += distance_meter(self.nodes[a[-1]]['node'], closet_point2)
        return route

    def calc_transition_probabilities_v2(self, roads, trajectory, beta=None):
        start_time = time.time()
        probabilities = []
        # With n nodes, there are only n-1 transition probability matrices,
        # because each transition probability matrix represents one locus point to the next locus point.
        # start_time_2 = time.time()
        for i in range(0, len(trajectory) - 1):
            #  previous point point1
            point1 = trajectory[i]
            # candidate segments of point1
            candidate_road_of_point1 = roads[i]
            # last point point2
            point2 = trajectory[i + 1]
            # candidate segments of  point2
            candidate_road_of_point2 = roads[i + 1]
            dts = []
            # for candidate segments of point2
            for road2 in candidate_road_of_point2:
                # Calculate the transition probability from road1 to road2 for all candidate segments of point1
                temp = []
                for road1 in candidate_road_of_point1:
                    route_distance = self.calc_road_distance_v2(road1, road2)
                    dt = abs(distance_meter(point1, point2) - route_distance)

                    # new_dt = dt if dt < 300 else float('inf')

                    temp.append(dt)
                # print(temp)
                dts.append(temp)
            probabilities.append(dts)

        # print("{} a {:.2f}s".format(len(trajectory), time.time() - start_time_2))
        # beta
        betas = []

        if beta:
            for i in range(0, len(trajectory) - 1):
                for j in range(0, len(probabilities[i])):
                    probabilities[i][j] = [1 / beta * math.exp(-dt / beta) for dt in probabilities[i][j]]
        else:
            for distances_set in probabilities:
                temp = []
                for d in distances_set:
                    temp = temp + d
                betas.append(self.calc_beta(sorted(temp)))

            for i in range(0, len(trajectory) - 1):
                beta = betas[i]
                if beta == 0:
                    beta = 1e-10
                    for j in range(0, len(probabilities[i])):
                        probabilities[i][j] = [1/beta*math.exp(-dt/beta) for dt in probabilities[i][j]]
                else:
                    for j in range(0, len(probabilities[i])):
                        probabilities[i][j] = [1/beta*math.exp(-dt/beta) for dt in probabilities[i][j]]
                # print("{} {:.2f} s {}".format(i, time.time() - start_time_1, betas[i]))
            # print("Calculate transition probability time： {:.2f} s".format(time.time() - start_time))
        return probabilities

    def viterbi(self, all_roads, e_probabilities, t_probabilities):
        start_time = time.time()
        initial_probabilities = e_probabilities[0]
        best = []  # [[{'index':index, 'probability':p}, {}, ..., {}], [], ..., []]
        temp = []  # The maximum probability of each candidate segment of a point from each candidate segment of a previous point to this segments
        # initial，calculate emission probability
        for i in range(0, len(initial_probabilities)):
            temp.append({"index": i, "probability": initial_probabilities[i]})
        # print('temp', temp)
        best.append(temp)
        # print('best', best)

        for i in range(1, len(all_roads)):
            temp = []
            # for all candidate segments
            for j in range(0, len(e_probabilities[i])):
                # t is the transition probability from all candidate sections of the previous point to a candidate section of the current point
                t = t_probabilities[i - 1][j]
                # The initial maximum probability is the first candidate segment
                try:
                    max_p = t[0] * e_probabilities[i][j] * best[i - 1][0]['probability']
                    index = 0
                except IndexError:
                    print(i, j)
                    print(e_probabilities[i-2])
                    print(e_probabilities[i-1])
                    print(e_probabilities[i])
                    print(best[i-1])
                    return
                s = {"index": index, "probability": max_p}
                # Calculate the maximum probability from all candidate segments of the previous point to a candidate segments of the current point
                # Record the index of the candidate segments of the previous point
                for k in range(1, len(t)):
                    p = t[k] * e_probabilities[i][j] * best[i - 1][k]['probability']
                    if p > max_p:
                        max_p = p
                        index = k
                        s["index"] = index
                        s["probability"] = max_p
                temp.append(s)
            best.append(temp)
        # backtracking to find the path with the highest probability
        max_p = best[-1][0]['probability']
        index = 0
        best_path = []
        for i in range(1, len(best[-1])):
            if best[-1][i]['probability'] > max_p:
                max_p = best[-1][i]['probability']
                index = i
        best_path.append(index)
        for i in range(len(best) - 1, 0, -1):
            index = best[i][index]['index']
            best_path.append(index)
        best_path.reverse()
        # print("viterbi running time： {:.2f} s".format(time.time() - start_time))
        result = []
        for i in range(0, len(all_roads)):
            result.append(all_roads[i][best_path[i]])

        return result
