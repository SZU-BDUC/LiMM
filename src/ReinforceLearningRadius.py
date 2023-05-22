import numpy as np
import pandas as pd
import math
from math import radians
import time
import os
import random
import pickle
from tqdm import tqdm

from data_process.hexagon import get_all_cell, get_key, label_point
from data_process.dis_latlon import distance_meter, get_bear, destination, get_intersection_point, \
    distance_point_to_segment
from learned_index import UseLearnedIndex
from R_Tree import HMMRTree

# agent's state
actions = ['reduce', 'enlarge', 'keep']


# Update environment
def update_env(state, states):

    env = list('-' * len(states))
    if state != states[-1]:
        env[math.ceil(state / 10)] = 'o'


# Calculate the next state
def get_next_state(state, action, states):

    if action == 'enlarge' and state != states[-1]:
        next_state = state + 10
    elif action == 'reduce' and state != states[0]:
        next_state = state - 10
    else:
        next_state = state # 保持当前状态
    return next_state


# takes the actions performed by the explorer and returns a list of all legitimate actions, independent of reward
def get_valid_actions(state, states):
    global actions

    valid_actions = set(actions)
    if state == states[-1]:
        valid_actions -= set(['enlarge'])
    if state == states[0]:
        valid_actions -= set(['reduce'])
    return list(valid_actions)


class UseReinforceLearnedRadius(object):
    def __init__(self, config):
        self.config = config
        self.side_length = 500

        # config
        self.epsilon = 0.9  # greedy rate
        self.alpha = 0.1  # learning rate
        self.gamma = 0.8  # diminishing reward value

        self.big_hexagon_path = np.load(self.config.data_dir + '/hexagon/big_hexagon.npy', allow_pickle=True)
        self.big_hexagon = self.big_hexagon_path.item()
        self.get_small_hexagon_path = np.load(self.config.data_dir + '/hexagon/get_small_hexagon.npy', allow_pickle=True)
        self.get_small_hexagon = self.get_small_hexagon_path.item()
        self.count_fractal_hexagon_path = np.load(self.config.data_dir + '/hexagon/count_fractal_big_hexagon.npy', allow_pickle=True)
        self.count_fractal_hexagon = []
        for item in self.count_fractal_hexagon_path:
            self.count_fractal_hexagon.append(item)

        self.q_table_path = self.config.result_dir + '/hexagon_q_table'
        if not os.path.exists(self.q_table_path):
            os.makedirs(self.q_table_path)

        self.gps_sample_path = self.config.result_dir + '/gps_sample'
        if not os.path.exists(self.gps_sample_path):
            os.makedirs(self.gps_sample_path)

        self.rein_tra_path = self.config.data_dir + '/rein_tra'
        if not os.path.exists(self.rein_tra_path):
            print('need to check', self.config.data_dir + '/rein_tra')

        self.rein_ground_truth = self.config.result_dir + '/rein_ground_truth'
        if not os.path.exists(self.rein_ground_truth):
            # need to run r-tree index based hmm

            print('\nrunning ground truth for reinforce learning searching radius\n')
            ground_trutb_mm = HMMRTree(self.config)
            ground_trutb_mm.run(100)
            print('ground truth matching finish\n')

            # print('need to check', self.config.result_dir + '/rein_ground_truth')

        self.construct()

    def construct(self):
        learned_index = UseLearnedIndex(self.config)

        # q_table
        record_label = []
        for root, dirs, files in os.walk(self.q_table_path):
            for file in files:
                if os.path.splitext(file)[-1] == '.csv':
                    locals()[os.path.splitext(file)[0]] = pd.read_csv(self.q_table_path + '/' + file,
                                                                      float_precision='round_trip', index_col=0)
                    record_label.append(os.path.splitext(file)[0])
        # print(record_label)

        # gps_sample
        gps_label = []
        gps_label1 = []
        for root, dirs, files in os.walk(self.gps_sample_path):
            for file in files:
                if os.path.splitext(file)[-1] == '.csv':
                    locals()[os.path.splitext(file)[0]] = pd.read_csv(self.gps_sample_path + '/' + file,
                                                                      float_precision='round_trip', index_col=0)
                    gps_label1.append(os.path.splitext(file)[0])
        # print(gps_label1)

        # num = 0
        # if not gps_label: # test
        print('creating gps label')
        for root, dirs, files in os.walk(self.rein_ground_truth):
            # print(files)
            for file in tqdm(files):
                if os.path.splitext(file)[-1] == '.npy':
                    # print(file)
                    matched_point_own1 = np.load(self.rein_ground_truth + '/' + file, allow_pickle=True)
                    matched_point_own = matched_point_own1.item()
                    for matched_id in matched_point_own:
                        # if search_matched_id == matched_id:
                        all_result = matched_point_own[matched_id]['match_detail']
                        # print(all_result[0])
                        track_file = pd.read_csv(self.rein_tra_path + '/' + matched_id, float_precision='round_trip')
                        track = []
                        for i in range(track_file.shape[0]):
                            a = float(track_file.lat[i])
                            b = float(track_file.lon[i])
                            track.append((a, b))

                        for track_node in track:
                            # num += 1
                            cell_id = label_point(track_node, self.side_length, self.big_hexagon)
                            cell_center_point = self.big_hexagon[cell_id]['center_point']
                            big_degree = get_bear(cell_center_point, track_node)
                            big_meter = round(distance_meter(cell_center_point, track_node), 0)

                            if cell_id in self.count_fractal_hexagon:
                                small_label = math.ceil(
                                    (((big_degree + 60 - math.degrees(
                                        math.asin((1 / 2) / math.sqrt(7)))) + 360) % 360) / 60)
                                small_point = self.get_small_hexagon[round(cell_id + 0.1 * small_label, 1)][
                                    'center_point']
                                to_small_meter = distance_meter(small_point, track_node)

                                if big_meter <= to_small_meter:
                                    label = cell_id

                                else:
                                    label = round(cell_id + 0.1 * small_label, 1)
                            else:
                                label = cell_id

                            # Create gps log table
                            if str(label) + '_gps_sample' in gps_label:
                                # q_table = str(label) + _q_tabel
                                gps = locals()[str(label) + '_gps_sample']
                            else:
                                # print(str(label) + '_gps_sample.csv', False)

                                locals()[str(label) + '_gps_sample'] = pd.DataFrame(
                                    columns=['raw_gps_lat', 'raw_gps_lon', 'match_id', 'match_start_id',
                                             'match_end_id'])
                                gps = locals()[str(label) + '_gps_sample']
                                gps_label.append(str(label) + '_gps_sample')

                            gps1 = gps.append({
                                'raw_gps_lat': track_node[0],
                                'raw_gps_lon': track_node[1],
                                'match_id': all_result[track.index(track_node)][0],
                                'match_start_id': all_result[track.index(track_node)][1],
                                'match_end_id': all_result[track.index(track_node)][2]}, ignore_index=True)

                            locals()[str(label) + '_gps_sample'] = gps1
        # print('gps_label', gps_label)

        print('construct q_table')
        for gps_sample_id in tqdm(gps_label):
            if gps_sample_id not in gps_label1:
                gps_sample = locals()[gps_sample_id]
                gps_sample.to_csv(self.gps_sample_path + '/' + gps_sample_id + '.csv')

                repeat_label = float(gps_sample_id.split('_')[0])

                # Create Q table
                if str(repeat_label) + '_q_table' in record_label:
                    # q_table = str(label) + _q_tabel
                    q_table = locals()[str(repeat_label) + '_q_table']
                else:
                    # print(str(repeat_label) + '_q_table.csv', False)

                    # whether it is a small hexagon
                    if 0 < repeat_label % 1 < 1 or repeat_label in self.count_fractal_hexagon:
                        states = list(range(0, 101, 10))
                    else:
                        states = list(range(0, 151, 10))

                    locals()[str(repeat_label) + '_q_table'] = pd.DataFrame(
                        data=[[0 for _ in actions] for _ in states], index=range(len(states)),
                        columns=actions)
                    q_table = locals()[str(repeat_label) + '_q_table']
                    record_label.append(str(repeat_label) + '_q_table')

                states = list(range(0, (len(q_table)-1) * 10 + 1, 10))
                # print('state', states)
                for sample_id in range(len(gps_sample)):
                    # every1, 0m is 0

                    result_search = [1] * len(states)
                    result_search[0] = 0

                    # Advance calculate reward
                    for sta in states[1:]:
                        o_c_sub, _ = learned_index.search(
                            (gps_sample.loc[sample_id, 'raw_gps_lat'], gps_sample.loc[sample_id, 'raw_gps_lon']), sta)  # Searched candidate sets (including subsets)

                        o_c_ids = [can[0] for can in o_c_sub]  # Candidate sets (no subsets)
                        o_c = list(np.unique(o_c_ids))

                        if str(gps_sample.loc[sample_id, 'match_id']) not in o_c:
                            result_search[math.ceil(sta / 10)] = 0
                        else:
                            # break
                            result_search[math.ceil(sta / 10)] = 1 / len(o_c)
                            # change reward
                        # print('_____________________________')
                        # print('o_c', o_c, type(o_c[0]), gps_sample.loc[i, 'match_id'], type(gps_sample.loc[i, 'match_id']),
                        #       'result', result_search[math.ceil(sta / 10)])

                    # Iterate 10 times
                    for i in range(10):
                        # current_state = states[math.ceil(len(states) / 2)]
                        # Change current state
                        current_state = states[-1]
                        total_steps = 0
                        # reward = 0
                        reward = result_search[-1]
                        # a = 1
                        while reward != 0:
                            current_index = math.ceil(current_state / 10)
                            if (random.uniform(0, 1) > self.epsilon) or ((q_table.loc[current_index] == 0).all()):

                                # if reward != 0:
                                #     current_action = 'reduce'
                                # else:
                                #     current_action = 'enlarge'
                                if i == 0:
                                    current_action = 'reduce'
                                else:
                                    current_action = np.random.choice(actions)  # 随机选取一个行动
                            else:
                                current_action = q_table.loc[current_index].idxmax()

                            next_state = get_next_state(current_state, current_action, states)
                            next_index = math.ceil(next_state / 10)
                            next_state_q_values = q_table.loc[next_index, get_valid_actions(next_state, states)]

                            # a = result_search[current_index]

                            # b = result_search[next_index]

                            # reward = abs(b - a)
                            reward = result_search[current_index]

                            q_table.loc[current_index, current_action] += self.alpha * (
                                    reward + self.gamma * next_state_q_values.max() -
                                    q_table.loc[current_index, current_action])

                            current_state = next_state
                            total_steps += 1

                            # if b == 0 and next_state == states[-1]:
                            if next_state == states[0] or result_search[-1] == 0:
                                break

                    locals()[str(repeat_label) + '_q_table'] = q_table
                # print(str(repeat_label), 'finish')
                locals()[str(repeat_label) + '_q_table'].to_csv(self.q_table_path + '/' + str(repeat_label) + '_q_table' + '.csv')

        # for re in record_label:
        #     q = locals()[re]
        #     q.to_csv(self.q_table_path + '/' + re + '.csv')
