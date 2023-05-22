from rtree import index as r_index
from data_process.dis_latlon import box_around_point
from data_process.dis_latlon import distance_point_to_segment
import numpy as np
import pandas as pd
import time
import psutil
import os
import pickle
from tqdm import tqdm

from hmm_map_matching import HmmMapMatching

class RTree(object):
    def __init__(self, all_segment, original_segment):
        self.all_segment = all_segment
        self.original_segment = original_segment
        p = r_index.Property()
        self.index = r_index.Index(properties=p)

        self.construct()

    def construct(self):
        """
        :param segment: segment
        :return: r-tree index segment npy
        The npy of the new segment can be matched with the order of the learning index
        """
        num = 0
        segment_changes = list()
        for segment_id in self.all_segment:
            segment_change = {
                'num': num,
                'link_id': self.all_segment[segment_id]['link_id'],
                'osm_way_id': self.all_segment[segment_id]['osm_way_id'],
                'name': self.all_segment[segment_id]['name'],
                'length': self.all_segment[segment_id]['length'],
                'node_a_id': self.all_segment[segment_id]['node_a_id'],
                'node_a': self.all_segment[segment_id]['node_a'],
                'node_b_id': self.all_segment[segment_id]['node_b_id'],
                'node_b': self.all_segment[segment_id]['node_b']
            }
            segment_changes.append(segment_change)
            num = num + 1

        self.segment_dict = {segment['num']: segment for segment in segment_changes}

        for segment_id in self.segment_dict:
            s_lon = min(self.segment_dict[segment_id]['node_a'][1], self.segment_dict[segment_id]['node_b'][1])
            s_lat = min(self.segment_dict[segment_id]['node_a'][0], self.segment_dict[segment_id]['node_b'][0])
            e_lon = max(self.segment_dict[segment_id]['node_a'][1], self.segment_dict[segment_id]['node_b'][1])
            e_lat = max(self.segment_dict[segment_id]['node_a'][0], self.segment_dict[segment_id]['node_b'][0])
            segment = (s_lon, s_lat, e_lon, e_lat)  # x-lon, y-lat
            sid = segment_id
            self.index.insert(sid, segment, obj=1)

    def search(self, point, radius):
        start_time = time.time()
        candidate_segment = []
        q = box_around_point(point, radius)
        q = (q[1], q[0], q[3], q[2])  # yx coordinate

        hits = self.index.intersection(q, objects=True)
        link_ids = []
        for item in hits:
            # thick-grained data
            split_id = self.segment_dict[item.id]['link_id'].split('_')
            link_id = split_id[0]

            # Fine-grained data
            # link_id = rtree_segment[item.id]['link_id']

            link_ids.append(link_id)
        link_ids = list(np.unique(link_ids))

        for link in link_ids:
            link_start = self.original_segment[link]['node_a_id']
            link_end = self.original_segment[link]['node_b_id']
            d, p, _ = distance_point_to_segment(point, self.original_segment[link]['node_a'],
                                                self.original_segment[link]['node_b'])
            # if d <= radius:
            candidate_segment.append([link, link_start, link_end, p, d])

        candidate_segment.sort(key=lambda x: x[-1])
        candidate_time = round(time.time() - start_time, 9)
        # print('num+can', len(candidate_segment))
        return candidate_segment, candidate_time

class HMMRTree(object):
    def __init__(self, config):
        self.config = config

    # # using r-tree for HMM
    def run(self, radius):
        # calculate IO
        p = psutil.Process()
        io_counters = p.io_counters()
        disk_usage_process = io_counters[2] + io_counters[3]  # read_bytes + write_bytes

        # Get Start time and first measure
        time_start = time.time()
        disk_io_counter = psutil.disk_io_counters()
        # Get start Read bytes and start read time
        start_read_bytes = disk_io_counter[2]
        # Get start Write bytes and start write time
        start_write_bytes = disk_io_counter[3]

        segment_all_dict = np.load(self.config.data_dir + '/segment_all_dict.npy', allow_pickle=True)
        segment_all_dict = segment_all_dict.item()

        original_segment = np.load(self.config.data_dir + '/original_segment.npy', allow_pickle=True)
        original_segment = original_segment.item()

        r_tree = RTree(segment_all_dict, original_segment)
        # r_index, rtree_segment = R_Tree.get_r_tree(segment_all_dict)

        # Wait before next measure
        # time.sleep(2)
        disk_io_counter = psutil.disk_io_counters()

        # Get end Read bytes and end read time
        end_read_bytes = disk_io_counter[2]
        # Get end Write bytes and end write time
        end_write_bytes = disk_io_counter[3]
        # Get end time
        end_time = time.time()

        # Compute time diff
        time_diff = end_time - time_start
        # Compute Read speed :  Read Byte / second
        read_speed = (end_read_bytes - start_read_bytes) / time_diff

        # Compute Write speed :  Read Byte / second
        write_speed = (end_write_bytes - start_write_bytes) / time_diff

        # Convert to Mb/s
        read_mega_bytes_sec = round(read_speed / (1024 ** 2), 2)
        print('read_mega_bytes_sec', read_mega_bytes_sec)
        write_mega_bytes_sec = round(write_speed / (1024 ** 2), 2)
        print('write_mega_bytes_sec', write_mega_bytes_sec)

        disk_io_counter = psutil.disk_io_counters()
        disk_total = disk_io_counter[2] + disk_io_counter[3]  # read_bytes + write_bytes

        a = disk_usage_process / disk_total * 100

        print('disk_usage_process / disk_total', a)

        print("## Build R-Tree Index costs {:.4f} s".format(time_diff))

        total = r_tree.index.__repr__() #+ len(segment_all_dict)*7
        # total = len(locals().values()) + len(globals().values()) + len(segment_all_dict)*7
        # print(total)

        print("## R-Tree Index Size", total)

        # store index
        r_tree_save_dis = open(self.config.data_dir + '/r_tree.pkl', 'wb')
        r_tree_str = pickle.dumps(r_tree)
        r_tree_save_dis.write(r_tree_str)
        r_tree_save_dis.close()

        output_dir = self.config.result_dir + '/rein_ground_truth'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        trajectory_list = list()
        error_tra_list = []
        num = 0
        # flag = False
        mm_text = HmmMapMatching(original_segment)
        mm_text.get_graph(original_segment)
        mm_text.get_node(original_segment)

        rein_tra_path = self.config.data_dir + '/rein_tra'
        for root, dirs, files in os.walk(rein_tra_path):
            for file in tqdm(files):
                if os.path.splitext(file)[-1] == '.csv':
                    num += 1
                    # if num > 5000:
                    #     flag = True
                    # if flag:
                    # print(num, file)
                    track_file = pd.read_csv(rein_tra_path + '/' + file, float_precision='round_trip')
                    track = []
                    for i in range(track_file.shape[0]):
                        a = float(track_file.lat[i])
                        b = float(track_file.lon[i])
                        track.append((a, b))

                    all_candidate = []
                    all_candidate_time = []

                    start = time.time()
                    for track_node in track:
                        add_num = 0
                        candidate_times = 0
                        while True:
                            new_radius = radius + 5 * add_num
                            add_num = add_num + 1

                            get_candidate_segments, candidate_time = r_tree.search(track_node, new_radius)
                            candidate_times += candidate_time

                            if len(get_candidate_segments) != 0:
                                break

                        all_candidate_time.append(candidate_times)
                        all_candidate.append(get_candidate_segments)

                    e = mm_text.calc_emission_probabilities_v2(all_candidate)

                    t = mm_text.calc_transition_probabilities_v2(all_candidate, track)
                    if t is 0:
                        error_tra_list.append(file)
                        continue

                    result = mm_text.viterbi(all_candidate, e, t)
                    end = time.time()
                    # print('result', result)
                    # print('running time', round(end - start, 2), 's')
                    # print('-' * 50)
                    trajectory_match = {
                        'id': file,
                        'match_time': round(end - start, 2),
                        'match_detail': result,
                        'candidate_time': all_candidate_time,
                        'candidate_detail': all_candidate
                    }

                    trajectory_list.append(trajectory_match)
                    # print(trajectory_match)
                    # print('-' * 50)

                    if num % 1000 == 0:
                        if trajectory_list:
                            trajectory = {a['id']: a for a in trajectory_list}
                            np.save(output_dir + '/trajectory_match_%d.npy' % num, trajectory)
                            trajectory_list = list()

                    if len(error_tra_list) != 0 and len(error_tra_list) % 100 == 0:
                        np.save(output_dir + '/error_trajectory_%d.npy' % num, error_tra_list)
                        error_tra_list = []

        if trajectory_list:
            trajectory = {a['id']: a for a in trajectory_list}
            np.save(output_dir + '/trajectory_match_100_%d.npy' % radius, trajectory)
        if error_tra_list:
            np.save(output_dir + '/error_trajectory_100_%d.npy' % radius, error_tra_list)
