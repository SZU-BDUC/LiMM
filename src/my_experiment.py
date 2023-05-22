import time
import os
import pandas as pd
import numpy as np
import osm2gmns
from tqdm import tqdm

import psutil
from math import sqrt, ceil, degrees, asin
from data_process.query_osm import get_segment_dict, get_all_segment_dict, get_small_dict
from data_process.hexagon import big_hexagon_center, get_segment_cell, get_foot_point_cell, get_key, get_small_cells, \
    get_all_cell, label_point
from data_process.dis_latlon import get_bear, distance_meter

from hmm_map_matching import HmmMapMatching

from learned_index import UseLearnedIndex

from ReinforceLearningRadius import UseReinforceLearnedRadius

# road network processing
class GetSegments(object):
    def __init__(self, config, location):
        self.config = config
        self.location = location
        self.side_length = 500
        self.threshold = 256
        if self.location == 'shenzhen':

            self.lat_min = 22.45
            self.lat_max = 22.83
            self.lon_min = 113.75
            self.lon_max = 114.58
        elif self.location == 'chengdu':

            self.lat_min = 30.64
            self.lat_max = 30.74
            self.lon_min = 104.03
            self.lon_max = 104.13
        elif self.location == 'big_chengdu':
            self.lat_min = 30.4
            self.lat_max = 30.9
            self.lon_min = 103.8
            self.lon_max = 104.3
        else:
            # porto
            self.lat_min = 41.05
            self.lat_max = 41.25
            self.lon_min = -8.75
            self.lon_max = -8.45
        print(self.location, self.lat_min, self.lat_max, self.lon_min, self.lon_max)
        self.get_segment()

    def get_segment(self):
        map_dir = self.config.data_dir + '/' + self.location + '.osm'
        if not os.path.exists(self.config.data_dir + '/unconsolidated'):
            get_map = osm2gmns.getNetFromFile(map_dir, strict_mode=True, default_lanes=False,
                                              network_types=('auto', 'bike', 'walk'))

            osm2gmns.outputNetToCSV(get_map, output_folder=self.config.data_dir + '/unconsolidated')

        link_dir = self.config.data_dir + '/unconsolidated/link.csv'
        node_dir = self.config.data_dir + '/unconsolidated/node.csv'

        if not os.path.exists(self.config.data_dir + '/hexagon'):
            os.makedirs(self.config.data_dir + '/hexagon')

        # osm segments split
        if not os.path.exists(self.config.data_dir + '/segment_all_dict.npy'):


            print('\nroad segment split')

            get_original_segment = get_segment_dict(link_dir, node_dir, self.lat_min, self.lat_max, self.lon_min,
                                                    self.lon_max)
            original_segment = get_all_segment_dict(get_original_segment)
            np.save(self.config.data_dir + '/original_segment.npy', original_segment)

            segment_dict = get_small_dict(get_original_segment)

            print('road segment split finish\n')


            # hexagon split
            print('hexagon split')
            big_hexagon = big_hexagon_center(self.lat_min, self.lat_max, self.lon_min, self.lon_max, self.side_length)
            np.save(self.config.data_dir + '/hexagon/big_hexagon.npy', big_hexagon)

            segment_split_dict = get_segment_cell(segment_dict, big_hexagon, self.side_length, self.lat_min,
                                                  self.lat_max, self.lon_min, self.lon_max)
            print('hexagon split finish\n')

            print('get vertical position cell')
            foot_point_get_big_hexagon, count_fractal_big_hexagon = get_foot_point_cell(segment_split_dict, big_hexagon,
                                                                                        self.side_length, self.threshold)
            print('get vertical position cell finish\n')

            np.save(self.config.data_dir + '/hexagon/foot_point_get_big_hexagon.npy', foot_point_get_big_hexagon)
            np.save(self.config.data_dir + '/hexagon/count_fractal_big_hexagon.npy', count_fractal_big_hexagon)

            print('get all segment dict')
            segment_all_dict = get_all_segment_dict(segment_split_dict)
            print('get all segment dict finish\n')
            np.save(self.config.data_dir + '/segment_all_dict.npy', segment_all_dict)



# using Limm for HMM
class HMMLimm(object):
    def __init__(self, config):
        self.config = config
        self.side_length = 500
        self.small_side_length = self.side_length / sqrt(7)
        self.threshold = 256

        self.process()

        self.big_hexagon_path = np.load(self.config.data_dir + '/hexagon/big_hexagon.npy', allow_pickle=True)
        self.big_hexagon = self.big_hexagon_path.item()
        self.count_fractal_hexagon_path = np.load(self.config.data_dir + '/hexagon/count_fractal_big_hexagon.npy',
                                                  allow_pickle=True)
        self.count_fractal_hexagon = []
        for item in self.count_fractal_hexagon_path:
            self.count_fractal_hexagon.append(item)


        self.get_small_hexagon_path = np.load(self.config.data_dir + '/hexagon/get_small_hexagon.npy', allow_pickle=True)
        self.get_small_hexagon = self.get_small_hexagon_path.item()

        self.q_table_path = self.config.result_dir + '/hexagon_q_table'
        if not os.path.exists(self.q_table_path):
            # create hexagon_q_table
            print('-'*50)
            print('creating q_table')
            UseReinforceLearnedRadius(self.config)
            print('creating q_table finish')
            print('-' * 50)
            # print('need to check hexagon_q_table')
            # print('_________________________')
            # print('pre end~')
            # quit()

        self.save_hexagon_path = self.config.data_dir + '/hexagon/hexagon_radius.npy'
        if not os.path.exists(self.save_hexagon_path):
            hexagon_list = list()
            for root, dirs, files in os.walk(self.q_table_path):
                for file in files:
                    # print(file, float(os.path.splitext(file)[0].split('_')[0]))
                    if os.path.splitext(file)[-1] == '.csv':
                        q_table = pd.read_csv(self.q_table_path + '/' + file, float_precision='round_trip', index_col=0)
                        redu = max(q_table['reduce'])
                        enla = max(q_table['enlarge'])
                        keep = max(q_table['keep'])
                        # 2 state
                        # if max(q_table['reduce']) > max(q_table['enlarge']):
                        #     new_radius = q_table['reduce'].idxmax() * 10 # + 20
                        # else:
                        #     new_radius = (q_table['enlarge'].idxmax() + 1) * 10 # + 20
                        # 3 state
                        if redu == max(redu, enla, keep):
                            new_radius = q_table['reduce'].idxmax() * 10
                        elif enla == max(redu, enla, keep):
                            new_radius = q_table['enlarge'].idxmax() * 10
                        else:
                            new_radius = q_table['keep'].idxmax() * 10

                        hexagon = {
                            'label_id': float(os.path.splitext(file)[0].split('_')[0]),
                            'radius': new_radius
                        }
                        hexagon_list.append(hexagon)

            self.save_hexagon = {a['label_id']: a for a in hexagon_list}
            # print(save_hexagon)
            np.save(self.save_hexagon_path, self.save_hexagon)
        else:
            save_hexagon_load = np.load(self.save_hexagon_path, allow_pickle=True)
            self.save_hexagon = save_hexagon_load.item()

    def process(self):
        big_hexagon = np.load(self.config.data_dir + '/hexagon/big_hexagon.npy', allow_pickle=True)
        big_hexagon = big_hexagon.item()

        foot_point_get_big_hexagon = np.load(self.config.data_dir + '/hexagon/foot_point_get_big_hexagon.npy',
                                             allow_pickle=True)
        foot_point_get_big_hexagon = foot_point_get_big_hexagon.item()

        count_fractal_big_hexagon_load = np.load(self.config.data_dir + '/hexagon/count_fractal_big_hexagon.npy',
                                                 allow_pickle=True)
        count_fractal_big_hexagon = []
        for item in count_fractal_big_hexagon_load:
            count_fractal_big_hexagon.append(item)

        if not os.path.exists(self.config.data_dir + '/hexagon/get_small_hexagon.npy'):
            print('\nget small hexagon')
            get_small_hexagon = get_small_cells(count_fractal_big_hexagon, self.side_length, big_hexagon)
            np.save(self.config.data_dir + '/hexagon/get_small_hexagon.npy', get_small_hexagon)
            print('\nget small hexagon finish\n')
        else:
            get_small_hexagon = np.load(self.config.data_dir + '/hexagon/get_small_hexagon.npy', allow_pickle=True)
            get_small_hexagon = get_small_hexagon.item()

        if os.path.exists(self.config.data_dir + '/hexagon/foot_point_get_all_hexagon.npy') is False:
            print('get small vertical position cell')
            foot_point_get_all_hexagon = get_all_cell(count_fractal_big_hexagon, foot_point_get_big_hexagon,
                                                      get_small_hexagon, self.small_side_length)
            np.save(self.config.data_dir + '/hexagon/foot_point_get_all_hexagon.npy', foot_point_get_all_hexagon)
            print('get small vertical position cell finish\n')
        else:
            foot_point_get_all_hexagon = np.load(self.config.data_dir + '/hexagon/foot_point_get_all_hexagon.npy',
                                                 allow_pickle=True)
            foot_point_get_all_hexagon = foot_point_get_all_hexagon.item()

        if os.path.exists(self.config.data_dir + '/hexagon/key_dict.npy') is False:
            print('get all key dict')
            key_dict, all_key = get_key(foot_point_get_all_hexagon, self.side_length, 60)
            np.save(self.config.data_dir + '/hexagon/key_dict.npy', key_dict)
            np.save(self.config.data_dir + '/hexagon/all_key.npy', all_key)
            print('get all key dict finish\n')

    # # using limm for HMM
    def run(self, original_radius):

        # Calculate IO
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

        original_segment = np.load(self.config.data_dir + '/original_segment.npy', allow_pickle=True)
        original_segment = original_segment.item()

        print('\nlearned index training\n')
        learned_index = UseLearnedIndex(self.config)
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
        print('-'*50)
        read_mega_bytes_sec = round(read_speed / (1024 ** 2), 2)
        print('read_mega_bytes_sec', read_mega_bytes_sec)
        write_mega_bytes_sec = round(write_speed / (1024 ** 2), 2)
        print('write_mega_bytes_sec', write_mega_bytes_sec)

        disk_io_counter = psutil.disk_io_counters()
        disk_total = disk_io_counter[2] + disk_io_counter[3]  # read_bytes + write_bytes

        a = disk_usage_process / disk_total * 100

        print('disk_usage_process / disk_total', a)

        print("## Build Learned Index costs {:.4f} s".format(time_diff))

        total = len(locals().values()) + len(globals().values()) + len(learned_index.all_segment) * 7

        print("## Learned Index Size %.0f" % total)

        print('-' * 50)

        output_dir = self.config.result_dir + '/limm'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            output_dir = self.config.result_dir + '/limm/limm_rein_text_5000'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            else:
                output_dir = self.config.result_dir + '/limm/limm_shenzhen_5000SML'
                os.makedirs(output_dir)
        print('output_dir', output_dir)
        
        trajectory_list = list()
        error_tra_list = []
        num = 0

        mm_text = HmmMapMatching(original_segment)
        mm_text.get_graph(original_segment)
        mm_text.get_node(original_segment)

        print('\nHMMLimm matching')
        for root, dirs, files in os.walk(self.config.trajectory_dir):
            for file in tqdm(files):
                if os.path.splitext(file)[-1] == '.csv':
                    # print(num, file)
                    track_file = pd.read_csv(self.config.trajectory_dir + '/' + file, float_precision='round_trip')
                    track = []
                    for i in range(track_file.shape[0]):
                        a = float(track_file.lat[i])
                        b = float(track_file.lon[i])
                        track.append((a, b))

                    all_candidate = []
                    all_candidate_time = []

                    start = time.time()
                    for track_node in track:
                        cell_id = label_point(track_node, self.side_length, self.big_hexagon)
                        cell_center_point = self.big_hexagon[cell_id]['center_point']
                        big_degree = get_bear(cell_center_point, track_node)
                        big_meter = round(distance_meter(cell_center_point, track_node), 0)

                        if cell_id in self.count_fractal_hexagon:
                            small_label = ceil(
                                (((big_degree + 60 - degrees(asin((1 / 2) / sqrt(7)))) + 360) % 360) / 60)
                            small_point = self.get_small_hexagon[round(cell_id + 0.1 * small_label, 1)]['center_point']
                            to_small_meter = distance_meter(small_point, track_node)

                            if big_meter <= to_small_meter:
                                label = cell_id

                            else:
                                label = round(cell_id + 0.1 * small_label, 1)
                        else:
                            label = cell_id

                        if label in self.save_hexagon:
                            radius = float(self.save_hexagon[label]['radius'])
                        else:
                            radius = original_radius
                        # print(radius)

                        add_num = 0
                        candidate_times = 0
                        while True:
                            new_radius = radius + 5 * add_num
                            add_num = add_num + 1

                            get_candidate_segments, candidate_time = learned_index.search(track_node, new_radius)
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
                        'candidate_time': all_candidate_time
                    }

                    trajectory_list.append(trajectory_match)
                    num += 1
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

        print('HMMLimm matching finish')
        print('-' * 50)

        if trajectory_list:
            trajectory = {a['id']: a for a in trajectory_list}
            np.save(output_dir + '/trajectory_match_%d.npy' % num, trajectory)
        if error_tra_list:
            np.save(output_dir + '/error_trajectory_%d.npy' % num, error_tra_list)


