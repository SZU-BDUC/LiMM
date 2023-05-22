import pandas as pd
from data_process.dis_latlon import distance_meter, get_bear
from tqdm import tqdm


def get_segment_dict(link_dir, node_dir, lat_min, lat_max, lon_min, lon_max):
    link = pd.read_csv(link_dir, encoding='ANSI', float_precision='round_trip', low_memory=False)
    link.sort_values(by=['osm_way_id'], ignore_index=True)
    node = pd.read_csv(node_dir, encoding='ANSI', float_precision='round_trip', low_memory=False)

    # segment number
    osm_ids = link.osm_way_id
    osm_id_single = osm_ids.drop_duplicates()

    # merge
    osm_way_ids = []
    line_strings_index = []
    for osm_id in tqdm(osm_id_single):
        osm_indexs = link[osm_ids == osm_id].index.tolist()
        if len(osm_indexs) == 1:
            osm_way_ids.append(osm_id)
            line_strings_index.append(osm_indexs)
        else:
            split1 = [osm_indexs[0]]
            from_node_id = link.from_node_id[split1[-1]]
            to_node_id = link.to_node_id[split1[-1]]

            for i in range(1, len(osm_indexs)):
                if link.from_node_id[osm_indexs[i]] == to_node_id and link.to_node_id[osm_indexs[i]] != from_node_id:
                    split1.append(osm_indexs[i])
                    from_node_id = link.from_node_id[split1[-1]]
                    to_node_id = link.to_node_id[split1[-1]]
            if len(split1) == len(osm_indexs):
                osm_way_ids.append(osm_id)
                line_strings_index.append(split1)
            else:
                osm_indexs2 = [i for i in osm_indexs if i not in split1]
                split2 = [osm_indexs2[-1]]
                from_node_id = link.from_node_id[split2[-1]]
                to_node_id = link.to_node_id[split2[-1]]

                for j in reversed(range(0, len(osm_indexs2)-1)):
                    if link.from_node_id[osm_indexs2[j]] == to_node_id and \
                            link.to_node_id[osm_indexs2[j]] != from_node_id:
                        split2.append(osm_indexs2[j])
                        from_node_id = link.from_node_id[split2[-1]]
                        to_node_id = link.to_node_id[split2[-1]]
                if len(split2) == len(osm_indexs2):
                    osm_way_ids.append(osm_id)
                    line_strings_index.append(split1)
                    osm_way_ids.append(osm_id)
                    line_strings_index.append(split2)
                else:
                    for osm_index in osm_indexs:
                        osm_way_ids.append(osm_id)
                        line_strings_index.append([osm_index])

    osm_new_way_ids = []
    line_new_strings_index = []

    for i in tqdm(range(len(line_strings_index))):
        osm_way_id = osm_way_ids[i]
        sub_line = line_strings_index[i]
        if len(sub_line) == 1:
            osm_new_way_ids.append(osm_way_id)
            line_new_strings_index.append(sub_line)
        else:
            split3 = []
            for j in range(len(sub_line)):
                to_node = link.to_node_id[sub_line[j]]
                node_index = node[node.node_id == to_node].index.tolist()
                if node.osm_highway[node_index[0]] == node.osm_highway[node_index[0]] or \
                        node.ctrl_type[node_index[0]] == node.ctrl_type[node_index[0]]:
                    split3.append(sub_line[j])

            if len(split3) == 0:
                osm_new_way_ids.append(osm_way_id)
                line_new_strings_index.append(sub_line)
            else:
                splits = []
                for split in sub_line:
                    if split not in split3:
                        splits.append(split)
                        if split == sub_line[-1]:
                            osm_new_way_ids.append(osm_way_id)
                            line_new_strings_index.append(splits)
                    else:
                        splits.append(split)
                        osm_new_way_ids.append(osm_way_id)
                        line_new_strings_index.append(splits)
                        splits = []

    all_osm_id = []
    all_road_points = []
    all_dis = []

    for k in tqdm(range(len(line_new_strings_index))):
        line_strings_indexs = line_new_strings_index[k]
        road_points = []

        for o_id in line_strings_indexs:
            geometry = link.geometry[o_id]
            line_strings = geometry.strip('LINESTRING ').strip('()').split(',')
            num_line_strings = len(line_strings)

            if not road_points:
                for i in range(num_line_strings):
                    point = line_strings[i].strip(' ').split(' ')
                    road_points.append((float(point[1]), float(point[0])))
            else:
                for i in range(1, num_line_strings):
                    point = line_strings[i].strip(' ').split(' ')
                    road_points.append((float(point[1]), float(point[0])))

        distance1 = []
        for j in range(1, len(road_points)):
            dis = distance_meter(road_points[j - 1], road_points[j])
            distance1.append(dis)

        # 添加节点
        add_points = []
        sep = []
        for y in range(len(distance1)):
            if distance1[y] > 200:
                add_point_num = int(distance1[y] // 200)
                for z in range(add_point_num):
                    add_point = (round(float(((add_point_num - z)*road_points[y][0] + (z+1)*road_points[y+1][0]) /
                                             (add_point_num + 1)), 7),
                                 round(float(((add_point_num - z)*road_points[y][1] + (z+1)*road_points[y+1][1]) /
                                             (add_point_num + 1)), 7))
                    add_points.append(add_point)
                    sep.append(y)

        if sep:
            single_sep = sorted(list(set(sep)))

            for insert in reversed(single_sep):
                sep_index = [i for i, x in enumerate(sep) if x is insert]
                for num in range(len(sep_index)):
                    road_points.insert(insert + 1 + num, add_points[sep_index[num]])

        distance = []
        for j in range(1, len(road_points)):
            dis = distance_meter(road_points[j - 1], road_points[j])
            distance.append(dis)

        # length
        all_distance = distance[0]
        road_point = []
        for x in range(len(distance)):
            if x != len(distance) - 1:
                if all_distance <= 200:
                    road_point.append(road_points[x])
                    all_distance += distance[x+1]

                else:
                    road_point.append(road_points[x])
                    all_osm_id.append(osm_new_way_ids[k])
                    all_road_points.append(road_point)
                    all_dis.append(all_distance - distance[x])

                    all_distance = distance[x] + distance[x + 1]
                    road_point = [road_points[x]]

            elif all_distance > 200:
                if distance[-1] < 75:
                    road_point.append(road_points[x])
                    road_point.append(road_points[x + 1])

                    all_osm_id.append(osm_new_way_ids[k])
                    all_road_points.append(road_point[:len(road_point) // 2 + 1])
                    new_distance = 0
                    for i in range(len(road_point) // 2):
                        new_distance += distance_meter(road_point[i], road_point[i + 1])
                    all_dis.append(new_distance)

                    all_osm_id.append(osm_new_way_ids[k])
                    all_road_points.append(road_point[len(road_point) // 2:])
                    new_distance = 0
                    for i in range(len(road_point) - len(road_point) // 2 - 1):
                        new_distance += distance_meter(road_point[i + len(road_point) // 2],
                                                       road_point[i + 1 + len(road_point) // 2])
                    all_dis.append(new_distance)

                else:
                    road_point.append(road_points[x])
                    all_osm_id.append(osm_new_way_ids[k])
                    all_road_points.append(road_point)
                    all_dis.append(all_distance - distance[-1])

                    all_osm_id.append(osm_new_way_ids[k])
                    all_road_points.append([road_points[x], road_points[x + 1]])
                    all_dis.append(distance[-1])

            else:
                road_point.append(road_points[x])
                road_point.append(road_points[x + 1])

                all_osm_id.append(osm_new_way_ids[k])
                all_road_points.append(road_point)
                all_dis.append(all_distance)

    link_new_dir2 = pd.DataFrame({
        'osm_id': all_osm_id,
        'line_strings_index': all_road_points,
        'dis': all_dis
    })

    # curve
    degree_road_point = []
    for i in tqdm(range(len(link_new_dir2))):
        road_points_len = len(link_new_dir2.line_strings_index[i])
        road_point_string = link_new_dir2.line_strings_index[i]
        if road_points_len == 2:
            degree_road_point.append(road_point_string)
        else:
            degree = []
            for j in range(1, road_points_len):
                bearing = get_bear(road_point_string[j-1], road_point_string[j])
                degree.append(bearing)

            subtraction_degree = []
            for k in range(1, len(degree)):
                subtraction_degree.append(degree[k] - degree[k - 1])
            if abs(sum(subtraction_degree)/len(subtraction_degree)) <= 5:
                degree_road_point.append([road_point_string[0], road_point_string[-1]])
            else:
                degree_road_point.append([road_point_string[0], road_point_string[road_points_len//2],
                                          road_point_string[-1]])

    link_new_dir2['new_road_point'] = degree_road_point
    # link_new_dir2.to_csv('link_new_dir2.csv')
    # print(link_new_dir2)

    link_sub = list()
    link_id = 0
    for i in tqdm(range(link_new_dir2.shape[0])):
        for j in range(len(link_new_dir2.new_road_point[i])-1):
            link_id += 1

            node_a_idx_y = node[node['y_coord'] == link_new_dir2.new_road_point[i][j][0]].index.tolist()

            node_a_idx_x = node[node['x_coord'] == link_new_dir2.new_road_point[i][j][1]].index.tolist()

            node_a_idx = list(set(node_a_idx_x) & set(node_a_idx_y))

            if len(node_a_idx) != 0:
                node_a_id = node.osm_node_id[node_a_idx[0]]
            else:
                node_a_id = []

            node_b_idx_y = node[node['y_coord'] == link_new_dir2.new_road_point[i][j+1][0]].index.tolist()

            node_b_idx_x = node[node['x_coord'] == link_new_dir2.new_road_point[i][j+1][1]].index.tolist()

            node_b_idx = list(set(node_b_idx_x) & set(node_b_idx_y))

            if len(node_b_idx) != 0:
                node_b_id = node.osm_node_id[node_b_idx[0]]
            else:
                node_b_id = []

            if lat_min <= link_new_dir2.new_road_point[i][j][0] <= lat_max and \
                    lon_min <= link_new_dir2.new_road_point[i][j][1] <= lon_max and \
                    lat_min <= link_new_dir2.new_road_point[i][j+1][0] <= lat_max and \
                    lon_min <= link_new_dir2.new_road_point[i][j+1][1] <= lon_max:
                link_sub1 = {
                    'link_id': str(link_id),
                    'osm_way_id': all_osm_id[i],
                    'name': link.name[link[link.osm_way_id == all_osm_id[i]].index.tolist()[0]],
                    'length': distance_meter(link_new_dir2.new_road_point[i][j], link_new_dir2.new_road_point[i][j + 1]),
                    'node_a_id': node_a_id,
                    'node_a': link_new_dir2.new_road_point[i][j],
                    'node_b_id': node_b_id,
                    'node_b': link_new_dir2.new_road_point[i][j+1]
                }
                link_sub.append(link_sub1)
    # print(link_sub)
    return {link1['link_id']: link1 for link1 in link_sub}


def get_small_dict(segment_dict):
    # If the segment is less than 100m, keep the whole section. Otherwise, keep the segmented sub-section
    # porto chengdu 25
    each_sub_road_length = 50 # shenzhen
    segment_dict_new = list()
    for s_id in tqdm(segment_dict):
        add_points = []
        if segment_dict[s_id]['length'] > each_sub_road_length:
            add_points.append(segment_dict[s_id]['node_a'])
            add_point_num = int(segment_dict[s_id]['length'] // each_sub_road_length)
            for z in range(add_point_num):
                add_point = (
                    round(float(((add_point_num - z) * segment_dict[s_id]['node_a'][0] + (z + 1) * segment_dict[s_id]['node_b'][0]) /
                                (add_point_num + 1)), 7),
                    round(float(((add_point_num - z) * segment_dict[s_id]['node_a'][1] + (z + 1) * segment_dict[s_id]['node_b'][1]) /
                                (add_point_num + 1)), 7))
                add_points.append(add_point)
            add_points.append(segment_dict[s_id]['node_b'])
            # print(add_points)
            for i in range(len(add_points) - 1):
                link_sub1 = {
                    'link_id': s_id + '_' + str(i),
                    'osm_way_id': segment_dict[s_id]['osm_way_id'],
                    'name': segment_dict[s_id]['name'],
                    'length': distance_meter(add_points[i], add_points[i+1]),
                    'node_a_id': segment_dict[s_id]['node_a_id'] if i == 0 else [],
                    'node_a': add_points[i],
                    'node_b_id': segment_dict[s_id]['node_b_id'] if i == len(add_points)-1 else [],
                    'node_b': add_points[i+1]
                }
                segment_dict_new.append(link_sub1)
        else:
            segment_dict_new.append(segment_dict[s_id])
    return {segment['link_id']: segment for segment in segment_dict_new}


def get_all_segment_dict(segment_split_dict):

    node = []

    for link_id in tqdm(segment_split_dict):
        node_a_id = segment_split_dict[link_id]['node_a_id']
        node_b_id = segment_split_dict[link_id]['node_b_id']

        if not node_a_id or str(node_a_id)[0] == 'n':
            node.append(segment_split_dict[link_id]['node_a'])

        if not node_b_id or str(node_b_id)[0] == 'n':
            node.append(segment_split_dict[link_id]['node_b'])

    node = pd.DataFrame({'nodes': node})
    node.drop_duplicates(inplace=True, ignore_index=True)

    for link_id in tqdm(segment_split_dict):
        node_a_id = segment_split_dict[link_id]['node_a_id']
        node_b_id = segment_split_dict[link_id]['node_b_id']

        if not node_a_id or str(node_a_id)[0] == 'n':
            segment_split_dict[link_id]['node_a_id'] = \
                str(node[node['nodes'] == segment_split_dict[link_id]['node_a']].index.tolist()[0])

        if not node_b_id or str(node_b_id)[0] == 'n':
            segment_split_dict[link_id]['node_b_id'] = \
                str(node[node['nodes'] == segment_split_dict[link_id]['node_b']].index.tolist()[0])

    return segment_split_dict
