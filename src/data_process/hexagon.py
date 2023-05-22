from data_process.dis_latlon import distance_meter, get_bear, destination, get_intersection_point,  distance_point_to_segment
from math import sqrt, floor, cos, radians, ceil, acos, degrees, asin
import pandas as pd
from tqdm import tqdm


# Calculate the center point of the big hexagon
def big_hexagon_center(lat_min, lat_max, lon_min, lon_max, side_length):

    """
    :param lat_min:
    :param lat_max:
    :param lon_min:
    :param lon_max:
    :param side_length:
    :return: Center point of traversal range (latitude and longitude, number of layers, ID)
        Draw hexagons with pointed center points pointy
        North is 0°, increasing clockwise
    """

    center_lat = round((lat_min + lat_max) / 2, 7)
    center_lon = round((lon_min + lon_max) / 2, 7)

    dis_max = distance_meter((center_lat, center_lon), (lat_max, lon_max))
    level = int(round(dis_max/(1.5*side_length), 0))

    center_points_2ds = []
    center_point = (center_lat, center_lon)
    center_points_2ds.append(center_point)
    center_points_2ds.append(center_point)

    labels = [0, 0]

    levels = [0, 0]

    # Draw hexagons with pointed center points pointy
    for i in tqdm(range(level+1)):
        for j in range(7):
            if j == 0:
                del center_points_2ds[-1]
                del labels[-1]
                del levels[-1]

                center_point = destination((center_lat, center_lon), bearing=0, dist=side_length*sqrt(3)*i)
                # North is 0°, increasing clockwise

                center_points_2ds.append(center_point)

                label = center_points_2ds.index(center_point)  # 3*i*(i-1) + 1
                labels.append(label)

                levels.append(i)
            else:
                for k in range(i):

                    center_point = destination(center_points_2ds[-1], bearing=120+60*(j-1), dist=side_length * sqrt(3))
                    center_points_2ds.append(center_point)

                    label = center_points_2ds.index(center_point)  # 3*i*(i-1) + i*(j-1) + k + 2
                    labels.append(label)

                    levels.append(i)
    del center_points_2ds[-1]
    del labels[-1]
    del levels[-1]
    lat_max1, lon_max1 = destination((lat_max, lon_max), 45, side_length*sqrt(2)*3)
    lat_min1, lon_min1 = destination((lat_min, lon_min), 225, side_length*sqrt(2)*3)
    # print(center_points_2ds)
    center_points_2ds = [None if i[0] >= lat_max1 or i[0] <= lat_min1 or i[1] >= lon_max1 or i[1] <= lon_min1
                         else i for i in center_points_2ds]

    center_points = pd.DataFrame({
        'center_point': center_points_2ds,
        'level': levels,
        'label': labels
    })
    # print(center_points)
    center_points = center_points.dropna(axis=0, how='any').reset_index(drop=True)

    center_points_dict = list()

    for i in range(center_points.shape[0]):
        center_point_dic = {
            'label': center_points.label[i],
            'level': center_points.level[i],
            'center_point': center_points.center_point[i]
        }
        center_points_dict.append(center_point_dic)

    return {center_point['label']: center_point for center_point in center_points_dict}


# calculate hierarchy
def get_level(label_center):
    """
    :param label_center: cell id
    :return: level
    """

    a = label_center // 6

    sqrt2a = floor(sqrt(2 * a))
    label1 = 3 * sqrt2a * (sqrt2a - 1)
    label2 = 3 * (sqrt2a + 1) * sqrt2a
    label3 = 3 * (sqrt2a + 2) * (sqrt2a + 1)

    if label1 < label_center <= label2:
        level = sqrt2a
    elif label2 < label_center <= label3:
        level = sqrt2a + 1
    elif label_center == 0:
        level = 0
    else:
        level = None
        print('error: need to check get level')
        exit()

    return level


# calculate  neighborhood
def get_near_labels(label_center):
    """
    :param label_center: cell ID
    :return: cell ID nearby 6 ID
    """

    level = get_level(label_center)

    if level >= 1:
        num = label_center-3*level*(level-1)-1
        num_a = num // level
        num_mod = num % level

        if label_center == 6:
            near_label1 = 0
            near_label2 = 1
            near_label3 = 5
            near_label4 = 16
            near_label5 = 17
            near_label6 = 18

        elif num_mod != 0 and num != 6*level - 1:
            # Two on the back floor, two on the front floor, two on the same floor
            near_label1 = 3 * (level - 1) * (level - 2) + num_a * (level - 1) + num_mod  # The number of the previous layer
            near_label2 = 3 * (level - 1) * (level - 2) + num_a * (level - 1) + num_mod + 1
            near_label3 = label_center - 1
            near_label4 = label_center + 1
            near_label5 = 3 * (level + 1) * level + num_a * (level + 1) + num_mod + 1  # The numbers in the next layer
            near_label6 = 3 * (level + 1) * level + num_a * (level + 1) + num_mod + 2

        elif num_mod == 0 and num != 0:
            # Three on the back floor, one on the front, two on the same floor
            near_label1 = 3*(level-1)*(level-2) + num_a*(level-1) + 1  # The number of the previous layer
            near_label2 = label_center - 1
            near_label3 = label_center + 1
            near_label4 = 3*(level+1)*level + num_a*(level+1)  # The numbers in the next layer
            near_label5 = 3*(level+1)*level + num_a*(level+1) + 1
            near_label6 = 3*(level+1)*level + num_a*(level+1) + 2

        elif num == 0:
            # The first number of each level
            near_label1 = label_center - 6 * (level-1)  # The number of the previous layer
            near_label2 = label_center + 1  # Number on the right
            near_label3 = label_center + 6 * level - 1  # The number on the left
            near_label4 = label_center + 6 * level  # The numbers in the next layer
            near_label5 = label_center + 6 * level + 1
            near_label6 = label_center + 6 * (2 * level + 1) - 1

        else:
            # The last number of each level
            near_label3 = 3 * (level - 1) * (level - 2) + 1  # The number of the previous layer
            near_label4 = 3 * level * (level - 1)
            near_label2 = label_center - 6 * level + 1  # Number on the right
            near_label5 = label_center - 1  # The number on the left
            near_label6 = 3 * (level + 2) * (level + 1) - 1  # The numbers in the next layer
            near_label1 = 3 * (level + 2) * (level + 1)

    else:
        near_label1 = 1
        near_label2 = 2
        near_label3 = 3
        near_label4 = 4
        near_label5 = 5
        near_label6 = 6

    near_labels = [int(near_label1), int(near_label2), int(near_label3), int(near_label4), int(near_label5),
                   int(near_label6)]
    # North is 0°, increasing clockwise
    if level == 1:
        near_labels[near_labels.index(min(near_labels))] = int(0)

    return near_labels


# calculate label
def label_point(p, side_length, center_points):
    """
    :param p:
    :param side_length:
    :param center_points:
    :return: A's cell ID
    """
    min_side_length = side_length * sqrt(3) / 2
    opf_side_length = 1.5 * side_length

    center_point = center_points[0]['center_point']

    bearing = get_bear(center_point, p)
    # print('bearing', bearing)
    meter = distance_meter(center_point, p)
    # print('meter', meter)
    dis_min_to_center = meter * cos(radians(abs((bearing % 60)-30)))  # cos is radians, math.radians
    level = int(ceil((dis_min_to_center-side_length)/opf_side_length))  # plus 1
    # print('level', level)

    if level is 0:
        label_t = 0
    elif bearing < 360-(30/level):
        label_t = 3 * level * (level - 1) + round(bearing / (360 / (6 * level)), 0) + 1
    else:
        label_t = 3 * level * (level - 1) + 1
    # print('p', p)
    # print('label_point', label_t)
    if label_t in center_points:
        find_center_point = center_points[label_t]['center_point']
        dis_find = distance_meter(find_center_point, p)
    else:
        dis_find = float('inf')

    # print('level', level)
    # print('label_t', label_t)
    # print('dis_find', dis_find)
    # print('*'*30)

    text_label = []
    text_dis = []

    if dis_find > opf_side_length:
        text_level = [level-1, level+1]
        for text in text_level:
            if text == 0:
                label_p = 0
            elif bearing < 360 - (30 / text):
                label_p = 3 * text * (text - 1) + round(bearing / (360 / (6 * text)), 0) + 1
            else:
                label_p = 3 * text * (text - 1) + 1
            text_label.append(label_p)
            if label_p in center_points:
                text_center_point = center_points[label_p]['center_point']
                dis_f = distance_meter(text_center_point, p)
                text_dis.append(dis_f)
            else:
                text_dis.append(float('inf'))

        text_level.append(level)
        text_label.append(label_t)
        text_dis.append(dis_find)

        # print('level', text_level)
        # print('label_t', text_label)
        # print('dis_find', text_dis)
        # print('*' * 30)

        if min(text_dis) <= opf_side_length:
            idx = text_dis.index(min(text_dis))
            label_t = text_label[idx]
            dis_find = text_dis[idx]

        # else:
        #     print('error need to check label_point')
            # print('point', p)
            # print('level', level)
            # print('label_t', label_t)
            # print('dis_find', dis_find)
            # print('*' * 30)

    # print('level', level)
    # print('label_t', label_t)
    # print('dis_find', dis_find)
    # print('*' * 30)

    if dis_find < min_side_length:
        label_points = label_t
    else:
        near_labels = get_near_labels(label_t)
        # print('near_labels', near_labels)

        dis_nears = []
        dis_nears1 = []

        for near in near_labels:
            if near in center_points:
                near_label_point = center_points[near]['center_point']
                dis_near = distance_meter(near_label_point, p)
                dis_nears.append(dis_near)
            else:
                dis_nears.append(float('inf'))
        # print('dis_nears', dis_nears)

        if dis_find <= min(dis_nears) and dis_find <= side_length + 1:
            label_points = label_t
        elif min(dis_nears) < dis_find and min(dis_nears) <= min_side_length:
            label_points = near_labels[dis_nears.index(min(dis_nears))]
        else:
            near_labels1 = get_near_labels(near_labels[dis_nears.index(min(dis_nears))])
            # print('near_labels1', near_labels1)

            for near in near_labels1:
                if near in center_points:
                    near_label_point1 = center_points[near]['center_point']
                    dis_near1 = distance_meter(near_label_point1, p)
                    dis_nears1.append(dis_near1)
                else:
                    dis_nears1.append(float('inf'))

            # print('dis_nears1', dis_nears1)

            if min(dis_nears) <= min(dis_nears1) and min(dis_nears) <= side_length + 1.5:
                label_points = near_labels[dis_nears.index(min(dis_nears))]
            elif min(dis_nears1) < min(dis_nears) and min(dis_nears1) <= side_length + 1.5:
                label_points = near_labels1[dis_nears1.index(min(dis_nears1))]
            else:
                print(p, 'error: need to get label point')
                label_points = -1
                # exit()

    return int(label_points)


# divide segment according to hexagons
def get_segment_cell(segment_dict, big_center_point, side_length, lat_min, lat_max, lon_min, lon_max):
    segment_dict_new = list()

    for segment_id in segment_dict:
        node_a = segment_dict[segment_id]['node_a']
        node_b = segment_dict[segment_id]['node_b']

        if lat_min <= node_a[0] <= lat_max and lon_min <= node_a[1] <= lon_max and lat_min <= node_b[0] <= lat_max \
                and lon_min <= node_b[1] <= lon_max:

            node_a_cell = label_point(node_a, side_length, big_center_point)
            node_a_near_cell = get_near_labels(node_a_cell)

            a_center = big_center_point[node_a_cell]['center_point']

            # print('node_b', node_b)
            node_b_cell = label_point(node_b, side_length, big_center_point)

            b_center = big_center_point[node_b_cell]['center_point']

            degree_edge = get_bear(a_center, b_center)

            if node_b_cell in node_a_near_cell:

                c = destination(a_center, round(degree_edge) + 30, side_length)

                d = destination(a_center, round(degree_edge) - 30, side_length)

                split_point = get_intersection_point(node_a, node_b, c, d)

                segment1 = {
                    'link_id': str(segment_id) + '_' + str(0),
                    'osm_way_id': segment_dict[segment_id]['osm_way_id'],
                    'name': segment_dict[segment_id]['name'],
                    'length': distance_meter(node_a, split_point),
                    'node_a_id': segment_dict[segment_id]['node_a_id'],
                    'node_a': node_a,
                    'node_b_id': [],
                    'node_b': split_point
                }

                segment_dict_new.append(segment1)

                segment2 = {
                    'link_id': str(segment_id) + '_' + str(1),
                    'osm_way_id': segment_dict[segment_id]['osm_way_id'],
                    'name': segment_dict[segment_id]['name'],
                    'length': distance_meter(split_point, node_b),
                    'node_a_id': [],
                    'node_a': split_point,
                    'node_b_id': segment_dict[segment_id]['node_b_id'],
                    'node_b': node_b
                }
                segment_dict_new.append(segment2)

            elif node_a_cell != node_b_cell:

                print('far:', segment_id)
                print(node_a)
                print(node_a_cell)
                print(node_b)
                print(node_b_cell)
                print('error: need to check get_segment_cell')

    segment_dict.update({segment['link_id']: segment for segment in segment_dict_new})
    return segment_dict


# Get the vertical position
def get_foot_point_cell(segment_dict, center_points, side_length, threshold):

    segment_ids = []
    lat = []
    lon = []
    labels = []
    dis = []
    degree = []

    for segment_id in tqdm(segment_dict):
        node_a = segment_dict[segment_id]['node_a']
        node_b = segment_dict[segment_id]['node_b']
        # print('node_b', segment_id, node_b)

        node_a_cell = label_point(node_a, side_length, center_points)

        node_a_near_cell = get_near_labels(node_a_cell)

        a_center = center_points[node_a_cell]['center_point']

        node_b_cell = label_point(node_b, side_length, center_points)

        b_center = center_points[node_b_cell]['center_point']

        if node_a_cell == node_b_cell:
            # foot_point, meter = get_foot_point(a_center, node_a, node_b)
            meter, foot_point, _ = distance_point_to_segment(a_center, node_a, node_b)

            bearing = get_bear(a_center, foot_point)

            segment_ids.append(segment_id)

            lat.append(foot_point[0])
            lon.append(foot_point[1])
            labels.append(node_a_cell)
            dis.append(meter)
            degree.append(bearing)

            if foot_point == node_a:
                # c = 1
                bearing = get_bear(a_center, node_b)
                meter = distance_meter(node_b, a_center)
                segment_ids.append(segment_id + '_' + 'e')

                lat.append(node_b[0])
                lon.append((node_b[1]))
                labels.append(node_b_cell)
                dis.append(meter)
                degree.append(bearing)

                segment_ids.append(segment_id + '_' + 'm')

                mid_lat = round((node_a[0] + node_b[0]) / 2, 7)
                mid_lon = round((node_a[1] + node_b[1]) / 2, 7)
                mid_bear = get_bear(a_center, (mid_lat, mid_lon))
                mid_meter = distance_meter((mid_lat, mid_lon), a_center)

                lat.append(mid_lat)
                lon.append(mid_lon)
                labels.append(node_a_cell)
                dis.append(mid_meter)
                degree.append(mid_bear)

            elif foot_point == node_b:
                # c = 1
                bearing = get_bear(a_center, node_a)
                meter = distance_meter(node_a, a_center)

                segment_ids.append(segment_id + '_' + 's')

                lat.append(node_a[0])
                lon.append((node_a[1]))
                labels.append(node_a_cell)
                dis.append(meter)
                degree.append(bearing)

                segment_ids.append(segment_id + '_' + 'm')

                mid_lat = round((node_a[0] + node_b[0]) / 2, 7)
                mid_lon = round((node_a[1] + node_b[1]) / 2, 7)
                mid_bear = get_bear(a_center, (mid_lat, mid_lon))
                mid_meter = distance_meter((mid_lat, mid_lon), a_center)

                lat.append(mid_lat)
                lon.append(mid_lon)
                labels.append(node_a_cell)
                dis.append(mid_meter)
                degree.append(mid_bear)
            else:
                bearing = get_bear(a_center, node_a)
                meter = distance_meter(node_a, a_center)
                segment_ids.append(segment_id + '_' + 's')

                lat.append(node_a[0])
                lon.append((node_a[1]))
                labels.append(node_a_cell)
                dis.append(meter)
                degree.append(bearing)

                bearing = get_bear(a_center, node_b)
                meter = distance_meter(node_b, a_center)

                segment_ids.append(segment_id + '_' + 'e')

                lat.append(node_b[0])
                lon.append((node_b[1]))
                labels.append(node_b_cell)
                dis.append(meter)
                degree.append(bearing)

        elif node_b_cell in node_a_near_cell:
            dis_aa = distance_meter(a_center, node_a)

            dis_ab = distance_meter(a_center, node_b)

            dis_ba = distance_meter(b_center, node_a)

            dis_bb = distance_meter(b_center, node_b)

            if min(dis_aa, dis_ab) <= min(dis_ba, dis_bb):
                segment_label = node_a_cell
            else:
                segment_label = node_b_cell

            mid_center = center_points[segment_label]['center_point']

            meter, foot_point, _ = distance_point_to_segment(mid_center, node_a, node_b)
            bearing = get_bear(mid_center, foot_point)

            segment_ids.append(segment_id)
            lat.append(foot_point[0])
            lon.append((foot_point[1]))
            labels.append(segment_label)
            dis.append(meter)
            degree.append(bearing)

            if foot_point == node_a:
                bearing = get_bear(a_center, node_b)
                meter = distance_meter(a_center, node_b)

                segment_ids.append(segment_id + '_' + 'e')

                lat.append(node_b[0])
                lon.append((node_b[1]))
                labels.append(node_b_cell)
                dis.append(meter)
                degree.append(bearing)

                segment_ids.append(segment_id + '_' + 'm')

                mid_lat = round((node_a[0] + node_b[0]) / 2, 7)
                mid_lon = round((node_a[1] + node_b[1]) / 2, 7)
                mid_bear = get_bear(mid_center, (mid_lat, mid_lon))
                mid_meter = distance_meter(mid_center, (mid_lat, mid_lon))

                lat.append(mid_lat)
                lon.append(mid_lon)
                labels.append(segment_label)
                dis.append(mid_meter)
                degree.append(mid_bear)
            elif foot_point == node_b:
                bearing = get_bear(a_center, node_a)
                meter = distance_meter(node_a, a_center)

                segment_ids.append(segment_id + '_' + 's')

                lat.append(node_a[0])
                lon.append((node_a[1]))
                labels.append(node_a_cell)
                dis.append(meter)
                degree.append(bearing)

                segment_ids.append(segment_id + '_' + 'm')

                mid_lat = round((node_a[0] + node_b[0]) / 2, 7)
                mid_lon = round((node_a[1] + node_b[1]) / 2, 7)
                mid_bear = get_bear(mid_center, (mid_lat, mid_lon))
                mid_meter = distance_meter(mid_center, (mid_lat, mid_lon))

                lat.append(mid_lat)
                lon.append(mid_lon)
                labels.append(segment_label)
                dis.append(mid_meter)
                degree.append(mid_bear)
            else:
                bearing = get_bear(a_center, node_a)
                meter = distance_meter(a_center, node_a)

                segment_ids.append(segment_id + '_' + 's')

                lat.append(node_a[0])
                lon.append((node_a[1]))
                labels.append(node_a_cell)
                dis.append(meter)
                degree.append(bearing)
                #

                bearing = get_bear(a_center, node_b)
                meter = distance_meter(a_center, node_b)

                segment_ids.append(segment_id + '_' + 'e')

                lat.append(node_b[0])
                lon.append((node_b[1]))
                labels.append(node_b_cell)
                dis.append(meter)
                degree.append(bearing)

        # else:
        #     print('far:', segment_id, segment_dict[segment_id]['osm_way_id'])
        #     print(node_a)
        #     print(node_a_cell)
        #     print(node_b)
        #     print(node_b_cell)
        #     print('error: need to check get_foot_point_cell')

    details = pd.DataFrame({
        'segment_id': segment_ids,
        'foot_point_lat': lat,
        'foot_point_lon': lon,
        'label': labels,
        'distance': dis,
        'degree': degree
    })

    details = details.sort_values(by=['label', 'distance', 'degree'], ignore_index=True)

    label_count = details['label'].value_counts()
    label_counts = pd.DataFrame(label_count)
    fractal_label = label_counts[label_counts['label'] >= threshold].index.tolist()

    # print('maximum threshold:', max(label_counts['label']))
    details_dict = list()

    for i in range(details.shape[0]):
        detail_dic = {
            'segment_id': details.segment_id[i],
            'foot_point_lat': details.foot_point_lat[i],
            'foot_point_lon': details.foot_point_lon[i],
            'label': details.label[i],
            'distance': details.distance[i],
            'degree': details.degree[i]
        }
        details_dict.append(detail_dic)
    # print('finish')

    return {detail['segment_id']: detail for detail in details_dict}, fractal_label


def fractal_point(label, side_length, big_hexagon):

    point = big_hexagon[label]['center_point']

    level = get_level(label)

    small_side = side_length/sqrt(7)

    small_points = [point]
    labels = [label]
    levels = [level]

    for i in range(6):
        degree = 60*i-degrees(acos((5/2)/sqrt(7)))
        small_point = destination(point, degree, small_side*sqrt(3))
        small_points.append(small_point)

        labels.append(round(label+0.1*(i+1), 1))
        levels.append(level + 0.1)

    small_center_points = pd.DataFrame({
        'center_point': small_points,
        'level': levels,
        'label': labels
    })
    small_center_points = small_center_points.dropna(axis=0, how='any').reset_index(drop=True)

    center_points_dict = list()

    for i in range(small_center_points.shape[0]):
        center_point_dic = {
            'label': small_center_points.label[i],
            'level': small_center_points.level[i],
            'center_point': small_center_points.center_point[i]
        }
        center_points_dict.append(center_point_dic)

    return {center_point['label']: center_point for center_point in center_points_dict}


def get_small_cells(fractal_cells, side_length, big_hexagon):

    center_points_dict = fractal_point(fractal_cells[0], side_length, big_hexagon)
    for i in tqdm(range(1, len(fractal_cells))):
        small_center = fractal_point(fractal_cells[i], side_length, big_hexagon)

        center_points_dict.update(small_center)

    return center_points_dict


def get_all_cell(count_fractal_hexagon, details, get_small_hexagon, small_side_length):
    for segment_id in tqdm(details):
        if details[segment_id]['label'] in count_fractal_hexagon \
                and details[segment_id]['distance'] > (small_side_length * sqrt(3) / 2):

            lat = details[segment_id]['foot_point_lat']
            lon = details[segment_id]['foot_point_lon']
            original_label = details[segment_id]['label']
            meter = details[segment_id]['distance']
            degree = details[segment_id]['degree']

            small_label = ceil((((degree + 60 - degrees(asin((1/2)/sqrt(7)))) + 360) % 360)/60)

            new_label = round(original_label + 0.1*small_label, 1)
            small_point = get_small_hexagon[new_label]['center_point']

            to_small_meter = distance_meter(small_point, (lat, lon))
            to_small_degree = get_bear(small_point, (lat, lon))

            if meter > to_small_meter:
                label = new_label

                details[segment_id]['label'] = label
                details[segment_id]['distance'] = to_small_meter
                details[segment_id]['degree'] = to_small_degree

    return details


def get_key(road_network, side_length, each_degree):
    c3 = 1
    c2 = 10 ** len(str(side_length))
    c1 = 10 ** (3 + len(str(side_length)))  # two-digit store degree_block, one-digit label decimal place

    key = []
    segment_ids = []
    for segment_id in tqdm(road_network):

        label = road_network[segment_id]['label']
        dis = int(round(road_network[segment_id]['distance'], 0))
        degree = road_network[segment_id]['degree']

        if (label - round(label, 0)) == 0:
            degree_block = ceil((degree % 360) / each_degree)
        else:
            degree_block = ceil(((degree + 30 - degrees(asin((1 / 2) / sqrt(7)))) % 360) / each_degree)

        key1 = c1 * label + c2 * degree_block + c3 * dis

        key.append(round(key1, 1))
        segment_ids.append(segment_id)

    key_dict = pd.DataFrame({
        'link_id': segment_ids,
        'key': key
    })
    key_dict = key_dict.sort_values(by=['key'], ignore_index=True)

    judge = key_dict.key[0]
    j = 0
    for i in tqdm(range(len(key) - 1)):
        if key_dict.key[i + 1] == judge:
            key_dict.loc[i+1, 'key'] = judge + (j+1)*1

            j = j + 1
        else:
            judge = key_dict.key[i + 1]
            j = 0

    all_key_dict = list()

    for j in tqdm(range(key_dict.shape[0])):

        # thick grit
        split_id = key_dict.link_id[j].split('_')
        link_id = split_id[0]

        # fine grit
        # link_id = key_dict.link_id[j]

        all_key = {
            'link_id': link_id,
            'key': key_dict.key[j],
            'pos': j*2
        }
        all_key_dict.append(all_key)

    return {key['pos']: key for key in all_key_dict}, list(key_dict.key)


def get_key_200(road_network, side_length, each_degree):
    c3 = 1
    c2 = 10 ** len(str(side_length))
    c1 = 10 ** (3 + len(str(side_length)))  # two-digit store degree_block, one-digit label decimal place

    key = []
    segment_ids = []
    for segment_id in road_network:

        label = road_network[segment_id]['label']
        degree = road_network[segment_id]['degree']

        text = ceil(((degree-30) % 360)/each_degree)
        degree_block = text if text != 0 else 6

        key1 = c1 * label + c2 * degree_block

        key.append(round(key1, 1))
        segment_ids.append(segment_id)

    key_dict = pd.DataFrame({
        'link_id': segment_ids,
        'key': key
    })
    key_dict = key_dict.sort_values(by=['key'], ignore_index=True)

    judge = key_dict.key[0]
    j = 0
    for i in range(len(key) - 1):
        if key_dict.key[i + 1] == judge:
            key_dict.loc[i+1, 'key'] = judge + (j+1)*1

            j = j + 1
        else:
            judge = key_dict.key[i + 1]
            j = 0

    all_key_dict = list()

    for j in range(key_dict.shape[0]):

        # thick grit
        split_id = key_dict.link_id[j].split('_')
        link_id = split_id[0]

        # fine grit
        # link_id = key_dict.link_id[j]

        all_key = {
            'link_id': link_id,
            'key': key_dict.key[j],
            'pos': j*2
        }
        all_key_dict.append(all_key)

    return {key['pos']: key for key in all_key_dict}, list(key_dict.key)
