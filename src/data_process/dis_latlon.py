from math import radians, cos, sin, asin, acos, sqrt, atan2, fabs, degrees, ceil, copysign
from data_process import dis_euclidean as dis_e


earth_radius = 6371393


def distance_meter(p1, p2):
    """Distance between two points.

    :param p1: (Lat, Lon)
    :param p2: (Lat, Lon)
    :return: Distance in meters
    """
    lat1, lon1 = p1
    lat2, lon2 = p2
    lat3, lon3 = radians(lat1), radians(lon1)
    lat4, lon4 = radians(lat2), radians(lon2)
    dist = distance_haversine_radians(lat3, lon3, lat4, lon4)
    return round(dist, 2)


def distance_point_to_segment(p, s1, s2, delta=0.0, constrain=True):
    """Distance between point and segment.

    :param s1: Segment start point
    :param s2: Segment end point
    :param p: Point to measure distance from path to
    :param delta: Stay away from the endpoints with this factor
    :param constrain:
    :return: (Distance in meters, projected location on segment, relative location on segment)
    """
    lat1, lon1 = s1  # Start point
    lat2, lon2 = s2  # End point
    lat3, lon3 = p
    lat4, lon4 = radians(lat1), radians(lon1)
    lat5, lon5 = radians(lat2), radians(lon2)
    lat6, lon6 = radians(lat3), radians(lon3)

    dist_hs = distance_haversine_radians(lat4, lon4, lat5, lon5)
    if dist_hs == 0:
        dist_ct, pi, ti = distance_meter(p, s1), s1, 0
        return dist_ct, pi, ti

    d13 = distance_haversine_radians(lat4, lon4, lat6, lon6)
    delta13 = d13 / earth_radius
    b13 = bearing_radians(lat4, lon4, lat6, lon6)
    b12 = bearing_radians(lat4, lon4, lat5, lon5)

    dxt = asin(sin(delta13) * sin(b13 - b12))
    # b13d12 = (b13 - b12) % (2 * math.pi)
    # if b13d12 > math.pi:
    #     b13d12 = 2 * math.pi - b13d12
    dist_ct = fabs(dxt) * earth_radius
    # Correct to negative value if point is before segment
    # sgn = -1 if b13d12 > (math.pi / 2) else 1
    sgn = copysign(1, cos(b12 - b13))
    dat = sgn * acos(cos(delta13) / abs(cos(dxt))) * earth_radius
    ti = dat / dist_hs

    if not constrain:
        lat_i, loni = destination_radians(lat4, lon4, b12, dat)
    elif ti > 1.0:
        ti = 1.0
        lat_i, loni = lat5, lon5
        dist_ct = distance_haversine_radians(lat6, lon6, lat_i, loni)
    elif ti < 0.0:
        ti = 0.0
        lat_i, loni = lat4, lon4
        dist_ct = distance_haversine_radians(lat6, lon6, lat_i, loni)
    else:
        lat_i, loni = destination_radians(lat4, lon4, b12, dat)
    pi = (round(degrees(lat_i), 7), round(degrees(loni), 7))

    return dist_ct, pi, ti


def distance_segment_to_segment(f1, f2, t1, t2):
    """Distance between segments. If no intersection within range, simplified to distance from f2 to [t1,t2].

    :param f1: From
    :param f2:
    :param t1: To
    :param t2:
    :return: (distance, proj on f, proj on t, rel pos on t)
    """
    # Translate lat-lon to x-y and apply the Euclidean function
    lat_f1, lon_f1 = f1
    lat_f1, lon_f1 = radians(lat_f1), radians(lon_f1)
    f1 = 0, 0  # Origin

    lat_f2, lon_f2 = f2
    lat_f2, lon_f2 = radians(lat_f2), radians(lon_f2)
    df1f2 = distance_haversine_radians(lat_f1, lon_f1, lat_f2, lon_f2)
    bf1f2 = bearing_radians(lat_f1, lon_f1, lat_f2, lon_f2)
    # print(f"bf1f2 = {bf1f2} = {degrees(bf1f2)} degrees")
    f2 = (df1f2 * cos(bf1f2),  df1f2 * sin(bf1f2))

    lat_t1, lon_t1 = t1
    lat_t1, lon_t1 = radians(lat_t1), radians(lon_t1)
    df1t1 = distance_haversine_radians(lat_f1, lon_f1, lat_t1, lon_t1)
    bf1t1 = bearing_radians(lat_f1, lon_f1, lat_t1, lon_t1)
    # print(f"bf1t1 = {bf1t1} = {degrees(bf1t1)} degrees")
    t1 = (df1t1 * cos(bf1t1), df1t1 * sin(bf1t1))

    lat_t2, lon_t2 = t2
    lat_t2, lon_t2 = radians(lat_t2), radians(lon_t2)
    dt1t2 = distance_haversine_radians(lat_t1, lon_t1, lat_t2, lon_t2)
    # print(f"dt1t2 = {dt1t2}")
    bt1t2 = bearing_radians(lat_t1, lon_t1, lat_t2, lon_t2)
    # print(f"bt1t2 = {bt1t2} = {degrees(bt1t2)} degrees")
    t2 = (t1[0] + dt1t2 * cos(bt1t2), t1[1] + dt1t2 * sin(bt1t2))

    d, pf, pt, u_f, u_t = dis_e.distance_segment_to_segment(f1, f2, t1, t2)
    pf = destination_radians(lat_f1, lon_f1, bf1f2, u_f * df1f2)
    pf = (degrees(pf[0]), degrees(pf[1]))
    pt = destination_radians(lat_t1, lon_t1, bt1t2, u_t * dt1t2)
    pt = (degrees(pt[0]), degrees(pt[1]))

    return d, pf, pt, u_f, u_t


def xy_to_gps(x, y, ref_lat, ref_lon):

    x_rad = float(x) / earth_radius
    y_rad = float(y) / earth_radius
    c = sqrt(x_rad * x_rad + y_rad * y_rad)

    ref_lat_rad = ref_lat
    ref_lon_rad = ref_lon

    ref_sin_lat = sin(ref_lat_rad)
    ref_cos_lat = cos(ref_lat_rad)

    if abs(c) > 0:
        sin_c = sin(c)
        cos_c = cos(c)

        lat_rad = asin(cos_c * ref_sin_lat + (x_rad * sin_c * ref_cos_lat) / c)
        lon_rad = (ref_lon_rad + atan2(y_rad * sin_c, c * ref_cos_lat * cos_c - x_rad * ref_sin_lat * sin_c))

        lat = degrees(lat_rad)
        lon = degrees(lon_rad)

    else:
        lat = degrees(ref_lat)
        lon = degrees(ref_lon)

    return lat, lon


def get_intersection_point(f1, f2, t1, t2):
    # Translate lat-lon to x-y and apply the Euclidean function
    lat_f1, lon_f1 = f1
    lat_f1_1, lon_f1_1 = radians(lat_f1), radians(lon_f1)
    f1 = 0, 0  # Origin

    lat_f2, lon_f2 = f2
    lat_f2_2, lon_f2_2 = radians(lat_f2), radians(lon_f2)
    df1f2 = distance_haversine_radians(lat_f1_1, lon_f1_1, lat_f2_2, lon_f2_2)
    bf1f2 = bearing_radians(lat_f1_1, lon_f1_1, lat_f2_2, lon_f2_2)
    # print(f"bf1f2 = {bf1f2} = {degrees(bf1f2)} degrees")
    f2 = (df1f2 * cos(bf1f2), df1f2 * sin(bf1f2))

    lat_t1, lon_t1 = t1
    lat_t1_1, lon_t1_1 = radians(lat_t1), radians(lon_t1)
    df1t1 = distance_haversine_radians(lat_f1_1, lon_f1_1, lat_t1_1, lon_t1_1)
    bf1t1 = bearing_radians(lat_f1_1, lon_f1_1, lat_t1_1, lon_t1_1)
    # print(f"bf1t1 = {bf1t1} = {degrees(bf1t1)} degrees")
    t1 = (df1t1 * cos(bf1t1), df1t1 * sin(bf1t1))

    lat_t2, lon_t2 = t2
    lat_t2_2, lon_t2_2 = radians(lat_t2), radians(lon_t2)
    dt1t2 = distance_haversine_radians(lat_t1_1, lon_t1_1, lat_t2_2, lon_t2_2)
    # print(f"dt1t2 = {dt1t2}")
    bt1t2 = bearing_radians(lat_t1_1, lon_t1_1, lat_t2_2, lon_t2_2)
    # print(f"bt1t2 = {bt1t2} = {degrees(bt1t2)} degrees")
    t2 = (t1[0] + dt1t2 * cos(bt1t2), t1[1] + dt1t2 * sin(bt1t2))

    a0 = f1[1]-f2[1]
    b0 = f2[0]-f1[0]
    c0 = f1[0]*f2[1]-f2[0]*f1[1]

    a1 = t1[1]-t2[1]
    b1 = t2[0]-t1[0]
    c1 = t1[0]*t2[1]-t2[0]*t1[1]

    dd = a0*b1-a1*b0
    if dd == 0:
        return None
    x = (b0*c1 - b1*c0)/dd
    y = (a1*c0 - a0*c1)/dd

    lat, lon = xy_to_gps(x, y, lat_f1_1, lon_f1_1)
    return round(lat, 7), round(lon, 7)


def project(s1, s2, p, delta=0.0):
    _, pi, ti = distance_point_to_segment(p, s1, s2, delta)
    return pi, ti


def box_around_point(p, dist):
    lat, lon = p
    lat_r, lon_r = radians(lat), radians(lon)
    # dia_dist = sqrt(2 * dist ** 2)
    dia_dist = dist * sqrt(2)
    lat_t, lon_t = destination_radians(lat_r, lon_r, radians(45), dia_dist)
    lat_b, lon_l = destination_radians(lat_r, lon_r, radians(225), dia_dist)
    lat_nt, lon_nt = degrees(lat_t), degrees(lon_t)
    lat_nb, lon_nl = degrees(lat_b), degrees(lon_l)
    return lat_nb, lon_nl, lat_nt, lon_nt


def interpolate_path(path, dd):
    """
    :param path: (lat, lon)
    :param dd: Distance difference (meter)
    :return:
    """
    path_new = [path[0]]
    for p1, p2 in zip(path, path[1:]):
        lat1, lon1 = p1[0], p1[1]
        lat2, lon2 = p2[0], p2[1]
        lat1, lon1 = radians(lat1), radians(lon1)
        lat2, lon2 = radians(lat2), radians(lon2)
        dist = distance_haversine_radians(lat1, lon1, lat2, lon2)
        if dist > dd:
            dt = int(ceil(dist / dd))
            dist_d = dist/dt
            dist_i = 0
            bearing = bearing_radians(lat1, lon1, lat2, lon2)
            for _ in range(dt):
                dist_i += dist_d
                lat_i, loni = destination_radians(lat1, lon1, bearing, dist_i)
                path_new.append((degrees(lat_i), degrees(loni)))
        path_new.append(p2)
    return path_new


def bearing_radians(lat1, lon1, lat2, lon2):
    """Initial bearing"""
    d_lon = lon2 - lon1
    y = sin(d_lon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(d_lon)
    return atan2(y, x)


def get_bear(p1, p2):
    lat1, lon1 = p1
    lat2, lon2 = p2
    lat11, lon11 = radians(lat1), radians(lon1)
    lat22, lon22 = radians(lat2), radians(lon2)

    bear = bearing_radians(lat11, lon11, lat22, lon22)

    degree1 = degrees(bear)
    degree = (degree1 + 360) % 360
    return round(degree, 7)


def distance_haversine_radians(lat1, lon1, lat2, lon2, radius=earth_radius):
    # type (float, float, float, float, float) -> float
    lat = lat2 - lat1
    lon = lon2 - lon1
    a = sin(lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(lon / 2) ** 2
    # dist = 2 * radius * asin(sqrt(a))
    dist = 2 * radius * atan2(sqrt(a), sqrt(1 - a))
    return dist


def destination_radians(lat1, lon1, bearing, dist):
    d = dist / earth_radius
    lat2 = asin(sin(lat1) * cos(d) + cos(lat1) * sin(d) * cos(bearing))
    lon2 = lon1 + atan2(sin(bearing) * sin(d) * cos(lat1), cos(d) - sin(lat1) * sin(lat2))
    return lat2, lon2


def destination(p, bearing, dist):
    lat1, lon1 = p
    lat11, lon11 = radians(lat1), radians(lon1)
    bear = radians(bearing)

    lat, lon = destination_radians(lat11, lon11, bear, dist)
    return round(degrees(lat), 7), round(degrees(lon), 7)


def lines_parallel(f1, f2, t1, t2, d=None):
    lat_f1, lon_f1 = f1
    lat_f1, lon_f1 = radians(lat_f1), radians(lon_f1)
    f1 = 0, 0  # Origin

    lat_f2, lon_f2 = f2
    lat_f2, lon_f2 = radians(lat_f2), radians(lon_f2)
    df1f2 = distance_haversine_radians(lat_f1, lon_f1, lat_f2, lon_f2)
    bf1f2 = bearing_radians(lat_f1, lon_f1, lat_f2, lon_f2)
    # print(f"bf1f2 = {bf1f2} = {degrees(bf1f2)} degrees")
    f2 = (df1f2 * cos(bf1f2), df1f2 * sin(bf1f2))

    lat_t1, lon_t1 = t1
    lat_t1, lon_t1 = radians(lat_t1), radians(lon_t1)
    df1t1 = distance_haversine_radians(lat_f1, lon_f1, lat_t1, lon_t1)
    bf1t1 = bearing_radians(lat_f1, lon_f1, lat_t1, lon_t1)
    # print(f"bf1t1 = {bf1t1} = {degrees(bf1t1)} degrees")
    t1 = (df1t1 * cos(bf1t1), df1t1 * sin(bf1t1))

    lat_t2, lon_t2 = t2
    lat_t2, lon_t2 = radians(lat_t2), radians(lon_t2)
    dt1t2 = distance_haversine_radians(lat_t1, lon_t1, lat_t2, lon_t2)
    # print(f"dt1t2 = {dt1t2}")
    bt1t2 = bearing_radians(lat_t1, lon_t1, lat_t2, lon_t2)
    # print(f"bt1t2 = {bt1t2} = {degrees(bt1t2)} degrees")
    t2 = (t1[0] + dt1t2 * cos(bt1t2), t1[1] + dt1t2 * sin(bt1t2))

    return dis_e.lines_parallel(f1, f2, t1, t2, d=d)
