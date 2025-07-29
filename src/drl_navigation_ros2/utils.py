from dataclasses import dataclass
import numpy as np
from builtin_interfaces.msg import Time
import math

"""计算两个时间戳之间的纳秒差（stamp2 - stamp1）"""
def get_time_diff_ns(stamp1: Time, stamp2: Time) -> int:
    # 将两个时间都转换为纳秒总数
    total_ns1 = stamp1.sec * 10**9 + stamp1.nanosec
    total_ns2 = stamp2.sec * 10**9 + stamp2.nanosec
    return  total_ns1 - total_ns2

@dataclass
class pos_data:
    name = None
    x = None
    y = None
    angle = None


def check_position(x, y, element_positions, min_dist):
    pos = True
    for element in element_positions:
        distance_vector = [element[0] - x, element[1] - y]
        distance = np.linalg.norm(distance_vector)
        if distance < min_dist:
            pos = False
    return pos


def set_random_position(name, element_positions):
    angle = np.random.uniform(-np.pi, np.pi)
    pos = False
    while not pos:
        x = np.random.uniform(-4.0, 4.0)
        y = np.random.uniform(-4.0, 4.0)
        pos = check_position(x, y, element_positions, 1.8)
    element_positions.append([x, y])
    eval_element = pos_data()
    eval_element.name = name
    eval_element.x = x
    eval_element.y = y
    eval_element.angle = angle
    return eval_element


def record_eval_positions(n_eval_scenarios=10):
    scenarios = []
    for _ in range(n_eval_scenarios):
        eval_scenario = []
        element_positions = [[-2.93, 3.17], [2.86, -3.0], [-2.77, -0.96], [2.83, 2.93]]
        for i in range(4, 8):
            name = "obstacle" + str(i + 1)
            eval_element = set_random_position(name, element_positions)
            eval_scenario.append(eval_element)

        eval_element = set_random_position("turtlebot3_waffle", element_positions)
        eval_scenario.append(eval_element)

        eval_element = set_random_position("target", element_positions)
        eval_scenario.append(eval_element)

        scenarios.append(eval_scenario)

    return scenarios


__all__ = ['wgs2gcj', 'gcj2wgs', 'gcj2wgs_exact',
           'distance', 'gcj2bd', 'bd2gcj', 'wgs2bd', 'bd2wgs']

earthR = 6378137.0

def outOfChina(lat, lng):
    return not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271)


def transform(x, y):
	xy = x * y
	absX = math.sqrt(abs(x))
	xPi = x * math.pi
	yPi = y * math.pi
	d = 20.0*math.sin(6.0*xPi) + 20.0*math.sin(2.0*xPi)

	lat = d
	lng = d

	lat += 20.0*math.sin(yPi) + 40.0*math.sin(yPi/3.0)
	lng += 20.0*math.sin(xPi) + 40.0*math.sin(xPi/3.0)

	lat += 160.0*math.sin(yPi/12.0) + 320*math.sin(yPi/30.0)
	lng += 150.0*math.sin(xPi/12.0) + 300.0*math.sin(xPi/30.0)

	lat *= 2.0 / 3.0
	lng *= 2.0 / 3.0

	lat += -100.0 + 2.0*x + 3.0*y + 0.2*y*y + 0.1*xy + 0.2*absX
	lng += 300.0 + x + 2.0*y + 0.1*x*x + 0.1*xy + 0.1*absX

	return lat, lng


def delta(lat, lng):
    ee = 0.00669342162296594323
    dLat, dLng = transform(lng-105.0, lat-35.0)
    radLat = lat / 180.0 * math.pi
    magic = math.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = math.sqrt(magic)
    dLat = (dLat * 180.0) / ((earthR * (1 - ee)) / (magic * sqrtMagic) * math.pi)
    dLng = (dLng * 180.0) / (earthR / sqrtMagic * math.cos(radLat) * math.pi)
    return dLat, dLng

# 世界大地坐标系转国测局坐标系
def wgs2gcj(wgsLat, wgsLng):
    if outOfChina(wgsLat, wgsLng):
        return wgsLat, wgsLng
    else:
        dlat, dlng = delta(wgsLat, wgsLng)
        return wgsLat + dlat, wgsLng + dlng

# 国测局坐标系转世界大地坐标系
def gcj2wgs(gcjLat, gcjLng):
    if outOfChina(gcjLat, gcjLng):
        return gcjLat, gcjLng
    else:
        dlat, dlng = delta(gcjLat, gcjLng)
        return gcjLat - dlat, gcjLng - dlng

# 更为准确的国测局坐标系转世界大地坐标系
def gcj2wgs_exact(gcjLat, gcjLng):
    initDelta = 0.01
    threshold = 0.000001
    dLat = dLng = initDelta
    mLat = gcjLat - dLat
    mLng = gcjLng - dLng
    pLat = gcjLat + dLat
    pLng = gcjLng + dLng
    for i in range(30):
        wgsLat = (mLat + pLat) / 2
        wgsLng = (mLng + pLng) / 2
        tmplat, tmplng = wgs2gcj(wgsLat, wgsLng)
        dLat = tmplat - gcjLat
        dLng = tmplng - gcjLng
        if abs(dLat) < threshold and abs(dLng) < threshold:
            return wgsLat, wgsLng
        if dLat > 0:
            pLat = wgsLat
        else:
            mLat = wgsLat
        if dLng > 0:
            pLng = wgsLng
        else:
            mLng = wgsLng
    return wgsLat, wgsLng

# 已测得计算两个国测局坐标系的点之间的距离是准的。其他坐标系之间的点的距离未测。
def distance(latA, lngA, latB, lngB): # 要求两点所属同一坐标系
    pi180 = math.pi / 180
    arcLatA = latA * pi180
    arcLatB = latB * pi180
    x = (math.cos(arcLatA) * math.cos(arcLatB) *
         math.cos((lngA - lngB) * pi180))
    y = math.sin(arcLatA) * math.sin(arcLatB)
    s = x + y
    if s > 1:
        s = 1
    if s < -1:
        s = -1
    alpha = math.acos(s)
    distance = alpha * earthR
    return distance

# 国测局坐标系转百度坐标系
def gcj2bd(gcjLat, gcjLng):
    if outOfChina(gcjLat, gcjLng):
        return gcjLat, gcjLng

    x = gcjLng
    y = gcjLat
    z = math.hypot(x, y) + 0.00002 * math.sin(y * math.pi)
    theta = math.atan2(y, x) + 0.000003 * math.cos(x * math.pi)
    bdLng = z * math.cos(theta) + 0.0065
    bdLat = z * math.sin(theta) + 0.006
    return bdLat, bdLng

# 百度坐标系转国测局坐标系
def bd2gcj(bdLat, bdLng):
    if outOfChina(bdLat, bdLng):
        return bdLat, bdLng

    x = bdLng - 0.0065
    y = bdLat - 0.006
    z = math.hypot(x, y) - 0.00002 * math.sin(y * math.pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * math.pi)
    gcjLng = z * math.cos(theta)
    gcjLat = z * math.sin(theta)
    return gcjLat, gcjLng

# 世界大地坐标系转百度坐标系
def wgs2bd(wgsLat, wgsLng):
    return gcj2bd(*wgs2gcj(wgsLat, wgsLng))


# 百度坐标系转世界大地坐标系
def bd2wgs(bdLat, bdLng):
    return gcj2wgs(*bd2gcj(bdLat, bdLng))