from dataclasses import dataclass
import numpy as np


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
