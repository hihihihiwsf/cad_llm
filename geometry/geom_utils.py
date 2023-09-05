import math
import numpy as np

def angle_from_vector_to_x(vec):
    assert vec.size == 2
    # We need to find a unit vector
    angle = 0.0

    l = np.linalg.norm(vec)
    uvec = vec/l

    # 2 | 1
    #-------
    # 3 | 4
    if uvec[0] >=0:
        if uvec[1] >= 0:
            # Qadrant 1
            angle = math.asin(uvec[1])
        else:
            # Qadrant 4
            angle = 2.0*math.pi - math.asin(-uvec[1])
    else:
        if vec[1] >= 0:
            # Qadrant 2
            angle = math.pi - math.asin(uvec[1])
        else:
            # Qadrant 3
            angle = math.pi + math.asin(-uvec[1])
    return angle


def convert_angle_to_1to360_range(angle_rad):
    """
    Converts the given angle in radians into 1-360 degrees range
    """
    angle = math.degrees(angle_rad)
    # Lifted from: https://stackoverflow.com/questions/12234574/calculating-if-an-angle-is-between-two-angles
    angle=(int(angle) % 360) + (angle-math.trunc(angle)) # converts angle to range -360 + 360
    if angle > 0.0:
        return angle
    else:
        return angle + 360.0


def angle_is_between(angle_rad, a_rad, b_rad):
    """
    Checks if angle is in between the range of a and b
    (All angles must be given in radians)
    """
    angle = convert_angle_to_1to360_range(angle_rad)
    a = convert_angle_to_1to360_range(a_rad)
    b = convert_angle_to_1to360_range(b_rad)
    if a < b:
        return a <= angle and angle <= b
    return a <= angle or angle <= b


def rads_to_degs(rads):
    return 180*rads/math.pi