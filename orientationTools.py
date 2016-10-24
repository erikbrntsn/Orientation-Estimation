import numpy as np
np.random.seed(0)


def load_src(name, fpath):
    import os
    import imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

mt = load_src('mathTools', 'quaternion/quaternionUtils.py')


def perpendicular(v, axis, normalize=False, vNormalized=True, axisNormalized=True):
    """ Return component of v perpendicular to axis """
    if not axisNormalized:
        axis /= np.linalg.norm(axis)
    if not vNormalized:
        v /= np.linalg.norm(v)
    vPerpendicular = v - v.dot(axis) * axis
    if normalize:
        norm = np.linalg.norm(vPerpendicular)
        if not norm == 0:
            return vPerpendicular / norm
        else:
            return vPerpendicular
    else:
        return vPerpendicular


def axisDot(vA, vB):
    vA = vA / np.linalg.norm(vA)
    vB = vB / np.linalg.norm(vB)
    axis = np.cross(vA, vB)
    axis /= np.linalg.norm(axis)
    dot = vA.dot(vB)
    return axis, dot


def axisDotSafe(vA, vB):
    vA = vA / np.linalg.norm(vA)
    vB = vB / np.linalg.norm(vB)
    axis = np.cross(vA, vB)
    axisNorm = np.linalg.norm(axis)
    if axisNorm != 0:
        axis /= axisNorm
    dot = vA.dot(vB)
    return axis, dot


def axisAngle(vA, vB):
    axis, dot = axisDot(vA, vB)
    return axis, np.arccos(dot)


def commonAxisAngle(vA, vB, uA, uB):
    # Find common axis
    dV = vA - vB
    dVNorm = np.linalg.norm(dV)
    dV /= dVNorm
    dU = uA - uB
    dUNorm = np.linalg.norm(dU)
    dU /= dUNorm
    axis = np.cross(dV, dU)
    norm = np.linalg.norm(axis)
    axis /= norm
    perpVA = perpendicular(vA, axis, True, True, True)
    perpVB = perpendicular(vB, axis, True, True, True)
    dotV = perpVA.dot(perpVB)
    # Needed due to finite numerical precision
    dotV = -1 if dotV < -1 else 1 if dotV > 1 else dotV
    return axis, dotV, norm, dVNorm, dUNorm


def axisAngleToQuaternion(axis, angle):
    halfAngle = angle / 2
    return mt.Quaternion(np.cos(halfAngle), *axis * np.sin(halfAngle))


def axisDotToQuaternion(axis, dot):
    return mt.Quaternion(np.sqrt((1 + dot) / 2), *np.sqrt((1 - dot) / 2) * axis)


def quaternionFromTwoVectorObservations(vA, vB, uA, uB):
    # Find common axis
    axis, dotV, norm, dVNorm, dUNorm = commonAxisAngle(vA, vB, uA, uB)
    perpUA = perpendicular(uA, axis, True, True, True)
    perpUB = perpendicular(uB, axis, True, True, True)
    dotU = perpUA.dot(perpUB)
    dotU = -1 if dotU < -1 else 1 if dotU > 1 else dotU
    if np.cross(uA, uB).dot(axis) < 0:
        if np.cross(vA, vB).dot(axis) < 0:
            axis *= -1
            # print("flipped")
        # else:
            # pass
            # print("u alone indicates flip is needed")
    # elif np.cross(vA, vB).dot(axis) < 0:
        # pass
        # print("v alone indicates flip is needed")
    # dot = 0.5 * (dotV + dotU)
    dot = (dotV*dVNorm + dotU*dUNorm) / (dVNorm + dUNorm)
    qual = (1 - norm) * (1 - 0.5 * abs(dotU - dotV))**2
    return axisDotToQuaternion(axis, dot), 1 - qual**2


def correctCommonAxisQuaternion(qNew, qOld):
    # q and -q represent the same rotation, but we do not want sudden sign changes - this is just a temporary workaround, we need to find a better way to handle sign flips
    if qNew.dot(qOld) < 0:
        qNew *= -1
    return qNew


def slerp(q1, q2, t):
    # # Spherical:
    # dot = q1.dot(q2)
    # dot = -1 if dot < -1 else 1 if dot > 1 else dot
    # theta = np.arccos(dot)
    # return (np.sin(t * theta) * q2 + np.sin((1 - t) * theta) * q1)/np.sin(theta)
    # Linear:
    return (1 - t) * q1 + t * q2


def eulerToQuaternion(e):
    """ Return quaternion representing rotation equivalent to rotation matrix R = R_Z(yaw) * R_Y(pitch) * R_X(roll)
    from euler angles [yaw(psi), pitch(theta), roll(phi)]. Quaternion convention: [w, i, j, k] """

    sYaw = np.sin(e[0] / 2)
    cYaw = np.cos(e[0] / 2)
    sPit = np.sin(e[1] / 2)
    cPit = np.cos(e[1] / 2)
    sRol = np.sin(e[2] / 2)
    cRol = np.cos(e[2] / 2)
    q = mt.Quaternion(cRol * cPit * cYaw + sRol * sPit * sYaw,
                      sRol * cPit * cYaw - cRol * sPit * sYaw,
                      cRol * sPit * cYaw + sRol * cPit * sYaw,
                      cRol * cPit * sYaw - sRol * sPit * cYaw)
    return q


def quaternionToAxisAngle(q):
    s = np.sqrt(1 - q[0] * q[0])
    if s == 0:
        return np.array([0, 0, 0]), 0
    return q[1:] / s, 2 * np.arccos(q[0])

# def quaternionToEuler(q):
#     """ Return euler angles (X <- Y <- Z) equivalent to quaternion q """
#     e = [math.atan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2)),
#          math.asin(2*(q[0]*q[2] - q[3]*q[1])),
#          math.atan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))]
#     return e


def quaternionToEuler(q):
    """ Return euler angles (R_Z*R_Y*R_X) equivalent to quaternion q """
    e = [np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2]**2 + q[3]**2)),
         np.arcsin(2 * (q[0] * q[2] - q[3] * q[1])),
         np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1]**2 + q[2]**2))]
    return e


# def quaternionToEuler(q):
#     """ Return euler angles (R_Z*R_Y*R_X) equivalent to quaternion q """
#     e = [np.arctan2(2*(q[1]*q[2] + q[0]*q[3]), q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3]),
#          np.arcsin(-2*(q[1]*q[3] - q[0]*q[2])),
#          np.arctan2(2*(q[2]*q[3] + q[0]*q[1]), q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3])]
#     return e


def quaternionToRotMat(q):
    """ Return rotation matrix constructed from quaternion q. """
    r = np.empty((3, 3))
    r[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    r[0, 1] = 2*(q[1]*q[2] - q[3]*q[0])
    r[0, 2] = 2*(q[1]*q[3] + q[2]*q[0])
    r[1, 0] = 2*(q[1]*q[2] + q[3]*q[0])
    r[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    r[1, 2] = 2*(q[2]*q[3] - q[1]*q[0])
    r[2, 0] = 2*(q[1]*q[3] - q[2]*q[0])
    r[2, 1] = 2*(q[2]*q[3] + q[1]*q[0])
    r[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    return r


def eulerToRotMat(e):
    """ Return rotation matrix rotating according to R = R_Z(yaw) * R_Y(pitch) * R_X(roll) from euler angles [yaw(psi), pitch(theta), roll(phi)] """
    sYaw = np.sin(e[0])
    cYaw = np.cos(e[0])
    sPit = np.sin(e[1])
    cPit = np.cos(e[1])
    sRol = np.sin(e[2])
    cRol = np.cos(e[2])
    r = np.empty((3, 3))
    r[0, 0] =  cPit * cYaw
    r[1, 0] =  cPit * sYaw
    r[2, 0] = -sPit
    r[0, 1] = -cRol * sYaw + cYaw * sRol * sPit
    r[1, 1] =  cRol * cYaw + sRol * sPit * sYaw
    r[2, 1] =  cPit * sRol
    r[0, 2] =  sRol * sYaw + cRol * cYaw * sPit
    r[1, 2] = -cYaw * sRol + cRol * sPit * sYaw
    r[2, 2] =  cRol * cPit
    return r


def mostOrthogonal(v):
    x = np.abs(v[0])
    y = np.abs(v[1])
    z = np.abs(v[2])
    rhs = (np.array([1, 0, 0]) if x < z else np.array([0, 0, 1])) if x < y else (np.array([0, 1, 0]) if y < z else np.array([0, 0, 1]))
    return np.cross(v, rhs)
