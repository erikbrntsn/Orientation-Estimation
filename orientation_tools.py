import numpy as np


def load_src(name, fpath):
    import os
    import imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))


mt = load_src('mathTools', 'quaternion/quaternion.py')


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
    # dVNorm = np.linalg.norm(dV)
    # dV /= dVNorm
    dU = uA - uB
    # dUNorm = np.linalg.norm(dU)
    # dU /= dUNorm
    axis = np.cross(dV, dU)
    norm = np.linalg.norm(axis)
    axis /= norm
    perpVA = perpendicular(vA, axis, True, True, True)
    perpVB = perpendicular(vB, axis, True, True, True)
    dotV = perpVA.dot(perpVB)
    # Needed due to finite numerical precision
    dotV = -1 if dotV < -1 else 1 if dotV > 1 else dotV
    # return axis, dotV, norm, dVNorm, dUNorm
    return axis, dotV


def axisAngleToQuaternion(axis, angle):
    halfAngle = angle / 2
    return mt.Quaternion(np.cos(halfAngle), *axis * np.sin(halfAngle))


def axisDotToQuaternion(axis, dot):
    return mt.Quaternion(np.sqrt((1 + dot) / 2), *np.sqrt((1 - dot) / 2) * axis)


def quaternionFromTwoVectorObservations(vA, vB, uA, uB):
    # Find common axis
    # axis, dotV, norm, dVNorm, dUNorm = commonAxisAngle(vA, vB, uA, uB)
    axis, dotV = commonAxisAngle(vA, vB, uA, uB)
    perpUA = perpendicular(uA, axis, True, True, True)
    perpUB = perpendicular(uB, axis, True, True, True)
    dotU = perpUA.dot(perpUB)
    dotU = -1 if dotU < -1 else 1 if dotU > 1 else dotU
    # if np.cross(uA, uB).dot(axis) < 0:
    #     if np.cross(vA, vB).dot(axis) < 0:
    #         axis *= -1
            # print("flipped")
        # else:
            # pass
            # print("u alone indicates flip is needed")
    # elif np.cross(vA, vB).dot(axis) < 0:
        # pass
        # print("v alone indicates flip is needed")
    # dot = 0.5 * (dotV + dotU)
    # dot = (dotV*dVNorm + dotU*dUNorm) / (dVNorm + dUNorm)
    dot = (dotU + dotV) / 2
    # qual = (1 - norm) * (1 - 0.5 * abs(dotU - dotV))**2
    # return axisDotToQuaternion(axis, dot), 1 - qual**2
    return axisDotToQuaternion(axis, dot)


def correctCommonAxisQuaternion(qNew, qOld):
    # q and -q represent the same rotation, but sudden sign changes can be problematic - this can
    # work as a temporary workaround. A better way to handle sign flips would be nice
    if qNew.dot(qOld) < 0:
        qNew *= -1
    return qNew


def slerp(q1, q2, t):
    # Linear Spherical Interpolation from q1 to q2 parameterized t
    # Spherical:
    dot = q1.dot(q2)
    if np.abs(dot) >= 1:
        return q1
    else:
        theta = np.arccos(dot)
        return (np.sin(t * theta) * q2 + np.sin((1 - t) * theta) * q1)/np.sin(theta)
    # Linear:
    # return (1 - t) * q1 + t * q2


def eulerToQuaternion(e):
    """ Return quaternion representing rotation equivalent to rotation matrix
    # R = R_Z(yaw) * R_Y(pitch) * R_X(roll) from euler angles [yaw(psi), pitch(theta), roll(phi)].
    # Quaternion convention: [w, i, j, k] """
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
#     """ Return euler angles (R_X*R_Y*R_Z) equivalent to quaternion q """
#     e = [np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2)),
#          np.arcsin( 2*(q[0]*q[2] - q[3]*q[1])),
#          np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))]
#     return e


def quaternionToEuler(q):
    """ Return euler angles (R_Z*R_Y*R_X) equivalent to quaternion q """
    e = [np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2]**2 + q[3]**2)),
         np.arcsin( 2 * (q[0] * q[2] - q[3] * q[1])),
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
    """ Return rotation matrix rotating according to R = R_Z(yaw) * R_Y(pitch) * R_X(roll) from
    euler angles [yaw(psi), pitch(theta), roll(phi)] """
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
    if x < y:
        rhs = np.array([1, 0, 0]) if x < z else np.array([0, 0, 1])
    else:
        rhs = np.array([0, 1, 0]) if y < z else np.array([0, 0, 1])
    return np.cross(v, rhs)


def dcm2Qua(rot):
    t = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if t > -0.99:
        r = np.sqrt(1 + t)
        s = 0.5 / r
        q = mt.Quaternion(0.5 * r,
                          (rot[2, 1] - rot[1, 2]) * s,
                          (rot[0, 2] - rot[2, 0]) * s,
                          (rot[1, 0] - rot[0, 1]) * s)
    elif rot[1, 1] < rot[0, 0] and rot[2, 2] < rot[0, 0]:
        t = rot[0, 0] + rot[1, 1] + rot[2, 2]
        r = np.sqrt(1 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        s = 0.5 / r
        mt.Quaternion((rot[2, 1] - rot[1, 2]) * s,
                      0.5 * r,
                      (rot[0, 1] + rot[1, 0]) * s,
                      (rot[2, 0] + rot[0, 2]) * s)
    elif rot[0, 0] < rot[1, 1] and rot[2, 2] < rot[1, 1]:
        raise NotImplemented
    else:
        raise NotImplemented
    return q / q.norm()

# def shustersRotation(vA, vB, uA, uB):


def eulerToQuaternionGeneral(euler, seq):
    # Assuming euler = (theta_3, theta_2, theta_1)
    # seq = abc, a,b,c, \in {x, y, z}. R = R_a(theta_3) * R_b(theta_2) * R_c(theta_1)
    # Example seq = 'xyx' => R = R_x(theta_3) * R_y(theta_2) * R_x(theta_1)
    q = 1
    for i in range(3):
        if seq[i] == 'x':
            q *= mt.Quaternion(np.cos(euler[i]/2), -np.sin(euler[i]/2), 0, 0)
        elif seq[i] == 'y':
            q *= mt.Quaternion(np.cos(euler[i]/2), 0, -np.sin(euler[i]/2), 0)
        elif seq[i] == 'z':
            q *= mt.Quaternion(np.cos(euler[i]/2), 0, 0, -np.sin(euler[i]/2))
    return q


def eulerToRotMatGeneral(euler, seq):
  # Assuming euler = (theta_3, theta_2, theta_1) and
  # seq = abc, (a,b,c, \in {x, y, z}). R = R_a(theta_3) * R_b(theta_2) * R_c(theta_1)
  # Example seq = 'xyx' => R = R_x(theta_3) * R_y(theta_2) * R_x(theta_1)
  r = 1
  for i in range(3):
    c = np.cos(euler[i])
    s = np.sin(euler[i])
    if seq[i] == 'x':
      r = np.dot(r, np.array([[ 1, 0, 0],
                              [ 0, c, s],
                              [ 0,-s, c]]))

    elif seq[i] == 'y':
      r = np.dot(r, np.array([[ c, 0,-s],
                              [ 0, 1, 0],
                              [ s, 0, c]]))

    elif seq[i] == 'z':
      r = np.dot(r, np.array([[ c, s, 0],
                              [-s, c, 0],
                              [ 0, 0, 1]]))
  return r
