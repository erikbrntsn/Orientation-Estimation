import numpy as np
from rotmats import allRotMats as allRotMatsP
from rotmats_active import allRotMats as allRotMatsA


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


# https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf
# Note error in paper for 'yxz' m_31 should have been m_13
def rotMatToEuler(r, order):
    # returns (theta_3, theta_2, theta_1) as in R_ijk = R_i(theta_3) * R_j(theta_2) * R_k(theta_1)
    if order == 'xyx':
        return np.array([np.arctan2(r[0, 1], r[0, 2]),
                         np.arctan2(np.sqrt(1 - r[0, 0]**2), r[0, 0]),
                         np.arctan2(r[1, 0], -r[2, 0])])

    elif order == 'xyz':
        return np.array([np.arctan2(-r[0, 1], r[0, 0]),
                         np.arctan2(r[0, 2], np.sqrt(1 - r[0, 2]**2)),
                         np.arctan2(-r[1, 2], r[2, 2])])

    elif order == 'xzx':
        return np.array([np.arctan2(r[0, 2], -r[0, 1]),
                         np.arctan2(np.sqrt(1 - r[0, 0]**2), r[0, 0]),
                         np.arctan2(r[2, 0], r[1, 0])])

    elif order == 'xzy':
        return np.array([np.arctan2(r[0, 2], r[0, 0]),
                         np.arctan2(-r[0, 1], np.sqrt(1 - r[0, 1]**2)),
                         np.arctan2(r[2, 1], r[1, 1])])

    elif order == 'yxy':
        return np.array([np.arctan2(r[1, 0], -r[1, 2]),
                         np.arctan2(np.sqrt(1 - r[1, 1]**2), r[1, 1]),
                         np.arctan2(r[0, 1], r[2, 1])])

    elif order == 'yxz':
        return np.array([np.arctan2(r[1, 0], r[1, 1]),
                         np.arctan2(-r[1, 2], np.sqrt(1 - r[1, 2]**2)),
                         np.arctan2(r[0, 2], r[2, 2])])

    elif order == 'yzx':
        return np.array([np.arctan2(-r[1, 2], r[1, 1]),
                         np.arctan2(r[1, 0], np.sqrt(1 - r[1, 0]**2)),
                         np.arctan2(-r[2, 0], r[0, 0])])

    elif order == 'yzy':
        return np.array([np.arctan2(r[1, 2], r[1, 0]),
                         np.arctan2(np.sqrt(1 - r[1, 1]**2), r[1, 1]),
                         np.arctan2(r[2, 1], -r[0, 1])])

    elif order == 'zxy':
        return np.array([np.arctan2(-r[2, 0], r[2, 2]),
                         np.arctan2(r[2, 1], np.sqrt(1 - r[2, 1]**2)),
                         np.arctan2(-r[0, 1], r[1, 1])])

    elif order == 'zxz':
        return np.array([np.arctan2(r[2, 0], r[2, 1]),
                         np.arctan2(np.sqrt(1 - r[2, 2]**2), r[2, 2]),
                         np.arctan2(r[0, 2], -r[1, 2])])

    elif order == 'zyx':
        return np.array([np.arctan2(r[2, 1], r[2, 2]),
                         np.arctan2(-r[2, 0], np.sqrt(1 - r[2, 0]**2)),
                         np.arctan2(r[1, 0], r[0, 0])])

    elif order == 'zyz':
        return np.array([np.arctan2(r[2, 1], -r[2, 0]),
                         np.arctan2(np.sqrt(1 - r[2, 2]**2), r[2, 2]),
                         np.arctan2(r[1, 2], r[0, 2])])


def quaternionToEuler(q, order):
    r = quaternionToRotMat(q)
    return rotMatToEuler(r, order)


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


def mostOrthogonal(v):
    x = np.abs(v[0])
    y = np.abs(v[1])
    z = np.abs(v[2])
    if x < y:
        rhs = np.array([1, 0, 0]) if x < z else np.array([0, 0, 1])
    else:
        rhs = np.array([0, 1, 0]) if y < z else np.array([0, 0, 1])
    return np.cross(v, rhs)


# http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
def rotMatToQuaternion(r):
    q = mt.Quaternion(np.sqrt(max([0, 1 + r[0, 0] + r[1, 1] + r[2, 2]])) / 2,
                      np.sqrt(max([0, 1 + r[0, 0] - r[1, 1] - r[2, 2]])) / 2,
                      np.sqrt(max([0, 1 - r[0, 0] + r[1, 1] - r[2, 2]])) / 2,
                      np.sqrt(max([0, 1 - r[0, 0] - r[1, 1] + r[2, 2]])) / 2)
    q[1] = -np.abs(q[1]) if r[2, 1] - r[1, 2] < 0 else np.abs(q[1])
    q[2] = -np.abs(q[2]) if r[0, 2] - r[2, 0] < 0 else np.abs(q[2])
    q[3] = -np.abs(q[3]) if r[1, 0] - r[0, 1] < 0 else np.abs(q[3])
    return q


# http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
def rotMatToQuaternionAlt(r):
  trace = r[0][0] + r[1][1] + r[2][2]
  if trace > 0:
    s = 0.5 / np.sqrt(trace + 1)
    mt.Quaternion(0.25 / s,
                  ( r[2][1] - r[1][2] ) * s,
                  ( r[0][2] - r[2][0] ) * s,
                  ( r[1][0] - r[0][1] ) * s)
  else:
    if r[0][0] > r[1][1] and r[0][0] > r[2][2]:
      s = 2.0 * np.sqrt( 1 + r[0][0] - r[1][1] - r[2][2])
      mt.Quaternion((r[2][1] - r[1][2] ) / s,
                    0.25 * s,
                    (r[0][1] + r[1][0] ) / s,
                    (r[0][2] + r[2][0] ) / s)
    elif r[1][1] > r[2][2]:
      s = 2.0 * np.sqrt( 1 + r[1][1] - r[0][0] - r[2][2])
      mt.Quaternion((r[0][2] - r[2][0] ) / s,
                    (r[0][1] + r[1][0] ) / s,
                    0.25 * s,
                    (r[1][2] + r[2][1] ) / s)
    else:
      s = 2.0 * np.sqrt( 1 + r[2][2] - r[0][0] - r[1][1] )
      mt.Quaternion((r[1][0] - r[0][1] ) / s,
                    (r[0][2] + r[2][0] ) / s,
                    (r[1][2] + r[2][1] ) / s,
                    0.25 * s)


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


def verifyQuaternionToEuler(q2e, order):
  error = 0
  nTests = 100
  for i in range(nTests):
    q = mt.Quaternion(*np.random.rand(4) - 0.5)
    q.normalize()
    v = np.random.rand(3) - 0.5
    v /= np.linalg.norm(v)

    e = q2e(q, order)
    r = allRotMatsA[order](e)

    error += np.linalg.norm(q.rotateVector(v) - r.dot(v))
  return error / nTests
