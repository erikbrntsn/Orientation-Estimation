import numpy as np
np.random.seed(0)


def load_src(name, fpath):
    import os
    import imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

mt = load_src('mathTools', '../mathTools/quaternion/quaternionUtils.py')


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
        # q and -q represent the same rotation, but we do not want sudden sign changes - this is just a temporary workaround, we need to find a better way to avoid sign flips
    if qNew.dot(qOld) < 0:
        qNew *= -1
    return qNew


def slerp(q1, q2, t):
    # dot = q1.dot(q2)
    # dot = -1 if dot < -1 else 1 if dot > 1 else dot
    # theta = np.arccos(dot)
    # return (np.sin(t * theta) * q2 + np.sin((1 - t) * theta) * q1)/np.sin(theta)
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


def davenportsQMethod(observations, reference, weights=None):
    if weights is None:
        weights = np.ones(observations.shape[1])
    b = np.zeros((3, 3))
    for i in range(weights.shape[0]):
        # Note: b is transposed here compared to the article by FC_Markley
        b += weights[i] * np.outer(reference[:, i], observations[:, i])
    z = np.array([b[1, 2] - b[2, 1],
                  b[2, 0] - b[0, 2],
                  b[0, 1] - b[1, 0]])
    s = b + b.T
    k = np.empty((4, 4))
    traceB = np.trace(b)
    k[0:-1, 0:-1] = s - np.eye(3)*traceB
    k[0:-1, -1] = z
    k[-1, 0:-1] = z
    k[-1, -1] = traceB
    eigenValues, eigenVectors = np.linalg.eig(k)
    maxEig = eigenVectors[:, eigenValues.argmax()]
    q = mt.Quaternion(maxEig[3], *maxEig[:3])
    q.normalize()
    # q = q.conjugate()
    return q


class CommonAxisEstimator(object):
    def __init__(self,
                 beta=0.005,
                 zeta=0.000005,
                 q=mt.Quaternion(),
                 mA=np.array([1.0, 2.0, 3.0] / np.sqrt(14)),
                 gA=np.array([0.0, 0.0, 1.0])):
        # Current estimate of orientation given as a quaternion
        self.q = q
        # Weighting between quaternion estimated from vector observations and the quaternion derivative obtained from the gyroscope
        self.beta = beta
        # Gravity in the Earth frame
        self.gA = gA
        # Earth's magnetic field in the Earth frame
        self.mA = mA
        # Gyroscope bias estiamte
        self.gyrBias = mt.Quaternion()
        # Low pass filter weight: [0, 1], [no change, full change]
        self.zeta = zeta
        # self.qDavenport = mt.Quaternion()

    def update(self, acc, gyr, mag, dt):
        acc /= np.linalg.norm(acc)
        mag /= np.linalg.norm(mag)
        gyrQ = mt.Quaternion(0, *gyr)

        # Calculate absolute quaternion that minimizes Whaba's problem
        qDavenport = davenportsQMethod(np.hstack((acc[:, None], mag[:, None])),
                                       np.hstack((self.gA[:, None], self.mA[:, None])))

        self.qDavenport = correctCommonAxisQuaternion(qDavenport, self.q)
        qAbsolute = self.qDavenport
        qual = 1

        # Bias estimation
        dqv = (qAbsolute - self.q) / dt
        dq_ = 0.5 * self.q * gyrQ
        bias = 2 * self.q.conjugate() * (dq_ - dqv)
        self.gyrBias += qual**2 * self.zeta * (bias - self.gyrBias)

        # Quaternion derivative estimate
        dq = 0.5 * self.q * (gyrQ - self.gyrBias)

        # Weighted average of derivative integration and absolute orientation estimate - Given by the formula for quaternion slerp - https://en.wikipedia.org/wiki/slerp (and the wiki's external links)
        print(self.q + dq)
        qIntegrated = self.q + dq * dt
        qIntegrated.normalize()
        # If the two quaternions are the same, then just use one instead of weighting them
        if qAbsolute.dot(qIntegrated) > 0.9999:
            self.q = qAbsolute
        else:
            self.q = slerp(qIntegrated, qAbsolute, self.beta * qual**2)
            self.q.normalize()


class MahonyEstimator(object):

    def __init__(self,
                 kEst=100,
                 kB=1000,
                 q=mt.Quaternion(),
                 mA=np.array([1.0, 2.0, 3.0]/np.sqrt(14)),
                 gA=np.array([0.0, 0.0, 1.0])):
        # Current estimate of orientation given as a quaternion
        self.q = q
        # Gravity in the Earth frame
        self.gA = gA
        # Earth's magnetic field in the Earth frame
        self.mA = mA
        # Weight applied to vector orientation estimate
        self.kEst = kEst
        # Weight used in gyro bias estimate
        self.kB = kB
        # Gyroscope bias estimate
        self.gyrBias = np.array([0.0, 0.0, 0.0])

    def update(self, acc, gyr, mag, dt):
        # Prepare measurements
        acc /= np.linalg.norm(acc)
        mag /= np.linalg.norm(mag)

        # Calculate orientation using the two field observations
        self.qAbsolute = davenportsQMethod(np.hstack((acc[:, None], mag[:, None])),
                                           np.hstack((self.gA[:, None], self.mA[:, None])))

        # Estimate quaternion error
        qTilde = self.q.conjugate() * self.qAbsolute
        # Field quaternion correction
        sv = qTilde[0] * qTilde[1:]

        # Gyrobias estimation
        self.gyrBias -= self.kB * sv * dt

        # Corrected rate of change of quaternion
        dq = 0.5 * self.q * mt.Quaternion(0, *gyr - self.gyrBias + self.kEst * sv)
        self.q += dq * dt
        self.q.normalize()


class MadgwickMagReadable(object):
    def __init__(self, beta, zeta):
        self.beta = beta
        self.zeta = zeta
        self.q = mt.Quaternion()
        self.g = mt.Quaternion(0.0, 0.0, 0.0, 1.0)
        self.gyrBias = mt.Quaternion(0.0, 0.0, 0.0, 0.0)

    def derivative(self, q, v):
        # Does not assume anything about the vector part of v. Possibilities of optimization are thus present.
        return np.array([[ 2*v[2]*q[3] - 2*v[3]*q[2],   2*v[2]*q[2] + 2*v[3]*q[3],               -4*v[1]*q[2] + 2*v[2]*q[1] - 2*v[3]*q[0],  -4*v[1]*q[3] + 2*v[2]*q[0] + 2*v[3]*q[1]],
                         [-2*v[1]*q[3] + 2*v[3]*q[1],   2*v[1]*q[2] - 4*v[2]*q[1] + 2*v[3]*q[0],  2*v[1]*q[1] + 2*v[3]*q[3],                -2*v[1]*q[0] - 4*v[2]*q[3] + 2*v[3]*q[2]],
                         [ 2*v[1]*q[2] - 2*v[2]*q[1],   2*v[1]*q[3] - 2*v[2]*q[0] - 4*v[3]*q[1],  2*v[1]*q[0] + 2*v[2]*q[3] - 4*v[3]*q[2],   2*v[1]*q[1] + 2*v[2]*q[2]]])

    def update(self, acc, gyr, mag, dt):
        # Prepare data
        acc = mt.Quaternion(0, *acc / np.linalg.norm(acc))
        mag = mt.Quaternion(0, *mag / np.linalg.norm(mag))
        gyr = mt.Quaternion(0, *gyr)

        # Update B-field
        # Rotate measurement into earth frame
        h = self.q * mag * self.q.conjugate()
        self.b = mt.Quaternion(0, np.sqrt(h[1]**2 + h[2]**2), 0, h[3])

        # Calculate objective function
        fg = (self.q.conjugate() * self.g * self.q - acc)[1:]
        fb = (self.q.conjugate() * self.b * self.q - mag)[1:]

        # Calculate Jacobian
        jg = self.derivative(self.q, self.g)
        jb = self.derivative(self.q, self.b)

        # Calculate gradient
        grad = mt.Quaternion(*(jg.T.dot(fg) + jb.T.dot(fb)))
        grad.normalize()

        # Gyro bias estimation
        gyrError = 2 * self.q.conjugate() * grad
        self.gyrBias += self.zeta * gyrError * dt

        dq_omega = 0.5 * self.q * (gyr - self.gyrBias)

        # Update quaternion estiamte
        self.q += (dq_omega - self.beta * grad) * dt
        self.q.normalize()
