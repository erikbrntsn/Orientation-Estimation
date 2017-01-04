import numpy as np
from fromVectorObservations import davenportsQMethod
from orientationTools import correctQuaFromVecObs, slerp


def load_src(name, fpath):
    import os
    import imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))


mt = load_src('mathTools', 'quaternion/quaternion.py')


class CommonAxisEstimator(object):
    def __init__(self,
                 beta=0.005,
                 zeta=0.000005,
                 q=mt.Quaternion()):
        # Current estimate of orientation given as a quaternion
        self.q = q
        # Gyroscope bias estiamte
        self.gyrBias = mt.Quaternion()
        # Weighting between quaternion estimated from vector observations and the quaternion
        # derivative obtained from the gyroscope
        self.beta = beta
        # Low pass filter weight: [0, 1], [no change, full change]
        self.zeta = zeta

    def update(self, gyr, qVec dt):
        gyrQ = mt.Quaternion(0, *gyr)

        # Bias estimation
        dqv = (qVec - self.q) / dt
        dq_ = 0.5 * self.q * gyrQ
        bias = 2 * self.q.conjugate() * (dq_ - dqv)
        self.gyrBias += self.zeta * (bias - self.gyrBias)

        # Quaternion derivative estimate
        dq = 0.5 * self.q * (gyrQ - self.gyrBias)

        # Weighted average of derivative integration and absolute orientation estimate - Given by
        # the formula for quaternion slerp - https://en.wikipedia.org/wiki/slerp (and the wiki's
        # external links)
        qIntegrated = self.q + dq * dt
        qIntegrated.normalize()
        # If the two quaternions are the same, then just use one instead of weighting them
        if qVec.dot(qIntegrated) > 0.9999:
            self.q = qVec
        else:
            self.q = slerp(qIntegrated, qVec, self.beta)
            self.q.normalize()


class MahonyEstimator(object):

    def __init__(self,
                 beta=100,
                 zeta=1000,
                 q=mt.Quaternion()):
        # Current estimate of orientation given as a quaternion
        self.q = q
        # Gyroscope bias estimate
        self.gyrBias = np.array([0.0, 0.0, 0.0])
        # Weight applied to vector orientation estimate
        self.beta = beta
        # Weight used in gyro bias estimate
        self.zeta = zeta

    def update(self, gyr, qVec, dt):
        # Estimate quaternion error
        qTilde = self.q.conjugate() * qVec
        # Field quaternion correction
        sv = qTilde[0] * qTilde[1:]
        # Gyrobias estimation
        self.gyrBias -= self.zeta * sv * dt
        # Corrected rate of change of quaternion
        dq = 0.5 * self.q * mt.Quaternion(0, *gyr - self.gyrBias + self.beta * sv)
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
        # Does not assume anything about the vector part of v. Possibilities of optimization are
        # thus present.
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

        dQOmega = 0.5 * self.q * (gyr - self.gyrBias)

        # Update quaternion estiamte
        self.q += (dQOmega - self.beta * grad) * dt
        self.q.normalize()
