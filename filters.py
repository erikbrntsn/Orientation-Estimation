import numpy as np


def load_src(name, fpath):
    import os
    import imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))


mt = load_src('mathTools', 'quaternion/quaternion.py')
ot = load_src('ot', 'orientation_tools.py')


class SlerpEstimator(object):
    def __init__(self, beta, zeta, q=mt.Quaternion()):
        # Current estimate of orientation given as a quaternion
        self.q = q.copy()
        self.beta = beta
        self.zeta = zeta
        self.gyrBias = np.zeros(3)

    def update(self, gyr, qExt, dt):
        # Quaternion derivative estimate
        dqC = 0.5 * self.q * mt.Quaternion(0, *gyr - self.gyrBias)
        qIntegratedC = self.q + dqC * dt
        qIntegratedC.normalize()

        # Bias estimation
        dq = 0.5 * self.q * mt.Quaternion(0, *gyr)
        qIntegrated = self.q + dq * dt
        qIntegrated.normalize()
        self.gyrBias += (((2 / dt) * self.q.conjugate() * (qIntegrated - qExt)).vectorPart() - self.gyrBias) * self.zeta

        # Weighted average of derivative integration and absolute orientation estimate - Given by
        # the formula for quaternion slerp - https://en.wikipedia.org/wiki/slerp (and the wiki's
        # external links)
        self.q = ot.slerp(qIntegratedC, qExt, self.beta)
        self.q.normalize()


class CommonAxisEstimator(object):
    def __init__(self, beta, zeta, q=mt.Quaternion()):
        # Current estimate of orientation given as a quaternion
        self.q = q.copy()
        # Gyroscope bias estiamte
        self.gyrBias = mt.Quaternion(0.0, 0.0, 0.0, 0.0)
        # Weighting between quaternion estimated from vector observations and the quaternion
        # derivative obtained from the gyroscope
        self.beta = beta
        # Low pass filter weight: [0, 1], [no change, full change]
        self.zeta = zeta

    def update(self, gyr, qExt, dt):
        gyrQ = mt.Quaternion(0, *gyr)

        # Bias estimation
        dqv = (qExt - self.q) / dt
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

        self.q = ot.slerp(qIntegrated, qExt, self.beta)
        self.q.normalize()


class MahonyEstimator(object):

    def __init__(self, beta, zeta, q=mt.Quaternion()):
        # Current estimate of orientation given as a quaternion
        self.q = q.copy()
        # Gyroscope bias estimate
        self.gyrBias = np.array([0.0, 0.0, 0.0])
        # Weight applied to vector orientation estimate
        self.beta = beta
        # Weight used in gyro bias estimate
        self.zeta = zeta

    def update(self, gyr, qExt, dt):
        # Estimate quaternion error
        qErr = self.q.conjugate() * qExt
        # Field quaternion correction
        sv = qErr[0] * qErr[1:]
        # Corrected rate of change of quaternion
        dq = 0.5 * self.q * mt.Quaternion(0, *gyr - self.gyrBias + self.beta * sv)
        # Gyrobias estimation
        self.gyrBias -= self.zeta * sv * dt
        self.q += dq * dt
        self.q.normalize()


class MadgwickEstimator(object):
    def __init__(self, beta, zeta, q=mt.Quaternion()):
        self.beta = beta
        self.zeta = zeta
        self.q = q.copy()
        self.g = mt.Quaternion(0.0, 0.0, 0.0, 1.0)
        self.gyrBias = np.zeros(3)

    # d\dq(R(q)*[0, 0, 1]^T) : q=SqE
    def derivativeG(self):
          return np.array([[2*self.q[2], -2*self.q[1],  2*self.q[0]],
                           [2*self.q[3], -2*self.q[0], -2*self.q[1]],
                           [2*self.q[0],  2*self.q[3], -2*self.q[2]],
                           [2*self.q[1],  2*self.q[2],  2*self.q[3]]])

    # d\dq(R(q)*[bx, 0, bz]^T) : q=SqE
    def derivativeB(self, b):
          return np.array([[ 2*b[1]*self.q[0] + 2*b[3]*self.q[2],  2*b[1]*self.q[3] - 2*b[3]*self.q[1], -2*b[1]*self.q[2] + 2*b[3]*self.q[0]],
                           [ 2*b[1]*self.q[1] + 2*b[3]*self.q[3],  2*b[1]*self.q[2] - 2*b[3]*self.q[0],  2*b[1]*self.q[3] - 2*b[3]*self.q[1]],
                           [-2*b[1]*self.q[2] + 2*b[3]*self.q[0],  2*b[1]*self.q[1] + 2*b[3]*self.q[3], -2*b[1]*self.q[0] - 2*b[3]*self.q[2]],
                           [-2*b[1]*self.q[3] + 2*b[3]*self.q[1],  2*b[1]*self.q[0] + 2*b[3]*self.q[2],  2*b[1]*self.q[1] + 2*b[3]*self.q[3]]])

     # d\dq(R(q)*r) : q=SqE
    def derivativeGeneral(self, r):
          return np.array([[ 2*r[1]*self.q[0] - 2*r[2]*self.q[3] + 2*r[3]*self.q[2],  2*r[1]*self.q[3] + 2*r[2]*self.q[0] - 2*r[3]*self.q[1], -2*r[1]*self.q[2] + 2*r[2]*self.q[1] + 2*r[3]*self.q[0]],
                           [ 2*r[1]*self.q[1] + 2*r[2]*self.q[2] + 2*r[3]*self.q[3],  2*r[1]*self.q[2] - 2*r[2]*self.q[1] - 2*r[3]*self.q[0],  2*r[1]*self.q[3] + 2*r[2]*self.q[0] - 2*r[3]*self.q[1]],
                           [-2*r[1]*self.q[2] + 2*r[2]*self.q[1] + 2*r[3]*self.q[0],  2*r[1]*self.q[1] + 2*r[2]*self.q[2] + 2*r[3]*self.q[3], -2*r[1]*self.q[0] + 2*r[2]*self.q[3] - 2*r[3]*self.q[2]],
                           [-2*r[1]*self.q[3] - 2*r[2]*self.q[0] + 2*r[3]*self.q[1],  2*r[1]*self.q[0] - 2*r[2]*self.q[3] + 2*r[3]*self.q[2],  2*r[1]*self.q[1] + 2*r[2]*self.q[2] + 2*r[3]*self.q[3]]])

    def update(self, acc, gyr, mag, dt):
        accQ = mt.Quaternion(0, *acc)
        magQ = mt.Quaternion(0, *mag)

        # Update B-field
        # Rotate measurement into earth frame
        h = self.q.conjugate() * magQ * self.q
        self.b = mt.Quaternion(0, np.sqrt(h[1]**2 + h[2]**2), 0, h[3])

        # Calculate objective function
        fg = (self.q * self.g * self.q.conjugate() - accQ).vectorPart()
        fb = (self.q * self.b * self.q.conjugate() - magQ).vectorPart()

        # Calculate Jacobian
        jg = self.derivativeG()
        jb = self.derivativeB(self.b)

        # Calculate gradient
        grad = mt.Quaternion(*(jg.dot(fg) + jb.dot(fb)))
        grad.normalize()

        # Gyro bias estimation
        gyrError = 2 * self.q.conjugate() * grad
        self.gyrBias += self.zeta * gyrError.vectorPart() * dt

        # Update quaternion estiamte
        dQOmega = 0.5 * self.q * mt.Quaternion(0, *gyr - self.gyrBias)
        self.q += (dQOmega - self.beta * grad) * dt
        self.q.normalize()
