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