# main articles describing the functionality implemented here:
# http://mortari.tamu.edu/Attitude-Estimation/J06.pdf
# http://mortari.tamu.edu/attitude-estimation/j03.pdf
# http://www.malcolmdshuster.com/Pub_2007a_C_cquest_MDS.pdf

# Note that the quaternion is implemented as q = (a, b*i, c*j, d*k)
# instead of convention used in Marley's article q = (b*i, c*j, d*k, a)

import numpy as np
from scipy import linalg
import orientation_tools as ot

# np.random.seed(2)


def load_src(name, fpath):
    import os
    import imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))


mt = load_src('mathTools', 'quaternion/quaternion.py')


def adj3x3(a):
    adj = np.empty((3, 3))
    for i in range(3):
        adj[:, i] = np.cross(a[i - 2, :], a[i - 1, :])
    return adj


def traceAdj3x3(a):
    return np.sum(np.array([a[1, 1] * a[2, 2] - a[2, 1] * a[1, 2] +
                            a[0, 0] * a[2, 2] - a[2, 0] * a[0, 2] +
                            a[0, 0] * a[1, 1] - a[1, 0] * a[0, 1]]))


def constructB(ref, obs, wei=None):
    if wei is None:
        wei = np.ones(obs.shape[1])
    b = np.zeros((3, 3))
    for i in range(wei.shape[0]):
        b += wei[i] * np.outer(obs[:, i], ref[:, i])
    return b


def lossFunction(rot, ref, obs, wei=None):
    if wei is None:
        wei = np.ones(obs.shape[1])[None, :]
    if isinstance(rot, mt.Quaternion):
        return wei.dot(np.sum((obs - rot.rotateVectors(ref))**2, axis=0))[0]
    else:
        return wei.dot(np.sum((obs - rot.dot(ref))**2, axis=0))[0]


def firstSolution(B):
    w, h = linalg.polar(B)
    detW = linalg.det(w)
    if detW > 0:
        # print("detW was positive")
        return w
    else:
        # print("detW was not positive")
        eigVal, eigVec = linalg.eig(h)
        indeces = np.argsort(eigVal)
        v = np.array([eigVec[:, indeces[2]],
                      eigVec[:, indeces[1]],
                      eigVec[:, indeces[0]]]).T
        return w.dot(v.dot(np.diag(np.array([1, 1, detW])).dot(v.T)))


def alternateSolution(b):
    return b.dot(linalg.inv(linalg.sqrtm(b.T.dot(b))))


def unconstrainedLeastSquares(b, ref, wei=None):
    if wei is None:
        wei = np.ones(ref.shape[1])[:, None]
    return b.dot(linalg.inv(constructB(ref, ref, wei)))


def helper(b):
    z = np.array([b[2, 1] - b[1, 2],
                  b[0, 2] - b[2, 0],
                  b[1, 0] - b[0, 1]])
    s = b + b.T
    traceB = np.trace(b)
    k = np.empty((4, 4))
    k[0:-1, 0:-1] = s - np.eye(3) * traceB
    k[0:-1, -1] = z
    k[-1, 0:-1] = z
    k[-1, -1] = traceB
    return k, s, traceB, z


def helper2(b):
    z = -np.array([b[1, 2] - b[2, 1],
                   b[2, 0] - b[0, 2],
                   b[0, 1] - b[1, 0]])
    s = b + b.T
    traceB = np.trace(b)
    return s, traceB, z


def davenportsQMethod(b):
    k, s, traceB, z = helper(b)
    eigVal, eigVec = linalg.eig(k)
    eigVecMax = eigVec[:, eigVal.argmax()]
    q = mt.Quaternion(eigVecMax[3], *eigVecMax[:3])
    return q


def quest(b, wei):
    s, traceB, z = helper2(b)
    lambdaMax, alpha, _ = calcLambdaMaxNewtonFOAM(b, wei.sum())
    alpha = lambdaMax**2 - traceB**2 + traceAdj3x3(s)
    gamma = alpha * (lambdaMax + traceB) - linalg.det(s)
    beta = lambdaMax - traceB
    x = (alpha * np.identity(3) + beta * s + s.dot(s)).dot(z)
    q = mt.Quaternion(gamma, *x)
    q.normalize()
    return q / q.norm()


def svd(b):
    u, s, vH = linalg.svd(b)
    detU = linalg.det(u)
    detV = linalg.det(vH)
    return u.dot(np.diag(np.array([1, 1, detU * detV])).dot(vH))


def svdWithError(b):
    u, s, vH = linalg.svd(b)
    detU = linalg.det(u)
    detV = linalg.det(vH)
    detUdetV = detU * detV
    aOpt = u.dot(np.diag(np.array([1, 1, detUdetV])).dot(vH))
    s1 = s[0, 0]
    s2 = s[1, 1]
    s3 = s[2, 2] * detUdetV
    p = u.dot(np.diag(np.array([1 / (s2 + s3), 1 / (s1 + s3), 1 / (s1 + s2)])).dot(u.T))
    return aOpt, p


def foam(b, wei):
    lambdaMax, alpha, normBSq = calcLambdaMaxNewtonFOAM(b, wei.sum())
    kappa = 0.5 * alpha
    return ((kappa + normBSq) * b + lambdaMax * adj3x3(b.T) - b.dot(b.T).dot(b)) / (kappa * lambdaMax - linalg.det(b))


def foamQ(b, wei):
    lambdaMax, alpha, normBSq = calcLambdaMaxNewtonFOAM(b, wei.sum())
    kappa = 0.5 * alpha
    a = ((kappa + normBSq) * b + lambdaMax * adj3x3(b.T) - b.dot(b.T).dot(b)) / (kappa * lambdaMax - linalg.det(b))
    q = ot.dcm2Qua(a)
    return q


def calcLambdaMaxESOQ(k, s, traceB, z):
    # from http://mortari.tamu.edu/attitude-estimation/j03.pdf
    b = -2 * traceB**2 + traceAdj3x3(s) - z.dot(z)
    c = 0
    I = np.arange(1, 4)
    for i in range(4):
        c += -linalg.det(k[np.meshgrid(I, I)])
        if i < 3:
            I[i] -= 1
    d = linalg.det(k)
    bThird = b / 3
    p = bThird**2 + 4 * d / 3
    q = bThird**3 - 4 * d * bThird + c**2 / 2
    u1 = 2 * np.sqrt(p) * np.cos(np.arccos(q / p**(3 / 2)) / 3) + bThird
    g1 = np.sqrt(u1 - b)
    g2 = 2 * np.sqrt(u1**2 - 4 * d)
    # Well.. This took a while... This | fucker apparently shouldn't be a minus as the article says
    #                                  v
    return 0.5 * (g1 + np.sqrt(-u1 - b + g2))


def calcLambdaMaxNewtonQUEST(s, traceB, z, lMax):
    alpha = lMax**2 - traceB**2 + traceAdj3x3(s)
    gamma = alpha * (lMax + traceB) - linalg.det(s)
    beta = lMax - traceB
    x = (alpha * np.identity(3) + beta * s + s.dot(s)).dot(z)
    return lMax - gamma * (gamma * beta - z.T.dot(x)) / (gamma**2 + x.dot(x))


def calcLambdaMaxNewtonESOQ(k, s, traceB, z, lMax):
    b = -2 * traceB**2 + traceAdj3x3(s) - z.dot(z)
    c = 0
    I = np.arange(1, 4)
    for i in range(4):
        c += -linalg.det(k[np.meshgrid(I, I)])
        if i < 3:
            I[i] -= 1
    d = linalg.det(k)
    return lMax - (lMax**4 + b * lMax**2 + c * lMax + d) / (4 * lMax**3 + 2 * b * lMax + c)


def calcLambdaMaxNewtonFOAM(b, lMax):
    normBSq = linalg.norm(b)**2
    alpha = lMax**2 - normBSq
    beta = 8 * linalg.det(b)
    lMax = lMax - (alpha**2 - lMax * beta - 4 * linalg.norm(adj3x3(b))**2) / (4 * lMax * alpha - beta)
    return lMax, alpha, normBSq


def esoq(b, wei):
    k, s, traceB, z = helper(b)
    lambdaMax, alpha, _ = calcLambdaMaxNewtonFOAM(b, wei.sum())
    h = k - lambdaMax * np.identity(4)
    I = np.arange(1, 4)
    diagElements = np.empty(4)
    for i in range(4):
        diagElements[i] = linalg.det(h[np.meshgrid(I, I)])
        if i < 3:
            I[i] -= 1
    # make sure this is correct - though a bit convoluted, this seems to be what the article by Daniele Mortari says
    farthestDiag = np.argmax(np.abs(diagElements))

    qElements = np.empty(4)
    I = np.arange(1, 4)
    J = np.array([i for i in range(4) if i != farthestDiag])
    for i in range(4):
        if i == farthestDiag:
            qElements[i] = diagElements[farthestDiag]
        else:
            qElements[i] = (-1)**(farthestDiag + i) * linalg.det(h[np.meshgrid(I, J)])
        if i < 3:
            I[i] -= 1
    q = mt.Quaternion(qElements[3], *qElements[:3])
    q.normalize()
    return q


def esoq2(b, wei):
    s, traceB, z = helper2(b)
    lambdaMax, _, _ = calcLambdaMaxNewtonFOAM(b, wei.sum())
    m = (lambdaMax - traceB) * ((lambdaMax + traceB) * np.identity(3) - s) - np.outer(z, z)
    adjM = adj3x3(m)
    maxCross = np.argmax(np.abs([adjM[0, 0], adjM[1, 1], adjM[2, 2]]))
    y = adjM[:, maxCross]
    q = mt.Quaternion(z.dot(y), *(lambdaMax - traceB) * y)
    return q / q.norm()


def calcLambdaMax2Obs(ref, obs, wei=None):
    if wei is None:
        return np.sqrt(2 * (1 + wei[0] * wei[1] * (obs[:, 0].dot(obs[:, 1] * ref[:, 0].dot(ref[:, 1]) + np.cross(obs[:, 0], obs[:, 1]) * np.cross(ref[:, 0], ref[:, 1])))))
    else:
        return np.sqrt(wei[0]**2 + wei[1]**2 + 2 * wei[0] * wei[1] * (obs[:, 0].dot(obs[:, 1] * ref[:, 0].dot(ref[:, 1]) + np.cross(obs[:, 0], obs[:, 1]) * np.cross(ref[:, 0], ref[:, 1]))))


if __name__ == "__main__":
    # Number of observations/references
    n = 100

    # Generate random reference vectors
    ref = np.random.rand(3, n) - 0.5
    ref /= linalg.norm(ref, axis=0)

    # Choose a random rotation
    trueQ = mt.Quaternion(*(np.random.rand(4) - 0.5))
    trueQ.normalize()

    # Construct noisy observations from the chosen rotation and references
    # obs = trueQ.rotateVectors(ref) + np.random.normal(loc=0, scale=0.1, size=(3, n))
    obs = trueQ.rotateVectors(ref) + 0.1 * (np.random.rand(3, n) - 0.5)
    obs /= linalg.norm(obs, axis=0)

    wei = np.ones(n)

    # Needed by all methods
    b = constructB(ref, obs)

    print("Loss from true rotation: {:0.4f}".format(lossFunction(trueQ, ref, obs)))
    print("Loss from firstSolution rotation: {:0.4f}".format(lossFunction(firstSolution(b), ref, obs)))
    if n > 2:
        print("Loss from alternateSolution rotation: {:0.4f}".format(lossFunction(alternateSolution(b), ref, obs)))
        print("Loss from unconstrainedLeastSquares rotation: {:0.4f}".format(lossFunction(unconstrainedLeastSquares(b, ref), ref, obs)))
    else:
        print("Loss from alternateSolution rotation: N/A. Needs more than to vectors")
        print("Loss from unconstrainedLeastSquares rotation: N/A. Needs more than to vectors")
    print("Loss from davenportsQMethod rotation: {:0.4f}".format(lossFunction(davenportsQMethod(b), ref, obs)))
    print("Loss from quest rotation: {:0.4f}".format(lossFunction(quest(b, wei), ref, obs)))
    print("Loss from svd rotation: {:0.4f}".format(lossFunction(svd(b), ref, obs)))
    print("Loss from foam rotation: {:0.4f}".format(lossFunction(foam(b, wei), ref, obs)))
    # print("Loss from foamQ rotation: {:0.4f}".format(lossFunction(foamQ(b, wei), ref, obs)))
    print("Loss from esoq rotation: {:0.4f}".format(lossFunction(esoq(b, wei), ref, obs)))
    print("Loss from esoq2 rotation: {:0.4f}".format(lossFunction(esoq2(b, wei), ref, obs)))
