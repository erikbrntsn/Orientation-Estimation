import numpy as np


def load_src(name, fpath):
    import os
    import imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))


mt = load_src('mathTools', 'quaternion/quaternion.py')

q = mt.Quaternion(*np.random.rand(4) - 0.5)
q.normalize()

vA = np.random.rand(3) - 0.5
vA /= np.linalg.norm(vA)

# Forward rotation
vF = q.rotateVector(vA)
# Backward rotation
vB = q.conjugate().rotateVector(vA)

crossAF = np.cross(vA, vF)
crossAF /= np.linalg.norm(crossAF)

crossBA = np.cross(vB, vA)
crossBA /= np.linalg.norm(crossBA)

print("\n")
print(crossAF)
print(crossBA)

vFF = q.rotateVector(vF)
crossFFF = np.cross(vFF, vF)
crossFFF /= np.linalg.norm(crossFFF)
print(crossFFF)
