
# Using cisi = np.array([c(\theta_1), s(\theta_1), c(\theta_2), s(\theta_2), c(\theta_2), s(\theta_2)])
# First rotation matrix is R_xyx = R_x(\theta_3) . R_y(\theta_2) . R_x(\theta_1)
# Second rotation matrix is R_xyz = R_x(\theta_3) . R_y(\theta_2) . R_z(\theta_1)
# And so on

import numpy as np


def anglesToCISI(angles):
  return np.array([np.cos(angles[0]), np.cos(angles[1]), np.cos(angles[2])]), \
         np.array([np.sin(angles[0]), np.sin(angles[1]), np.sin(angles[2])])


allRotMats = dict()


def rotMatXYX(angles):
  c, s = anglesToCISI(angles)
  return np.array([[c[1], s[0]*s[1], c[0]*s[1]],
                   [s[1]*s[2], c[0]*c[2] - c[1]*s[0]*s[2], -c[0]*c[1]*s[2] - c[2]*s[0]],
                   [-c[2]*s[1], c[0]*s[2] + c[1]*c[2]*s[0], c[0]*c[1]*c[2] - s[0]*s[2]]])
allRotMats['xyx'] = rotMatXYX


def rotMatXYZ(angles):
  c, s = anglesToCISI(angles)
  return np.array([[c[0]*c[1], -c[1]*s[0], s[1]],
                   [c[0]*s[1]*s[2] + c[2]*s[0], c[0]*c[2] - s[0]*s[1]*s[2], -c[1]*s[2]],
                   [-c[0]*c[2]*s[1] + s[0]*s[2], c[0]*s[2] + c[2]*s[0]*s[1], c[1]*c[2]]])
allRotMats['xyz'] = rotMatXYZ


def rotMatXZX(angles):
  c, s = anglesToCISI(angles)
  return np.array([[c[1], -c[0]*s[1], s[0]*s[1]],
                   [c[2]*s[1], c[0]*c[1]*c[2] - s[0]*s[2], -c[0]*s[2] - c[1]*c[2]*s[0]],
                   [s[1]*s[2], c[0]*c[1]*s[2] + c[2]*s[0], c[0]*c[2] - c[1]*s[0]*s[2]]])
allRotMats['xzx'] = rotMatXZX


def rotMatXZY(angles):
  c, s = anglesToCISI(angles)
  return np.array([[c[0]*c[1], -s[1], c[1]*s[0]],
                   [c[0]*c[2]*s[1] + s[0]*s[2], c[1]*c[2], -c[0]*s[2] + c[2]*s[0]*s[1]],
                   [c[0]*s[1]*s[2] - c[2]*s[0], c[1]*s[2], c[0]*c[2] + s[0]*s[1]*s[2]]])
allRotMats['xzy'] = rotMatXZY


def rotMatYXY(angles):
  c, s = anglesToCISI(angles)
  return np.array([[c[0]*c[2] - c[1]*s[0]*s[2], s[1]*s[2], c[0]*c[1]*s[2] + c[2]*s[0]],
                   [s[0]*s[1], c[1], -c[0]*s[1]],
                   [-c[0]*s[2] - c[1]*c[2]*s[0], c[2]*s[1], c[0]*c[1]*c[2] - s[0]*s[2]]])
allRotMats['yxy'] = rotMatYXY


def rotMatYXZ(angles):
  c, s = anglesToCISI(angles)
  return np.array([[c[0]*c[2] + s[0]*s[1]*s[2], c[0]*s[1]*s[2] - c[2]*s[0], c[1]*s[2]],
                   [c[1]*s[0], c[0]*c[1], -s[1]],
                   [-c[0]*s[2] + c[2]*s[0]*s[1], c[0]*c[2]*s[1] + s[0]*s[2], c[1]*c[2]]])
allRotMats['yxz'] = rotMatYXZ


def rotMatYZX(angles):
  c, s = anglesToCISI(angles)
  return np.array([[c[1]*c[2], -c[0]*c[2]*s[1] + s[0]*s[2], c[0]*s[2] + c[2]*s[0]*s[1]],
                   [s[1], c[0]*c[1], -c[1]*s[0]],
                   [-c[1]*s[2], c[0]*s[1]*s[2] + c[2]*s[0], c[0]*c[2] - s[0]*s[1]*s[2]]])
allRotMats['yzx'] = rotMatYZX


def rotMatYZY(angles):
  c, s = anglesToCISI(angles)
  return np.array([[c[0]*c[1]*c[2] - s[0]*s[2], -c[2]*s[1], c[0]*s[2] + c[1]*c[2]*s[0]],
                   [c[0]*s[1], c[1], s[0]*s[1]],
                   [-c[0]*c[1]*s[2] - c[2]*s[0], s[1]*s[2], c[0]*c[2] - c[1]*s[0]*s[2]]])
allRotMats['yzy'] = rotMatYZY


def rotMatZXY(angles):
  c, s = anglesToCISI(angles)
  return np.array([[c[0]*c[2] - s[0]*s[1]*s[2], -c[1]*s[2], c[0]*s[1]*s[2] + c[2]*s[0]],
                   [c[0]*s[2] + c[2]*s[0]*s[1], c[1]*c[2], -c[0]*c[2]*s[1] + s[0]*s[2]],
                   [-c[1]*s[0], s[1], c[0]*c[1]]])
allRotMats['zxy'] = rotMatZXY


def rotMatZXZ(angles):
  c, s = anglesToCISI(angles)
  return np.array([[c[0]*c[2] - c[1]*s[0]*s[2], -c[0]*c[1]*s[2] - c[2]*s[0], s[1]*s[2]],
                   [c[0]*s[2] + c[1]*c[2]*s[0], c[0]*c[1]*c[2] - s[0]*s[2], -c[2]*s[1]],
                   [s[0]*s[1], c[0]*s[1], c[1]]])
allRotMats['zxz'] = rotMatZXZ


def rotMatZYX(angles):
  c, s = anglesToCISI(angles)
  return np.array([[c[1]*c[2], -c[0]*s[2] + c[2]*s[0]*s[1], c[0]*c[2]*s[1] + s[0]*s[2]],
                   [c[1]*s[2], c[0]*c[2] + s[0]*s[1]*s[2], c[0]*s[1]*s[2] - c[2]*s[0]],
                   [-s[1], c[1]*s[0], c[0]*c[1]]])
allRotMats['zyx'] = rotMatZYX


def rotMatZYZ(angles):
  c, s = anglesToCISI(angles)
  return np.array([[c[0]*c[1]*c[2] - s[0]*s[2], -c[0]*s[2] - c[1]*c[2]*s[0], c[2]*s[1]],
                   [c[0]*c[1]*s[2] + c[2]*s[0], c[0]*c[2] - c[1]*s[0]*s[2], s[1]*s[2]],
                   [-c[0]*s[1], s[0]*s[1], c[1]]])
allRotMats['zyz'] = rotMatZYZ


