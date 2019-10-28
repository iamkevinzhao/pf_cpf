import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import block_diag
import warnings
warnings.simplefilter('ignore', np.RankWarning)

class StateFusion:
    def __init__(self):
        self.last_stereo = None
        self.last_lidar = None
        self.state = []
        self.ts = []
        self.stereo_cov = np.diag([0.05, 0.05, 0.05])
        self.model_cov = np.diag([0.1, 0.1, 0.1])
        self.lidar_cov = np.diag([0.7, 0.7, 0.7])
        pass
    def sensor_cb(self, data):
        if data['type'] == 'lidar':
            self.last_lidar = data
            if len(self.state) == 0:
                self.state.append(np.array([data['p']]).transpose())
                self.ts.append(data['t'])
            return

        if len(self.state) == 0:
            self.last_stereo = data
            return

        if self.last_stereo is None:
            self.last_stereo = data
            if self.last_lidar is not None:
                self.state.append(np.array([self.last_lidar['p']]).transpose())
                self.ts.append(data['t'])
            return

        stereo_state = self.state[-1] + np.array([data['p']]).transpose() - np.array([self.last_stereo['p']]).transpose()
        model_state = self.model_state(data['t'])
        if self.last_lidar is None:
            print('error')
        lidar_state = np.array([self.last_lidar['p']]).transpose()

        (state, cov) = self.cpf(stereo_state, model_state, lidar_state)
        self.state.append(state)
        self.ts.append(data['t'])
        self.last_stereo = data
        pass
    def cpf(self, stereo, model, lidar):
        # print(stereo, model, lidar)
        inf = 10000000
        P1 = self.stereo_cov.copy()
        P2 = self.model_cov.copy()
        P3 = self.lidar_cov.copy()
        if stereo is None:
            stereo = np.array([[0, 0, 0]]).transpose()
            P1 = np.diag([inf, inf, inf])
        if model is None:
            model = np.array([[0, 0, 0]]).transpose()
            P2 = np.diag([inf, inf, inf])
        if lidar is None:
            lidar = np.array([[0, 0, 0]]).transpose()
            P3 = np.diag([inf, inf, inf])
        x_hat = np.vstack((stereo, model, lidar))
        P = block_diag(P1, P2, P3)
        # print(x_hat)
        P_inv = np.linalg.inv(P)
        M = np.vstack((np.identity(3), np.identity(3), np.identity(3)))
        P_tilde = np.linalg.inv((M.transpose().dot(P_inv)).dot(M))
        x_tilde = ((P_tilde.dot(M.transpose())).dot(P_inv)).dot(x_hat)
        # print(x_tilde, P_tilde)
        return x_tilde, P_tilde
        pass
    def model_state(self, ts):
        buf_len = 10
        if len(self.state) < buf_len:
            return None
        t = np.array([self.ts[ii] for ii in range(len(self.ts) - buf_len, len(self.ts))])
        p = np.array([self.state[ii].transpose()[0] for ii in range(len(self.state) - buf_len, len(self.state))])
        # print(p)
        fitx = np.poly1d(np.polyfit(t, p[:, 0], 1))
        fity = np.poly1d(np.polyfit(t, p[:, 1], 1))
        fita = np.poly1d(np.polyfit(t, p[:, 2], 1))
        return np.array([[fitx(ts), fity(ts), fita(ts)]]).transpose()
        pass
    def wrap_angles(self, state):
        while state[2] > math.pi:
            state[2] = state[2] - math.pi
        while state[2] < -math.pi:
            state[2] = state[2] + math.pi

#######################################################

state_fusion = StateFusion()
outlier_removal = StateFusion()
outlier_removal.d9 = 10000

filename = "/home/kevin/pf_cpf/data/orb_pose.txt"
stereo_ts = []
stereo_pose = []
with open(filename) as f:
    for line in f:
        parse = line.strip().split()
        stereo_ts.append(int(parse[0]))
        measurement = [float(parse[1]) * 1.06, float(parse[2]) * 1.06, float(parse[3])]
        stereo_pose.append(measurement)

filename = "/home/kevin/pf_cpf/data/pamr_pose.txt"
lidar_ts = []
lidar_pose = []
i = -1
with open(filename) as f:
    for line in f:
        # i = i + 1
        # if (i % 10) != 0:
        #     continue
        parse = line.strip().split()
        if len(lidar_ts):
            if int(parse[0]) == lidar_ts[-1]:
                continue
        lidar_ts.append(int(parse[0]))
        x = -0.3
        y = 0
        phi = float(parse[3])
        h = float(parse[1])
        k = float(parse[2])
        xx = x * math.cos(phi) - y * math.sin(phi) + h
        yy = x * math.sin(phi) + y * math.cos(phi) + k

        if len(lidar_pose):
            xx = xx - (lidar_pose[0][0] - 0)
            yy = yy - (lidar_pose[0][1] - 0)
        lidar_pose.append([xx, yy, float(parse[3])])

stereo_idx = 0
lidar_idx = 0
while (stereo_idx < (len(stereo_ts) - 1)) and (lidar_idx < (len(lidar_ts) - 1)):

    stereo = {'t': stereo_ts[stereo_idx],
              'p': [stereo_pose[stereo_idx][0],
                    stereo_pose[stereo_idx][1],
                    stereo_pose[stereo_idx][2]],
              'type': 'stereo'}
    lidar = {'t': lidar_ts[lidar_idx],
             'p': [lidar_pose[lidar_idx][0],
                   lidar_pose[lidar_idx][1],
                   lidar_pose[lidar_idx][2]],
             'type': 'lidar'}

    if stereo['t'] < lidar['t']:
        state_fusion.sensor_cb(stereo.copy())
        outlier_removal.sensor_cb(stereo.copy())
        if stereo_idx < (len(stereo_ts) - 1):
            stereo_idx = stereo_idx + 1
    else:
        state_fusion.sensor_cb(lidar.copy())
        outlier_removal.sensor_cb(lidar.copy())
        if lidar_idx < (len(lidar_ts) - 1):
            lidar_idx = lidar_idx + 1

plt.figure(0)
plt.plot([lidar_pose[i][0] for i in range(len(lidar_pose))],
         [lidar_pose[i][1] for i in range(len(lidar_pose))],
         '-', label='lidar', linewidth=1)
plt.plot([stereo_pose[i][0] for i in range(len(stereo_pose))],
         [stereo_pose[i][1] for i in range(len(stereo_pose))],
         '-', label='stereo', linewidth=1)
plt.plot([state_fusion.state[i][0] for i in range(len(state_fusion.state))],
         [state_fusion.state[i][1] for i in range(len(state_fusion.state))],
         '-', label='cpf', linewidth=1)
plt.plot([outlier_removal.state[i][0] for i in range(len(outlier_removal.state))],
         [outlier_removal.state[i][1] for i in range(len(outlier_removal.state))],
         '-', label='cpf_outlier', linewidth=1)
plt.title('global position / meter')
plt.legend(loc='upper right')
plt.gca().set_aspect('equal')

# plt.figure(1)
# plt.plot([state_fusion.d[i][0] for i in range(0, len(state_fusion.d))],
#          [state_fusion.d[i][1] for i in range(0, len(state_fusion.d))],
#          '-', label='d', markersize=1)
# plt.legend(loc='upper right')
# plt.title('CPF distance over time')


plt.show()