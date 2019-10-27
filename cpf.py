import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import block_diag
import warnings
warnings.simplefilter('ignore', np.RankWarning)

# x = -0.3
# y = 0
# phi = 0 * math.pi / 180
# h = 1
# k = 0
# xx = x * math.cos(phi) - y * math.sin(phi) + h
# yy = x * math.cos(phi) - y * math.sin(phi) + k
# print(xx, yy)
# exit()

initial_measurement = []
filename = "/home/kevin/pf_cpf/data/orb_pose.txt"
stereo_ts = []
stereo_pose = []
i = -1
with open(filename) as f:
    for line in f:
        parse = line.strip().split()
        stereo_ts.append(int(parse[0]))
        measurement = [float(parse[1]) * 1.06, float(parse[2]) * 1.06, float(parse[3])]
        # if not len(initial_measurement):
        #     initial_measurement = measurement.copy()
        #     measurement = [0, 0, 0]
        # else:
        #     measurement[0] = measurement[0] - initial_measurement[0]
        #     measurement[1] = measurement[1] - initial_measurement[1]
        # print(measurement)
        stereo_pose.append(measurement)

plt.plot([stereo_pose[i][0] for i in range(len(stereo_pose))],
         [stereo_pose[i][1] for i in range(len(stereo_pose))],
         '-', label='stereo', markersize=1)

filename = "/home/kevin/pf_cpf/data/pamr_pose.txt"
lidar_ts = []
lidar_pose = []
with open(filename) as f:
    for line in f:
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
        # lidar_pose.append([float(parse[1]), float(parse[2]), float(parse[3])])

print(len(stereo_ts))

plt.plot([lidar_pose[i][0] for i in range(len(lidar_pose))],
         [lidar_pose[i][1] for i in range(len(lidar_pose))],
         '-', label='lidar', markersize=1)

plt.legend(loc='upper right')
plt.gca().set_aspect('equal')
plt.show()
exit()

class StateFusion:

    def __init__(self):
        self.last_lidar = []
        self.stereo_his = []
        self.trace = []
        self.state = []
        self.state_his = []
        self.d = []
        # cov = np.diag([0.1, 0.1, 0.1])
        self.cov = np.diag([0.1, 0.1, 0.1])
        self.t_approx = 0.2 * 1e9
        self.Q = np.diag([0.0221, 0.0221, 0.033]) * 0.03
        # Q = np.diag([0.007, 0.007, 0.0733])
        self.lidar_cov = np.diag([0.00334, 0.00334, 0.0433]) * 0.03
        self.stereo_cov = np.diag([0.0153, 0.0153, 0.0388]) * 0.03
        # self.stereo_cov = np.diag([0.0203, 0.0103, 0.0388]) * 0.03
        self.d9 = 16.9
        self.trace_t = []
    def sensor_cb(self, data):
        if data['type'] == 'stereo':
            self.stereo_his.append(data)
            return

        if len(self.last_lidar) == 0:
            if data['type'] == 'lidar':
                self.last_lidar = data
            # return

        ts = data['t']
        self.predict(ts, data)

        stereo_2 = []
        stereo_1 = []
        for stereo in reversed(self.stereo_his):
            if abs(data['t'] - stereo['t']) < self.t_approx:
                stereo_2 = stereo
                break
        if len(stereo_2):
            for stereo in reversed(self.stereo_his):
                if stereo['t'] > self.last_lidar['t']:
                    continue
                if abs(self.last_lidar['t'] - stereo['t']) < self.t_approx:
                    stereo_1 = stereo
                    break

        dp_lidar = np.array([data['p']]) - np.array([self.last_lidar['p']])

        if len(stereo_2) and len(stereo_1):
            dp_stereo = np.array([stereo_2['p']]) - np.array([stereo_1['p']])
            self.correct(ts, dp_lidar, dp_stereo)
        else:
            self.correct(ts, dp_lidar)

        if data['type'] == 'lidar':
            self.last_lidar = data
    def predict(self, ts, data):
        buf_len = 5
        if len(self.state_his) < buf_len:
            self.state_his.append(data)
            return
        t = np.array([self.state_his[ii]['t'] for ii in range(len(self.state_his) - buf_len, len(self.state_his))])
        p = np.array([self.state_his[ii]['p'] for ii in range(len(self.state_his) - buf_len, len(self.state_his))])
        # # print(t)
        fitx = np.poly1d(np.polyfit(t, p[:, 0], 2))
        fity = np.poly1d(np.polyfit(t, p[:, 1], 2))
        fita = np.poly1d(np.polyfit(t, p[:, 2], 2))
        state = {}
        state['p'] = [fitx(data['t']), fity(data['t']), fita(data['t'])]
        self.cov = self.Q
        self.wrap_angles(state['p'])
        self.state_his.append(state)
        pass
    def correct(self, ts, dp_lidar, dp_stereo = []):
        inf = 10000000000
        x1 = np.array([[0, 0, 0]]).transpose()
        q1 = np.diag([inf, inf, inf])
        x2 = x1.copy()
        q2 = q1.copy()
        x3 = x1.copy()
        q3 = q1.copy()
        if len(self.state_his) > 2:
            x1 = (np.array([self.state_his[-1]['p']]) - np.array([self.state_his[-2]['p']])).transpose()
            q1 = self.cov
        x2 = dp_lidar.transpose()
        q2 = self.lidar_cov
        if len(dp_stereo):
            x3 = dp_stereo.transpose()
            q3 = self.stereo_cov
        x_hat = np.vstack((x1, x2, x3))
        P = block_diag(q1, q2, q3)
        P_inv = np.linalg.inv(P)
        M = np.vstack((np.identity(3), np.identity(3), np.identity(3)))
        P_tilde = np.linalg.inv((M.transpose().dot(P_inv)).dot(M))
        x_tilde = ((P_tilde.dot(M.transpose())).dot(P_inv)).dot(x_hat)
        np.set_printoptions(precision=3, suppress=True)
        if len(self.state_his) < 2:
            ppp = self.state_his[-1]['p'] + x_tilde[:, 0]
        else:
            ppp = self.state_his[-2]['p'] + x_tilde[:, 0]
        state = {}
        state['p'] = [ppp[0], ppp[1], ppp[2]]
        state['t'] = ts
        self.state_his[-1] = state
        self.cov = P[0:3, 0:3]
        self.trace.append(state['p'])
        self.trace_t.append(ts)

        x_tilde3 = np.vstack((x_tilde, x_tilde, x_tilde))
        d = (((x_hat - x_tilde3).transpose()).dot(P_inv)).dot(x_hat - x_tilde3)[0][0]
        self.d.append([ts / 1e9, d])
        if d > self.d9:
            ppp = self.state_his[-2]['p'] + x2[:, 0]
            P[6, 6] = inf
            P[7, 7] = inf
            P[8, 8] = inf
            P_inv = np.linalg.inv(P)
            P_tilde = np.linalg.inv((M.transpose().dot(P_inv)).dot(M))
            x_tilde = ((P_tilde.dot(M.transpose())).dot(P_inv)).dot(x_hat)
            if len(self.state_his) < 2:
                ppp = self.state_his[-1]['p'] + x_tilde[:, 0]
            else:
                ppp = self.state_his[-2]['p'] + x_tilde[:, 0]
            state['p'] = [ppp[0], ppp[1], ppp[2]]
            self.state_his[-1] = state
            self.trace[-1] = state['p']
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

stop = 1570543968000000000
filename = "/home/kevin/pf_cpf/data/pamr_pose.txt"
lidar_ts = []
lidar_pose = []
with open(filename) as f:
    for line in f:
        parse = line.strip().split()
        if int(parse[0]) > stop:
            break
        lidar_ts.append(int(parse[0]))
        lidar_pose.append([float(parse[1]), float(parse[2]), float(parse[3])])

filename = "/home/kevin/pf_cpf/data/orb_pose.txt"
stereo_ts = []
stereo_pose = []
i = -1
with open(filename) as f:
    for line in f:
        i = i + 1
        if (i % 5) != 0:
            continue
        parse = line.strip().split()
        if int(parse[0]) > stop:
            break
        stereo_ts.append(int(parse[0]))
        measurement = [float(parse[1]), float(parse[2]), float(parse[3]) + math.pi]
        state_fusion.wrap_angles(measurement)
        if measurement[2] > (150 * math.pi / 180):
            measurement[2] -= math.pi
        stereo_pose.append(measurement)

for i in range(0, len(stereo_pose)):
    stereo_pose[i][0] = -stereo_pose[i][0]
    stereo_pose[i][1] = -stereo_pose[i][1]

dx = lidar_pose[0][0] - stereo_pose[0][0]
dy = lidar_pose[0][1] - stereo_pose[0][1]
for i in range(0, len(stereo_pose)):
    stereo_pose[i][0] = stereo_pose[i][0] + dx
    stereo_pose[i][1] = stereo_pose[i][1] + dy

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

    # if (stereo['t'] > (stop - 1000000000)) or (lidar['t'] > (stop - 1000000000)):
    #     break
    # print(stereo['t'], lidar['t'])
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

for i in range(len(lidar_pose)):
    if abs(lidar_pose[i][0]) > 10 or abs(lidar_pose[i][1]) > 10:
        lidar_pose[i][0] = 0
        lidar_pose[i][1] = 0

# print('-------------')
# print(state_fusion.trace[1], lidar_pose[1])
# print('-------------')

del lidar_pose[-3:]
del lidar_ts[-3:]
del stereo_pose[-2:]
del stereo_ts[-2:]
del state_fusion.trace[-2:]
del state_fusion.trace_t[-2:]
del outlier_removal.trace[-1:]
del outlier_removal.trace_t[-1:]

plt.figure(0)
plt.plot([lidar_pose[i][0] for i in range(len(lidar_pose))],
         [lidar_pose[i][1] for i in range(len(lidar_pose))],
         '-', label='lidar', markersize=1)
plt.plot([stereo_pose[i][0] for i in range(len(stereo_pose) - 2)],
         [stereo_pose[i][1] for i in range(len(stereo_pose) - 2)],
         '-', label='stereo', markersize=1)
plt.plot([state_fusion.trace[i][0] for i in range(len(state_fusion.trace))],
         [state_fusion.trace[i][1] for i in range(len(state_fusion.trace))],
         '-', label='cpf', markersize=1)
plt.plot([outlier_removal.trace[i][0] for i in range(len(outlier_removal.trace))],
         [outlier_removal.trace[i][1] for i in range(len(outlier_removal.trace))],
         '-', label='cpf_outlier', markersize=1)
plt.title('global position / meter')
ground = np.loadtxt('data/groundtruth.txt')
print("(%f, %f, %f) vs. (%f, %f, %f) " %
      (ground[-1, 0], ground[-1, 1], ground[-1, 2],
      state_fusion.trace[-1][0], state_fusion.trace[-1][1], state_fusion.trace[-1][2]))
plt.plot(ground[:, 0], ground[:, 1], '-', label='ground-truth', markersize=1)
plt.legend(loc='lower right')
plt.gca().set_aspect('equal')

plt.figure(1)
plt.plot([state_fusion.d[i][0] for i in range(0, len(state_fusion.d))],
         [state_fusion.d[i][1] for i in range(0, len(state_fusion.d))],
         '-', label='d', markersize=1)
plt.plot([state_fusion.d[i][0] for i in range(0, len(state_fusion.d))],
         [16.9 for i in range(0, len(state_fusion.d))], '-')
plt.title('CPF distance over time')

plt.figure(2)
plt.plot([lidar_ts[i] for i in range(len(lidar_ts))],
         [math.degrees(lidar_pose[i][2]) for i in range(len(lidar_pose))],
         '-', label='lidar', markersize=1)
plt.plot([stereo_ts[i] for i in range(len(stereo_pose))],
         [math.degrees(stereo_pose[i][2]) for i in range(len(stereo_pose))],
         '-', label='stereo', markersize=1)
plt.plot([state_fusion.trace_t[i] for i in range(len(state_fusion.trace_t))],
         [math.degrees(state_fusion.trace[i][2]) for i in range(len(state_fusion.trace))],
         '-', label='cpf', markersize=1)
plt.plot([outlier_removal.trace_t[i] for i in range(len(outlier_removal.trace_t))],
         [math.degrees(outlier_removal.trace[i][2]) for i in range(len(outlier_removal.trace))],
         '-', label='cpf_outlier', markersize=1)
ground = np.delete(ground, -1, 0)
plt.plot([ground[i][4] for i in range(len(ground))],
         [math.degrees(ground[i][2]) for i in range(len(ground))],
         '-', label='groundtruth', markersize=1)
plt.legend(loc='upper right')
plt.title('global heading (degree) over time')

plt.show()