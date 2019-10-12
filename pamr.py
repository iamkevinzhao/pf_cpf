import matplotlib.pyplot as plt
import math
import numpy as np

class StateFusion:
    last_lidar = []
    stereo_his = []
    vol = []
    trace = []
    state = []
    state_t = []
    cov = np.zeros([6, 6])
    t_approx = 0.15 * 1e9
    # Q = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    Q = np.diag([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
    # lidar_cov = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    lidar_cov = np.diag([0.2, 0.2, 0.5, 0.5, 0.5, 0.5])
    # stereo_cov = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    stereo_cov = np.diag([0.5, 0.5, 0.1, 0.1, 0.1, 0.1])
    def A(self, t):
        return np.array([[1, 0, 0, t, 0, 0],
                         [0, 1, 0, 0, t, 0],
                         [0, 0, 1, 0, 0, t],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])
    def sensor_cb(self, data):
        if data['type'] == 'stereo':
            data['cov'] = self.lidar_cov
        else:
            data['cov'] = self.stereo_cov

        if data['type'] == 'stereo':
            self.stereo_his.append(data)
            return

        if len(self.last_lidar) == 0:
            if data['type'] == 'lidar':
                self.last_lidar = data
            return

        ts = data['t'] - self.last_lidar['t']
        self.predict(ts)

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

        dp_lidar = np.array(data['p']) - np.array(self.last_lidar['p'])

        if len(stereo_2) and len(stereo_1):
            dp_stereo = np.array(stereo_2['p']) - np.array(stereo_1['p'])
            self.correct(data, dp_lidar, dp_stereo)
            print(data['t'] / float(1e9), stereo_2['t'] / float(1e9), self.last_lidar['t'] / float(1e9), stereo_1['t'] / float(1e9))
        else:
            self.correct(ts, dp_lidar)
            print(data['t'] / float(1e9), self.last_lidar['t'] / float(1e9))

        if data['type'] == 'lidar':
            self.last_lidar = data
    def predict(self, ts):
        pass
    def correct(self, ts, dp_lidar, dp_stereo = []):
        pass
    # def predict(self, data):
    #     if len(self.state) == 0:
    #         self.state = np.array(data['p'] + data['v']).transpose()
    #         self.state_t = data['t']
    #         self.cov = self.Q
    #         return
    #     t = data['t'] - self.state_t
    #     self.state_t = data['t']
    #     A = self.A(t / 10e9)
    #     self.state = A.dot(self.state)
    #     self.wrap_angles(self.state)
    #     self.cov = A.dot(self.cov).dot(A.transpose()) + self.Q
    #     # self.cov = self.Q
    #     # print(self.cov)
    #     pass
    # def correct(self, data):
    #     x_hat = np.concatenate((self.state, data['p'] + data['v']))
    #     P = np.zeros((12, 12))
    #     for c in range(0, 6):
    #         for r in range(0, 6):
    #             P[r, c] = self.cov[r, c]
    #     for c in range(0, 6):
    #         for r in range(0, 6):
    #             P[r + 6, c + 6] = data['cov'][r, c]
    #     P_inv = np.linalg.inv(P)
    #     M = np.vstack((np.identity(6), np.identity(6)))
    #     P_tilde = np.linalg.inv(M.transpose().dot(P_inv).dot(M))
    #     P_tilde = M.dot(P_tilde).dot(M.transpose())
    #     x_tilde = P_tilde.dot(P_inv).dot(x_hat)
    #     self.state = x_tilde[0:6]
    #     self.cov = P_tilde[0:6, 0:6]
    #     self.wrap_angles(self.state)
    #     self.trace.append(self.state)
    #     pass
    def wrap_angles(self, state):
        while state[2] > math.pi:
            state[2] = state[2] - math.pi
        while state[2] < -math.pi:
            state[2] = state[2] + math.pi

#######################################################

state_fusion = StateFusion()

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
        stereo_pose.append([float(parse[1]), float(parse[2]), float(parse[3])])

dx = lidar_pose[0][0] - stereo_pose[0][0]
dy = lidar_pose[0][1] - stereo_pose[0][1]
for i in range(0, len(stereo_pose)):
    stereo_pose[i][0] += dx
    stereo_pose[i][1] += dy

stereo_idx = 0
lidar_idx = 0
while (stereo_idx < (len(stereo_ts) - 1)) or (lidar_idx < (len(lidar_ts) - 1)):

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

    if (stereo['t'] > (stop - 1000000000)) or (lidar['t'] > (stop - 1000000000)):
        break
    # print(stereo['t'], lidar['t'])
    if stereo['t'] < lidar['t']:
        state_fusion.sensor_cb(stereo)
        if stereo_idx < (len(stereo_ts) - 1):
            stereo_idx = stereo_idx + 1
    else:
        state_fusion.sensor_cb(lidar)
        if lidar_idx < (len(lidar_ts) - 1):
            lidar_idx = lidar_idx + 1

exit()

gth = []
gth.append([0, -164, 90])
gth.append([-14.3, 0, 100])
gth.append([-15.4, 30, 100])
gth.append([-22, 60, 90])
gth.append([-33.8, 90, 45])
gth.append([-30, 105, 30])
gth.append([-13.1, 120, 20])
gth.append([0, 132.5, 10])
gth.append([30, 140, 10])
gth.append([60, 143.8, -5])
gth.append([90, 138.5, -5])
gth.append([120, 136.6, -3])
gth.append([150, 135.1, -2])
gth.append([180, 129.7, -2])
gth.append([210, 126.8, -2])
gth.append([240, 123.1, 0])
gth.append([270, 122.5, 0])
gth.append([300, 122.1, 0])
gth.append([330, 121.6, 0])
gth.append([360, 120.7, 0])
gth.append([390, 123.5, 3])
gth.append([420, 124, 3])
gth.append([450, 125.4, 4])
gth.append([465, 126.9, 5])
gth.append([495, 118.7, 30])
gth.append([525, 118, 50])
gth.append([555, 150, 80])
gth.append([561.9, 180, 90])
gth.append([563, 210, 95])
gth.append([559.8, 240, 100])
gth.append([556.7, 270, 100])
gth.append([555, 300, 105])
gth.append([548.7, 330, 105])
gth.append([542.8, 360, 100])
gth.append([540, 390, 95])
gth.append([538, 420, 93])
gth.append([536.8, 450, 90])
gth.append([536.8, 480, 90])
gth.append([534.5, 510, 85])
gth.append([537.7, 540, 85])
gth.append([537.5, 570, 85])
gth.append([540, 600, 80])
gth.append([542, 630, 75])
gth.append([545.1, 660, 70])
gth.append([549.2, 690, 70])
gth.append([555.6, 720, 70])
gth.append([563.6, 750, 70])
gth.append([567.5, 780, 75])
gth.append([570, 810, 80])
gth.append([572.5, 840, 85])
gth.append([572.3, 870, 90])
gth.append([572.2, 900, 93])
gth.append([570.5, 930, 95])
gth.append([569.5, 960, 95])
gth.append([569.3, 990, 92])
gth.append([568, 1020, 91])
gth.append([566.8, 1050, 90])
gth.append([567.2, 1080, 90])
gth.append([567.6, 1110, 90])
gth.append([566.7, 1140, 90])
gth.append([566.8, 1170, 92])
gth.append([563, 1200, 92])
gth.append([561.6, 1230, 92])
gth.append([558, 1260, 92])
gth.append([557.5, 1290, 90])
gth.append([547.4, 1320, 65])
# gth.append([540, 1335.2, 45])
gth.append([540, 1350, 35])
gth.append([568.8, 1380, 30])
gth.append([600, 1396.4, 10])
gth.append([630, 1404, 5])
gth.append([660, 1406.2, 1])
gth.append([690, 1405.7, -2])
gth.append([720, 1398.1, 0])
gth.append([750, 1397.8, 0])
gth.append([780, 1395.9, 0])
gth.append([810, 1394.5, 0])
gth.append([840, 1392.5, 2])
gth.append([870, 1393, 3])
gth.append([900, 1393, 4])
gth.append([930, 1395, 5])
gth.append([960, 1397.8, 5])
gth.append([990, 1403, 5])
gth.append([1020, 1407, 5])
gth.append([1042.5, 1410, 4])

for i in range(0, len(gth)):
   x = gth[i][0] + 35 * math.cos(gth[i][2] * math.pi / 180.0)
   y = gth[i][1] + 35 * math.sin(gth[i][2] * math.pi / 180.0) + 164 - 35
   x = x / 100.0 + 2.0
   y = y / 100.0 - 2.7
   gth[i] = [x, y]

for i in range(len(lidar_pose)):
    if abs(lidar_pose[i][0]) > 10 or abs(lidar_pose[i][1]) > 10:
        lidar_pose[i][0] = 0
        lidar_pose[i][1] = 0
# print(len(stereo_ts), len(lidar_pose))
plt.figure(0)
plt.plot([lidar_pose[i][0] for i in range(len(lidar_pose))],
         [lidar_pose[i][1] for i in range(len(lidar_pose))],
         '.', label='lidar', markersize=1)
plt.plot([stereo_pose[i][0] for i in range(len(stereo_pose))],
         [stereo_pose[i][1] for i in range(len(stereo_pose))],
         '.', label='stereo', markersize=1)
# plt.plot([gth[i][0] for i in range(len(gth))],
#          [gth[i][1] for i in range(len(gth))],
#          '.', label='groundtruth', markersize=1)
plt.plot([state_fusion.trace[i][0] for i in range(len(state_fusion.trace))],
         [state_fusion.trace[i][1] for i in range(len(state_fusion.trace))],
         '.', label='cpf', markersize=1)
plt.legend(loc='upper left')
plt.gca().set_aspect('equal')

plt.figure(1)
plt.plot([i for i in range(len(state_fusion.vol))],
         [state_fusion.vol[i][0] for i in range(len(state_fusion.vol))], '.', markersize=5)
# plt.plot([i for i in range(len(state_fusion.vol))],
#         [state_fusion.vol[i][1] for i in range(len(state_fusion.vol))], '.', markersize=3)
# plt.plot([i for i in range(len(state_fusion.vol))],
#         [state_fusion.vol[i][2] for i in range(len(state_fusion.vol))], '.', markersize=3)
plt.show()

# filename = "/home/kevin/KeyFrameTrajectory_TUM_Format.txt"
# stereo_ts = []
# stereo_pose = []
# i = -1
# with open(filename) as f:
#     for line in f:
#         # i = i + 1
#         # if (i % 5) != 0:
#         #     continue
#         parse = line.strip().split()
#         # stereo_ts.append(int(parse[0]))
#         stereo_pose.append([float(parse[1]), float(parse[2]), float(parse[3])])
# plt.figure(2)
# plt.plot([stereo_pose[i][0] for i in range(len(stereo_pose))],
#          [stereo_pose[i][2] for i in range(len(stereo_pose))],
#          '.', label='stereo', markersize=1)
# plt.gca().set_aspect('equal')
# plt.show()