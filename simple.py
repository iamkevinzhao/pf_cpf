import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from scipy.interpolate import interp1d

bound = (0.0, 20.0)
step_size = (bound[1] - bound[0]) / 10

d_threshold = 40

def H(state):
    # return np.sin(50 * np.deg2rad(state))
    x = np.linspace(0, 30, num=11, endpoint=True)
    y = np.cos(-x ** 2 / 9.0)
    f = interp1d(x, y, kind='cubic')
    # f = interp1d([0, 5, 10, 15, 20, 25, 30], 2.0 * np.array([0, 0.1, 0.3, 0.34, 0.5, 0.9, 1.4]), 'cubic')
    return f(state)

class Particle:
    s = []
    m = []
    d = float('inf')

particles = []
particle_num = 100
for i in range(0, particle_num):
    particle = Particle()
    particle.s = np.random.uniform(bound[0], bound[1])
    particle.m = H(particle.s)
    particles.append(particle)

ground = Particle()
ground.s = (bound[1] - bound[0]) / 20.0

model = copy.deepcopy(ground)

# dev_s = step_size * 0.1
# dev_m = 0.3
dev_s = 0.3
dev_m = 0.3

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

def cpf(particle, model):
    x_hat = np.array([model.s, model.m]).transpose()
    x_tilde = np.array([particle.s, particle.m]).transpose()
    P = np.array([[math.pow(dev_s, 2), 0], [0, math.pow(dev_m, 2)]])
    P_inv = np.linalg.inv(P)
    return (x_hat - x_tilde).transpose().dot(P_inv).dot(x_hat - x_tilde)

for step in range(0, 8):
    model.s += step_size
    model.m = H(model.s)

    ground.s += step_size + np.random.normal(0, dev_s, 1)
    ground.m = H(ground.s) + np.random.normal(0, dev_m, 1)

    model.m = ground.m

    for particle in particles:
        particle.s += step_size + np.random.normal(0, dev_s, 1)[0]
        particle.m = H(particle.s)

    best = Particle()
    for particle in particles:
        particle.d = cpf(particle, model)
        if particle.d < best.d:
            best = copy.deepcopy(particle)

    particles.sort(key=lambda p: p.d)
    index = int(len(particles) * 0.5)
    for i in range(index, len(particles)):
        lottery = int(index * np.random.uniform(0.0, 1.0))
        particles[i] = copy.deepcopy(particles[lottery])

    sin_x = np.arange(bound[0], bound[1], (bound[1] - bound[0]) / 1000)
    sin_y = [H(x) for x in sin_x]
    plt.plot(sin_x, sin_y)
    plt.plot([p.s for p in particles], [p.m for p in particles], 'or', markersize=2)
    plt.plot(model.s, ground.m, 'ok', markersize=5)
    plt.plot(best.s, best.m, 'ok', markersize=2)
    plt.plot([best.s, model.s], [best.m, ground.m])
    plt.ylim((-2, 2))
    plt.xlim((bound[0], bound[1]))
    plt.gca().set_aspect('equal')
    plt.draw()
    plt.pause(1)
    plt.clf()
