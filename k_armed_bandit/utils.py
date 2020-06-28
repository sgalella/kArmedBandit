import numpy as np
import matplotlib.pyplot as plt

def get_optimal_action(bandit):
    max_R = [sum(x) for x in list(bandit.values())]
    return max_R.index(max(max_R))

def run_bandit(bandit, num_steps, epsilon, action):
    k = len(bandit)
    Q = np.zeros((k, ))
    N = np.zeros((k, ))
    R = np.zeros((num_steps, ))
    A = np.zeros((num_steps, ))
    for iteration in range(num_steps):
        if np.random.random() > epsilon:
            idx = np.random.choice(np.flatnonzero(Q == Q.max()))
        else:
            idx = np.random.choice(k)
        mean, std = bandit[idx]
        R[iteration] = np.random.normal(mean, std)
        A[iteration] = 1 if idx == action else 0
        N[idx] += 1
        Q[idx] += (1 / (N[idx])) * (R[iteration] - Q[idx])
    return Q, R, A


def plot_bandit_dist(bandit):
    k = len(bandit)
    num_points = 10000
    data = np.zeros((num_points, k))
    actions = range(k)
    for action in actions:
        mean, std = bandit[action]
        data[:, action] = np.random.normal(mean, std, size=(num_points, ))
    plt.figure(figsize=(7, 5))
    plt.violinplot(data, positions=actions)
    plt.plot(np.mean(data, axis=0), '.r', markersize=10)
    plt.xlabel("Action", fontsize=12)
    plt.xticks(actions)
    plt.legend([r"$q_*$"], fontsize=12)
    plt.ylabel("Reward distribution", fontsize=12)
