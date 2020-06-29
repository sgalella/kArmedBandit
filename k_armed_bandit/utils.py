import numpy as np
import matplotlib.pyplot as plt


def get_optimal_action(bandit):
    max_R = [sum(x) if isinstance(x, tuple) else x for x in list(bandit.values())]
    return max_R.index(max(max_R))


def run_bandit_stat(bandit, num_steps, epsilon):
    k = len(bandit)
    Q = np.zeros((k, ))
    N = np.zeros((k, ))
    R = np.zeros((num_steps, ))
    A = np.zeros((num_steps, ))
    best_action = get_optimal_action(bandit)
    for iteration in range(num_steps):
        if np.random.random() > epsilon:
            idx = np.random.choice(np.flatnonzero(Q == Q.max()))
        else:
            idx = np.random.choice(k)
        mean, std = bandit[idx]
        R[iteration] = np.random.normal(mean, std)
        A[iteration] = 1 if idx == best_action else 0
        N[idx] += 1
        Q[idx] += (1 / (N[idx])) * (R[iteration] - Q[idx])
    return Q, R, A


def run_bandit_nonstat(k, num_steps, epsilon, alpha):
    Q = np.zeros((k, ))
    N = np.zeros((k, ))
    R = np.zeros((num_steps, ))
    A = np.zeros((num_steps, ))
    bandit = {idx:0 for idx in range(k)}
    history_R = [[0] for idx in range(k)]
    for iteration in range(num_steps):
        if np.random.random() > epsilon:
            idx = np.random.choice(np.flatnonzero(Q == Q.max()))
        else:
            idx = np.random.choice(k)
        R[iteration] = bandit[idx]
        best_action = get_optimal_action(bandit)
        A[iteration] = 1 if idx == best_action else 0
        N[idx] += 1
        n = len(history_R[idx])
        Q[idx] += history_R[idx][0] * (1 - alpha) ** n + sum([alpha * (1 - alpha) ** (n - i) * history_R[idx][i] for i in range(1, n)])
        history_R[idx].append(R[iteration])
        for idx in range(k):
            bandit[idx] += np.random.normal(0, 0.01)
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
