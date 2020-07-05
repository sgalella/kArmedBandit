import numpy as np
import matplotlib.pyplot as plt


def get_optimal_action(bandit):
    max_R = [sum(x) if isinstance(x, tuple) else x for x in bandit]
    return max_R.index(max(max_R))


def argmax(Q):
    return np.random.choice(np.flatnonzero(Q == Q.max()))


def get_probabilities_action(H):
    return np.exp(H) / np.sum(np.exp(H))


def run_bandit_gradient(bandit, num_steps, alpha, baseline=True):
    k = len(bandit)
    actions = range(k)
    H = np.zeros((k, ))
    R = np.zeros((num_steps, ))
    R_mean = 0
    A = np.zeros((num_steps, ))
    best_action = get_optimal_action(bandit)
    for iteration in range(num_steps):
        prob = get_probabilities_action(H)
        idx = np.random.choice(actions, p=prob)
        mean, std = bandit[idx]
        R[iteration] = np.random.normal(mean, std)
        A[iteration] = 1 if idx == best_action else 0
        if baseline:
            R_mean += 1 / (iteration + 1) * (R[iteration] - R_mean)
        for action in actions:
            if action == idx:
                update = alpha * (R[iteration] - R_mean) * (1 - prob[action])
                H[action] += update if not np.isnan(update) else 0
            else:
                update = alpha * (R[iteration] - R_mean) * prob[action]
                H[action] -= update if not np.isnan(update) else 0
    return H, R, A


def run_bandit_ucb(bandit, num_steps, alpha=None, initial_values=None, c=1):
    k = len(bandit)
    if initial_values is None:
        Q = np.zeros((k, ))
    else:
        assert initial_values.shape == (k, )
        Q = initial_values.copy()
    N = np.zeros((k, ))
    R = np.zeros((num_steps, ))
    A = np.zeros((num_steps, ))
    best_action = get_optimal_action(bandit)
    for iteration in range(num_steps):
        action_upb = Q + c * np.sqrt(np.log(iteration) / N)
        action_upb[np.where(np.isnan(action_upb))] = np.inf
        idx = argmax(action_upb)
        mean, std = bandit[idx]
        R[iteration] = np.random.normal(mean, std)
        A[iteration] = 1 if idx == best_action else 0
        if alpha is None:
            N[idx] += 1
            Q[idx] += (1 / (N[idx])) * (R[iteration] - Q[idx])
        else:
            Q[idx] += alpha * (R[iteration] - Q[idx])
    return Q, R, A


def run_bandit_stat(bandit, num_steps, epsilon, alpha=None, initial_values=None):
    k = len(bandit)
    if initial_values is None:
        Q = np.zeros((k, ))
    else:
        assert initial_values.shape == (k, )
        Q = initial_values.copy()
    N = np.zeros((k, ))
    R = np.zeros((num_steps, ))
    A = np.zeros((num_steps, ))
    best_action = get_optimal_action(bandit)
    for iteration in range(num_steps):
        if np.random.random() > epsilon:
            idx = argmax(Q == Q.max())
        else:
            idx = np.random.choice(k)
        mean, std = bandit[idx]
        R[iteration] = np.random.normal(mean, std)
        A[iteration] = 1 if idx == best_action else 0
        if alpha is None:
            N[idx] += 1
            Q[idx] += (1 / (N[idx])) * (R[iteration] - Q[idx])
        else:
            Q[idx] += alpha * (R[iteration] - Q[idx])
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
            idx = argmax(Q == Q.max())
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
