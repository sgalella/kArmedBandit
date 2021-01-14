import numpy as np
import matplotlib.pyplot as plt


def get_optimal_action(bandit):
    """
    Returns the best action of a bandit.

    Args:
        bandit (list): Contains mean and std / reward for each k.

    Returns:
        int: Index of best action
    """
    max_R = [sum(x) if isinstance(x, tuple) else x for x in bandit]
    return max_R.index(max(max_R))


def argmax(Q):
    """
    Returns index of maximum Q. Breaks ties randomly.

    Args:
        Q (np.array): Q values for the different k.

    Returns:
        int: Index of maximum Q.
    """
    return np.random.choice(np.flatnonzero(Q == Q.max()))


def get_probabilities_action(H):
    """
    Computes the probabilities of selection given H.

    Args:
        H (np.array): Preference for each action.

    Returns:
        np.array: Probability vector.
    """
    return np.exp(H) / np.sum(np.exp(H))


def run_bandit_gradient(bandit, num_steps, alpha, baseline=True):
    """
    Computes the gradient bandit algorithm.

    Args:
        bandit (list): Contains mean and std / reward for each k. 
        num_steps (int): Total number of steps of the simulation.  
        alpha (int): Weight of recent rewards.
        baseline (bool, optional): Use baseline rewards. Defaults to True.

    Returns:
        H (np.array): Preference for each action.
        R (np.array): Reward obtained at each time step.
        A (np.array): Action at each time step.
    """
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
    """
    Computes the Upper-Confidence-Bound action selection algorithm.

    Args:
        bandit (list): Contains mean and std / reward for each k. 
        num_steps (int): Total number of steps of the simulation.  
        alpha (int, optional): Weight of recent rewards. Defaults to None.
        initial_values (np.array, optional): Initial Q values. Defaults to None.
        c (int, optional): Confidence bound parameter. Defaults to 1.

    Returns:
        H (np.array): Preference for each action.
        R (np.array): Reward obtained at each time step.
        A (np.array): Action at each time step.
    """
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
    """
    Runs the epsilon-greedy stationary problem.

    Args:
        bandit (list): Contains mean and std / reward for each k.  
        num_steps (int): Total number of steps of the simulation.  
        epsilon (int): Probability of not selecting optimal action.
        alpha (int, optional): Weight of recent rewards. Defaults to None.
        initial_values (np.array, optional): Initial Q values. Defaults to None.

    Returns:
        H (np.array): Preference for each action.
        R (np.array): Reward obtained at each time step.
        A (np.array): Action at each time step.
    """
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


def run_bandit_nonstat(k, num_steps, epsilon, alpha, initial_values=None):
    """
    Runs the epislon-greedy nonstationary problem. Bandits are initialized at 0
    and are updated at each step.

    Args:
        k (int): Number of different actions in bandit.
        num_steps (int): Total number of steps of the simulation.  
        epsilon (int): Probability of not selecting optimal action.
        alpha (int, optional): Weight of recent rewards. Defaults to None.
        initial_values (np.array, optional): Initial Q values. Defaults to None.

    Returns:
        H (np.array): Preference for each action.
        R (np.array): Reward obtained at each time step.
        A (np.array): Action at each time step.
    """
    if initial_values is None:
        Q = np.zeros((k, ))
    else:
        assert initial_values.shape == (k, )
        Q = initial_values.copy()
    R = np.zeros((num_steps, ))
    A = np.zeros((num_steps, ))
    bandit = [0 for idx in range(k)]
    for iteration in range(num_steps):
        if np.random.random() > epsilon:
            idx = argmax(Q == Q.max())
        else:
            idx = np.random.choice(k)
        R[iteration] = bandit[idx]
        best_action = get_optimal_action(bandit)
        A[iteration] = 1 if idx == best_action else 0
        Q[idx] += alpha * (R[iteration] - Q[idx])
        for idx in range(k):
            bandit[idx] += np.random.normal(0, 0.01)
    return Q, R, A


def plot_bandit_dist(bandit):
    """
    Plots distribution of rewards for each k in bandit.

    Args:
        bandit (list): Contains mean and std / reward for each k.
    """
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
    plt.show()
