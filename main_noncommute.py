import pennylane as qml
from pennylane import numpy as np
import math
import jax
import jax.numpy as jnp
import optax
import time
from make_data import D1, D2
import random

# Experimental settings -----
n_obs = 2  # number of observables
n_runs = 15  # how often the entire experiment is repeated
n_orders_train = 2  # how many randomly chosen orders to train on
n_orders_test = 0  # how many randomly chosen orders to test on

# Dataset settings -----
scores = [0.1, 0.9] # goodness scores for dataset 1 
rescale_coefficient = 0.9 # percentage of rescaling for dataset 2 

epochs = 150  # number of training steps

# some random seeds
rng_params = np.random.default_rng(seed=2)
random.seed(0)
rng_data = np.random.default_rng(seed=284)
# -------------------


def U(weights, wires):
    n_obs = len(wires)
    for i in range(n_obs):
        qml.RX(weights[i][0], wires=wires[i])
    for i in range(n_obs):
        qml.RZ(weights[i][1], wires=wires[i])
    for i in range(n_obs):
        qml.RX(weights[i][2], wires=wires[i])
    for i in range(n_obs - 1):
        qml.IsingXX(weights[i][3], wires=[wires[i], wires[(i + 1) % n_obs]])

def obs_check(weights):
    wires=range(n_obs)
    for i in range(n_obs):
        qml.RX(weights[i][0], wires=wires[i])
    for i in range(n_obs):
        qml.RZ(weights[i][1], wires=wires[i])
    for i in range(n_obs):
        qml.RX(weights[i][2], wires=wires[i])
    for i in range(n_obs - 1):
        qml.IsingXX(weights[i][3], wires=[wires[i], wires[(i + 1) % n_obs]])


@jax.jit
@qml.qnode(qml.device("default.mixed", wires=2 * n_obs))
def circuit(params, order):
    """
    :param params:
    :param order (list[int]): permutation of indices indicating the order of observables
    :return: probability distribution of computational basis measurements
    """
    n_obs = len(order)
    for i, o in enumerate(order):
        qml.adjoint(U)(params[o], wires=range(n_obs))
        qml.PhaseDamping(1, wires=0)
        qml.CNOT(wires=[0, n_obs + i])
        U(params[o], wires=range(n_obs))
    return qml.probs(wires=range(n_obs, 2 * n_obs))


def loss_fn(params, orders, target_distributions):
    model_distributions = jnp.stack([circuit(params, order) for order in orders])
    diff = model_distributions - target_distributions
    return jnp.sum(diff * diff) / len(orders)

def get_noncommutativity(params):
    matrix_fn = qml.matrix(obs_check)
    observables = []
    for i in params: 
        mat = matrix_fn(np.asarray(i))
        observables.append(np.asarray(mat))

    elem_list = []
    for num, i in enumerate(observables[:-1]): 
        for j in observables[(num + 1):]: 
            prod = np.matmul(np.matrix(i), np.matrix(j))
            reverse_prod = np.matmul(np.matrix(j), np.matrix(i))
            comm = np.subtract(prod, reverse_prod)
            comm_tr = np.matrix(comm).getH()
            comm_prod = np.matmul(comm_tr, comm)
  
            for idx, matrix in enumerate(comm_prod): 
                comm_prod[idx] = np.sqrt(matrix)
            elem = np.trace(comm_prod)
            elem_list.append(elem)
    return np.sum(elem_list)


@jax.jit
def update(params, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(params, orders_train, targets_train)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, loss, opt_state


histories_train = []
histories_test = []
non_commute = []
for i in range(n_runs):
    # pick random initial parameters
    # params = rng_params.uniform(low=-math.pi, high=math.pi, size=(n_obs, n_obs, 4))
    params = np.array([[[rng_params.uniform(low=-math.pi, high=math.pi) for i in range(4)] for i in range(n_obs) ]] * n_obs)

    # pick unique random orders
    random_orders = []
    while len(random_orders) < (n_orders_test + n_orders_train):
        candidate = list(range(n_obs))
        random.shuffle(candidate)
        if candidate not in random_orders:
            random_orders.append(candidate)
    orders_train = random_orders[:n_orders_train]


    # generate random instances of the data distributions
    targets_train = jnp.stack(
        [
            artificial_data_sampler2(np.asarray(order), rescale_coefficient = rescale_coefficient, seed = 123)
            for order in orders_train
        ]
    )

    opt = optax.adam(learning_rate=0.1)
    opt_state = opt.init(params)

    loss_history_train = []
    loss_history_test = []
    commute_score = [] 
    for j in range(epochs):
        # print("step", i)
        time0 = time.time()
        commute_score.append(get_noncommutativity(params))
        params, loss, opt_state = update(params, opt_state)
        loss_history_train.append(loss)

    non_commute.append(commute_score)
    histories_train.append(loss_history_train)
    print("Run " + str(i) + " is completed!")

a = np.array(histories_train)
c = np.array(non_commute)
np.savetxt(
    f"noncommutativity-order-effects/results/dataset2-{n_obs}_score-{rescale_coefficient}_train_.csv",
    a,
)

np.savetxt(
    f"noncommutativity-order-effects/results/datase2-{n_obs}_score-{rescale_coefficient}_commute.csv",
    c,
)
