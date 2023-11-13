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
n_obs = 5  # number of observables
n_runs = 15  # how often the entire experiment is repeated
train_list = [4, 8]  # list of how many randomly chosen orders to train on
n_orders_test = 10  # how many randomly chosen orders to test on
scores = [0.5, 0.8] # goodness scores for D1

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

def obs(weights):
    wires = [0, 1]
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

@jax.jit
def update(params, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(params, orders_train, targets_train)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, loss, opt_state




for n_orders_train in train_list: 
    histories_train = []
    histories_test = []
    for i in range(n_runs):
        # pick random initial parameters1`
        params = rng_params.uniform(low=-math.pi, high=math.pi, size=(n_obs, n_obs, 4))

        # pick unique random orders
        random_orders = []
        while len(random_orders) < (n_orders_test + n_orders_train):
            candidate = list(range(n_obs))
            random.shuffle(candidate)
            if candidate not in random_orders:
                random_orders.append(candidate)
        orders_train = random_orders[:n_orders_train]
        orders_test = random_orders[n_orders_train:]

        # generate random instances of the data distributions
        targets_train = jnp.stack(
            [
                artificial_data_sampler(np.asarray(order), rng=rng_data)
                for order in orders_train
            ]
        )
        targets_test = jnp.stack(
            [
                artificial_data_sampler(np.asarray(order), rng=rng_data)
                for order in orders_test
            ]
        )

        opt = optax.adam(learning_rate=0.1)
        opt_state = opt.init(params)

        loss_history_train = []
        loss_history_test = []
        for j in range(epochs):
            # print("step", i)
            time0 = time.time()
            params, loss, opt_state = update(params, opt_state)
            loss_history_train.append(loss)
            test_loss = loss_fn(params, orders_test, targets_test)
            loss_history_test.append(test_loss)
            print(i, time.time() - time0)
        histories_train.append(loss_history_train)
        histories_test.append(loss_history_test)
        print("Run " + str(i) + " is completed!")

    a = np.array(histories_train)
    b = np.array(histories_test)
    np.savetxt(
        f"/Users/kgili/Downloads/noncommutativity-order-effects/results/dataset_train_losses-{n_obs}_obs-{n_orders_train}_train.csv",
        a,
    )
    np.savetxt(
        f"/Users/kgili/Downloads/noncommutativity-order-effects/results/dataset_train_losses-{n_obs}_obs-{n_orders_train}_test.csv",
        b,
    )

