import numpy as np
from matplotlib import pyplot as plt
from qutip import Bloch


def plot_bloch_sphere_evolution(dUs: np.ndarray, init_state: int = 0) -> None:
    n_states, state_size = dUs.shape[:2]
    assert state_size == 2, 'Error - can currently plot on the Bloch sphere only states of size 2.'
    state_over_time = np.zeros((n_states + 1, state_size), dtype=np.complex)
    state_over_time[0, init_state] = 1

    for i, dU in enumerate(dUs):
        state_over_time[i + 1] = dU @ state_over_time[i]

    phi = np.diff(np.angle(state_over_time))[:, 0]
    theta = np.arctan2(*zip(*np.abs(state_over_time[:, ::-1]))) * 2

    x_coord_on_sphere = np.sin(theta) * np.cos(phi)
    y_coord_on_sphere = np.sin(theta) * np.sin(phi)
    z_coord_on_sphere = np.cos(theta)

    bloch_sphere = Bloch()
    bloch_sphere.make_sphere()

    bloch_sphere.add_points([x_coord_on_sphere, y_coord_on_sphere, z_coord_on_sphere])

    bloch_sphere.render()


if __name__ == '__main__':
    def get_H(t):
        return np.array([[0, 1], [1, 0]]) + 5 * np.array([[np.cos(t), 0], [0, np.cos(t)]])


    dt = 1e-2
    dUs = np.array([np.eye(2) - 1j * get_H(t + dt / 2) * dt for t in np.arange(0, 25, dt)])

    plot_bloch_sphere_evolution(dUs)

    plt.show()
