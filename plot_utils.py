import numpy as np
from matplotlib import pyplot as plt
from qutip import Bloch
from ground_truth import naive_simulation, get_Ht_test1
import hermitian_functions


def plot_U_elements(T, dt, get_Ht):
    '''
        Plots the four dimensions of the 2x2 matrix U over time T, calculating U at intervals sized dt
    '''
    x = np.linspace(0, T, int(T/dt)+1)
    U_T, y = naive_simulation(get_Ht, T, dt)

    y1 = [y[i][0,0] for i in range(len(y))]
    y2 = [y[i][0,1] for i in range(len(y))]
    y3 = [y[i][1,0] for i in range(len(y))]
    y4 = [y[i][1,1] for i in range(len(y))]

    fig = plt.figure(figsize=(10,20))
    ax = plt.axes(projection ='3d')

    y1r = [y1[i].real for i in range(len(y1))]
    y1i = [y1[i].imag for i in range(len(y1))]
    y2r = [y2[i].real for i in range(len(y2))]
    y2i = [y2[i].imag for i in range(len(y2))]
    y3r = [y3[i].real for i in range(len(y3))]
    y3i = [y3[i].imag for i in range(len(y3))]
    y4r = [y4[i].real for i in range(len(y4))]
    y4i = [y4[i].imag for i in range(len(y4))]


    ax.plot(x, y1r, y1i, label = "y1 (0,0)")
    ax.plot(x, y2r, y2i, label = "y2 (0,1)")
    ax.plot(x, y3r, y3i, label = "y3 (1,0)")
    ax.plot(x, y4r, y4i, label = "y4 (1,1)")
    ax.legend()
    ax.set_xlabel("time t")
    ax.set_ylabel("real")
    ax.set_zlabel("complex")


def plot_H_elements(get_Ht):
    x = np.linspace(-5, 10, 300)
    y = [get_Ht(t) for t in x]

    y1 = [y[i][0, 0] for i in range(len(y))]
    y2 = [y[i][0, 1] for i in range(len(y))]
    y3 = [y[i][1, 0] for i in range(len(y))]
    y4 = [y[i][1, 1] for i in range(len(y))]

    fig = plt.figure(figsize=(10, 20))
    ax = plt.axes(projection='3d')

    y1r = [y1[i].real for i in range(len(y1))]
    y1i = [y1[i].imag for i in range(len(y1))]
    y2r = [y2[i].real for i in range(len(y2))]
    y2i = [y2[i].imag for i in range(len(y2))]
    y3r = [y3[i].real for i in range(len(y3))]
    y3i = [y3[i].imag for i in range(len(y3))]
    y4r = [y4[i].real for i in range(len(y4))]
    y4i = [y4[i].imag for i in range(len(y4))]

    ax.plot(x, y1r, y1i, label="y1 (0,0)")
    ax.plot(x, y2r, y2i, label="y2 (0,1)")
    ax.plot(x, y3r, y3i, label="y3 (1,0)")
    ax.plot(x, y4r, y4i, label="y4 (1,1)")
    ax.legend()


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

    plot_U_elements(50, 0.01, get_Ht_test1)
    plt.show()

    plot_H_elements(get_Ht_test1)
    plt.show()
