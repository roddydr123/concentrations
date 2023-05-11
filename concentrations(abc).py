import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit


def laplacian(grid, dx):
    
    return ((np.roll(grid, 1, axis=1) +
            np.roll(grid, -1, axis=1) +
            np.roll(grid, 1, axis=0) +
            np.roll(grid, -1, axis=0) -
            4 * grid) / dx**2)


def update_grid(grid, D, q, p, dx, dt, grid_size):

    a = grid[0,:,:]
    b = grid[1,:,:]
    c = grid[2,:,:]

    new_a = dt * ((D * laplacian(a, dx)) +
               (q * a * (1 - a - b - c)) -
               (p * a * c)) + a

    new_b = dt * ((D * laplacian(b, dx)) +
               (q * b * (1 - a - b - c)) -
               (p * a * b)) + b
    
    new_c = dt * ((D * laplacian(c, dx)) +
               (q * c * (1 - a - b - c)) -
               (p * b * c)) + c
    
    new_d = 1 - new_a - new_b - new_c
    
    grid = np.stack((new_a, new_b, new_c, new_d))

    tau = np.argmax(grid, axis=0) + 1

    return grid, tau


def animation(grid, D, q, p, dx, dt, grid_size):

    tau = np.zeros((grid_size, grid_size))
    fig, ax = plt.subplots()
    im = ax.imshow(tau, animated=True)
    # cbar = fig.colorbar(im, ax=ax)

    # choose a large range so that it will likely converge before then, but will never
    # continue forever.
    for i in range(1000000):

        # move one step forward in the simulation, updating phi at every point.
        grid, tau = update_grid(grid, D, q, p, dx, dt, grid_size)

        # every 50 sweeps update the animation.
        if i % 50 == 0:
            
            plt.cla()
            # cbar.remove()
            im = ax.imshow(tau, interpolation=None, animated=True)
            # cbar = fig.colorbar(im, ax=ax)
            plt.draw()
            plt.pause(0.00001)


def taskb(grid, D, q, p, dx, dt, grid_size):

    tau = np.zeros((grid_size, grid_size))

    nsteps = 20000

    frac_a = []
    frac_b = []
    frac_c = []

    # choose a large range so that it will likely converge before then, but will never
    # continue forever.
    for i in range(nsteps):

        # move one step forward in the simulation, updating phi at every point.
        grid, tau = update_grid(grid, D, q, p, dx, dt, grid_size)

        frac_a.append(np.sum(tau == 1) / grid_size**2)
        frac_b.append(np.sum(tau == 2) / (grid_size**2))
        frac_c.append(np.sum(tau == 3) / (grid_size**2))

    plt.plot(np.arange(nsteps) * dt, frac_a, label="a")
    plt.plot(np.arange(nsteps) * dt, frac_b, label="b")
    plt.plot(np.arange(nsteps) * dt, frac_c, label="c")
    plt.legend()
    plt.show()


def taskc(grid, D, q, p, dx, dt, grid_size):

    tau = np.zeros((grid_size, grid_size))

    nsteps = 1000000

    adsorption_times = []

    for n in range(10):
        grid = np.random.rand(3, grid_size, grid_size) * (1/3)
        for i in range(nsteps):

            # move one step forward in the simulation, updating phi at every point.
            grid, tau = update_grid(grid, D, q, p, dx, dt, grid_size)

            frac_a = np.sum(tau == 1) / grid_size**2
            frac_b = np.sum(tau == 2) / (grid_size**2)
            frac_c = np.sum(tau == 3) / (grid_size**2)

            if frac_a == 1 or frac_b == 1 or frac_c == 1:
                # if reached as=dsorption, record time and stop.
                adsorption_times.append(i * dt)
                print(i * dt)
                break

            if (i * dt) >= 1000:
                # if taking too long, disregard and run again.
                n -= 1
                break

    print(np.mean(adsorption_times), np.std(adsorption_times))


def sine_fit(x, a, b, c, d):
    return a * np.sin((x * b) + d) + c


def taskd(grid, D, q, p, dx, dt, grid_size):

    nsteps = 500000

    point1 = [10, 43]
    point2 = [36, 1]

    found = False

    while found == False:
        grid = np.random.rand(3, grid_size, grid_size) * (1/3)
        point1_list = []
        point2_list = []

        point1_behaviour = []
        point2_behaviour = []
        times = []
        for i in range(nsteps):

            # move one step forward in the simulation, updating phi at every point.
            grid, tau = update_grid(grid, D, q, p, dx, dt, grid_size)

            frac_a = np.sum(tau == 1) / grid_size**2
            frac_b = np.sum(tau == 2) / (grid_size**2)
            frac_c = np.sum(tau == 3) / (grid_size**2)

            point1_list.append(grid[0,point1[0],point1[1]])
            point2_list.append(grid[0,point2[0],point2[1]])
            point1_behaviour.append(tau[point1[0], point1[1]])
            point2_behaviour.append(tau[point2[0], point2[1]])
            times.append(i * dt)

            if frac_a == 1 or frac_b == 1 or frac_c == 1:
                print(i*dt)
                break

            if (i * dt) >= 300:
                found = True
                break

    times = np.array(times)
    point1_list = np.array(point1_list)
    point2_list = np.array(point2_list)

    fig = plt.figure()
    ax1 = fig.add_subplot(411)
    popt, pcov = curve_fit(sine_fit, times, point1_list, p0=[0.8, 1/5, 2, 10])
    print(popt)
    ax1.plot(times, sine_fit(times, *popt))
    ax1.plot(times, point1_list)
    ax1.set_xlabel("time/s")
    ax1.set_ylabel("value of a field")

    ax2 = fig.add_subplot(412)
    popt, pcov = curve_fit(sine_fit, times, point2_list, p0=[0.8, 1/5, 2, 10])
    print(popt)
    ax2.plot(times, sine_fit(times, *popt))
    ax2.plot(times, point2_list)
    ax2.set_xlabel("time/s")
    ax2.set_ylabel("value of a field")

    ax3 = fig.add_subplot(413)
    popt, pcov = curve_fit(sine_fit, times, point1_behaviour, p0=[0.8, 1/5, 2, 10])
    print(popt)
    ax3.plot(times, sine_fit(times, *popt))
    ax3.plot(times, point1_behaviour)
    ax3.set_xlabel("time/s")
    ax3.set_ylabel("value of tau field")

    ax4 = fig.add_subplot(414)
    popt, pcov = curve_fit(sine_fit, times, point2_behaviour, p0=[0.8, 1/5, 2, 10])
    print(popt)
    ax4.plot(times, sine_fit(times, *popt))
    ax4.plot(times, point2_behaviour)
    ax4.set_xlabel("time/s")
    ax4.set_ylabel("value of tau field")

    plt.show()


def taskf(grid, D, q, p, dx, dt, grid_size):

    nsteps = 20000
    measure_every = 100

    probs = []
    distances = []

    for iteration in range(nsteps):

        grid, tau = update_grid(grid, D, q, p, dx, dt, grid_size)

        if iteration % measure_every == 0:
            for row in tau:
                point = 25
                point_type = row[point]
                for j, cell in enumerate(row):
                    if cell == point_type:
                        probs.append(1)
                    else:
                        probs.append(0)
                    distances.append(abs(j - point))

    probs = np.array(probs)
    distances = np.array(distances)

    r_array = set(distances)
    reduced_probs = np.zeros(len(set(distances)))
    for i, distance in enumerate(r_array):
        av = np.average(probs[distances == distance])
        reduced_probs[distance] = av

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(list(r_array), reduced_probs)
    ax1.set_xlabel("Distance from point")
    ax1.set_ylabel("Probability of having same tau")

    plt.show()



def main():
    """Evaluate command line args to choose a function.
    """

    mode = sys.argv[1]

    grid_size = 50
    dx = 1

    # choose the simulation parameters.
    q = 1
    p = 0.5
    D = 0.5

    # a = grid[:,:,0], b= grid[:,:,1] etc
    initial_grid = np.random.rand(3, grid_size, grid_size) * (1/3)

    dt = float(sys.argv[2])
    if mode == "vis":
        animation(initial_grid, D, q, p, dx, dt, grid_size)
    elif mode == "2":
        taskb(initial_grid, D, q, p, dx, dt, grid_size)
    elif mode == "3":
        taskc(initial_grid, D, q, p, dx, dt, grid_size)
    elif mode == "4":
        taskd(initial_grid, D, q, p, dx, dt, grid_size)
    elif mode == "5":
        taskf(initial_grid, D, q, p, dx, dt, grid_size)


if __name__=="__main__":
    main()