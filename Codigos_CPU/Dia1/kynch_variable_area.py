import numpy as np
import matplotlib.pyplot as plt


u_inf   = 1.0 #0.0125
phi_max = 0.519852941
C_exp   = 2.0  
T_end  = 1.0
largo  = 0.3

# opt=0: área fija, opt=1: área variable
opt = 1

phi_hat = phi_max / (1.0 + C_exp)

def fbk(phi: float):
    """
    f(phi) = u_inf * phi * (1 - phi/phi_max)^C_exp  para 0<phi<phi_max,
    y 0 fuera de ese rango.
    """
    if phi <= 0.0 or phi >= phi_max:
        return 0.0
    return u_inf * phi * pow(1.0 - phi/phi_max, C_exp)


def Area(z: float, opt: int):
    """Área transversal: variable si opt=1, constante (=1) si opt=0."""
    pp = 1.0 / 6.0
    qq = 0.5
    if opt == 1:
        return ((pp + qq * z) / (pp + qq)) ** (1.0 / qq)
    elif opt == 0:
        return 1.0
    else:
        raise ValueError("opt debe ser 0 (fija) o 1 (variable)")


def CFL(phi, dz):
    cfl=0.8
    return cfl * dz / max(np.max(phi), 1e-14)


def Flujo(phi1: float, phi2: float):
    """
    Flujo tipo Godunov.
    """
    if phi1 < phi2:
        return min(fbk(phi1), fbk(phi2))
    elif (phi_hat - phi1) * (phi_hat - phi2) < 0.0:
        return fbk(phi_hat)
    else:
        return max(fbk(phi1), fbk(phi2))


def Method(nc: int, opt: int = 0):
    """
    Devuelve: phi (incluye ghost), dz, dt, Abound, Ac
    """
    # arreglos 0..nc+1
    phi = np.zeros(nc + 2, dtype=float)
    F   = np.zeros(nc + 2, dtype=float)

    phi[1:nc+1] = 0.175
    phi[0] = 0.0
    phi[nc+1] = 1.0

    dz = largo / float(nc)
    z_plot = np.linspace(-dz/2, largo + dz/2, nc + 2)

    Abound = np.zeros(nc + 1, dtype=float)
    Ac     = np.zeros(nc + 1, dtype=float)

    # puntos z para Abound(j): z = -j*dz + largo
    for j in range(1, nc + 1):
        z = -j * dz + largo
        Abound[j] = Area(z, opt)

    # puntos z para Ac(j): z = -(j-0.5)*dz + largo
    for j in range(1, nc + 1):
        z = -(j - 0.5) * dz + largo
        Ac[j] = Area(z, opt)

    Ac[0] = 1.0
    dt = CFL(phi, dz)

    t = 0.0

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot(np.flip(phi), z_plot, lw=2)     # x = concentración (phi), y = altura (z)
    ax.set_xlabel("Concentración (phi)")
    ax.set_ylabel("Altura (z)")
    ax.set_title("Perfil de concentración vs altura")
    ax.grid(True)

    plot_every = 1
    it = 0
    
    while t <= T_end:
        for j in range(1, nc + 1):
            F[j] = Flujo(phi[j], phi[j + 1])

        phi[0]    = 0.0
        phi[nc+1] = phi_max

        F[0]      = 0.0
        F[nc+1]   = 0.0

        # phi(j) = phi(j) - (1/Abound(j))*(dt/dz)*(Ac(j)*F(j)-Ac(j-1)*F(j-1))
        for j in range(1, nc + 1):
            phi[j] = phi[j] - (1.0 / Abound[j]) * (dt / dz) * (Ac[j] * F[j] - Ac[j - 1] * F[j - 1])

        t = t + dt
        it = it + 1

        if it % plot_every == 0:
            line.set_xdata(np.flip(phi))
            line.set_ydata(z_plot)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.1)

    plt.ioff()
    plt.show()

    return phi, dz, dt, Abound, Ac


nc = 500
print(nc)

phi, dz, dt, Abound, Ac = Method(nc, opt=opt)


