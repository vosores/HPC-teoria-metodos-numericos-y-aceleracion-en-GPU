import numpy as np
import matplotlib.pyplot as plt

def vMLB(j, v, d, d_f, diam, num_especies, g):
    u_0 = 0.02416
    lam = 4.7
    phi_max = 0.6

    # Factor de dificultad de sedimentar
    if (np.min(v) > 0.0) and (np.sum(v) < phi_max):
        v_f = (1.0 - np.sum(v)) ** (lam - 2.0)
    else:
        v_f = 0.0

    vectrho = d - d_f

    dot = vectrho * v
    dot2 = (diam**2 / diam[0]**2) * v * (vectrho - np.sum(dot))

    # vMLB = (-g*diam(1))/(18*u_0)*v_f*( (diam(j)^2/diam(1)^2)*(vectrho(j)-sum(dot)) - sum(dot2) )
    val = (-g * diam[0]) / (18.0 * u_0) * v_f * (
        (diam[j]**2 / diam[0]**2) * (vectrho[j] - np.sum(dot)) - np.sum(dot2)
    )
    return float(val)


def flujovert(j, phil, phir, num_especies, d, d_f, diam, g):
    # V(i)=vMLB(i,phir,...)
    V = np.zeros(num_especies, dtype=float)
    for i in range(num_especies):
        V[i] = vMLB(i, phir, d, d_f, diam, num_especies, g)

    term1 = 0.5 * (phil[j] * vMLB(j, phil, d, d_f, diam, num_especies, g)
                   + phir[j] * vMLB(j, phir, d, d_f, diam, num_especies, g))
    term2 = -0.5 * abs(vMLB(j, phir, d, d_f, diam, num_especies, g)) * (phir[j] - phil[j])
    term3 = -0.5 * phil[j] * abs(
        vMLB(j, phil, d, d_f, diam, num_especies, g) - vMLB(j, phir, d, d_f, diam, num_especies, g)
    ) * np.sign(phir[j] - phil[j] if (phir[j] - phil[j]) != 0 else 1.0)

    return float(term1 + term2 + term3)

def solver(phil, phir, tF, num_especies, g, d, diam, df, num_capas=0):
    h = np.zeros(num_especies, dtype=float)
    z = 0.0

    if tF == 0:
        for i in range(num_especies):
            y = vMLB(i, phil, d, df, diam, num_especies, g)
            dd = np.array([z, y], dtype=float)
            h[i] = phil[i] * np.max(dd) + phir[i] * np.min(dd)

    elif tF == 1:
        for i in range(num_especies):
            v_r = vMLB(i, phir, d, df, diam, num_especies, g)
            v_l = vMLB(i, phil, d, df, diam, num_especies, g)

            h[i] = (0.5 * (phir[i] * v_r + phil[i] * v_l)
                    - abs(v_r) * (phir[i] - phil[i])
                    - (phil[i] / 2.0) * abs(v_l - v_r) * np.sign(phir[i] - phil[i] if (phir[i] - phil[i]) != 0 else 1.0))

    elif tF == 2:
        for i in range(num_especies):
            h[i] = flujovert(i, phil, phir, num_especies, d, df, diam, g)

    else:
        raise ValueError("tF debe ser 0, 1 o 2")

    return h

def Flujo(phi, num_celdas, num_especies, diam, d, g, df, tF=2):
    """
    phi shape: (num_especies, num_celdas+1)
    F   shape: (num_especies, num_celdas+2)
    """
    F = np.zeros((num_especies, num_celdas + 2), dtype=float)
    F[:, 0] = 0.0

    # Interfaces internas
    for i in range(1, num_celdas + 1):  # i=2..num_celdas+1 en Fortran
        phil = phi[:, i - 1].copy()
        phir = phi[:, i].copy()
        h = solver(phil, phir, tF, num_especies, g, d, diam, df)
        F[:, i] = h

    phil = np.zeros(num_especies, dtype=float)
    phir = np.zeros(num_especies, dtype=float)
    h = solver(phil, phir, tF, num_especies, g, d, diam, df)
    F[:, num_celdas + 1] = h

    return F


def metodo(phi, F, num_celdas, num_especies, dt, dx):
    for j in range(num_especies):
        for i in range(num_celdas + 1):
            phi[j, i] = phi[j, i] - (dt / dx) * (F[j, i + 1] - F[j, i])
    return phi

def save_lines_csv(path, x, phi):
    cols = ["Position[m]"] + [f"phi{k+1}" for k in range(phi.shape[0])]
    header = ",".join(cols)

    nwrite = min(len(x), phi.shape[1])
    nwrite = nwrite - 1

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for i in range(nwrite):
            row = [f"{x[i]:.8f}"] + [f"{phi[k, i]:.8f}" for k in range(phi.shape[0])]
            f.write(",".join(row) + "\n")


num_celdas = 50
num_especies = 2

dtmax = 0.1
dt = 1.0e-5
tf = 10.0
dts = 1.0

LL = 1.0
dx = LL / num_celdas
g = 9.81

diam = np.array([4.96e-4, 1.25e-4], dtype=float)
d = (1.0 / 1208.0) * np.array([2790.0, 2790.0], dtype=float)
df = 1208.0 / 1208.0 

x = np.array([i * dx for i in range(num_celdas + 1)], dtype=float)
phi = np.zeros((num_especies, num_celdas + 1), dtype=float)

# Condición inicial
phi[0, :] = 0.05
phi[1, :] = 0.01


plt.ion()
z = x  # altura

plt.ion()

fig, ax = plt.subplots()
lines = []
for k in range(num_especies):
    # X = concentración (phi), Y = altura (z)
    (ln,) = ax.plot(phi[k, :], z, label=f"phi{k+1}")
    lines.append(ln)

ax.set_xlabel("Concentración (phi)")
ax.set_ylabel("Altura (z)")
ax.set_title("Sedimentación: concentración vs altura")
ax.legend(loc="best")

ax.set_xlim(0.0, max(1e-6, 1.2 * np.max(phi)))
ax.set_ylim(z.min(), z.max())

time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

fig.canvas.draw()
fig.canvas.flush_events()

plot_every = 2000

tiempo = 0.0
ts = dts
n_iter = 1
n_iter_save = 1

while tiempo < tf:
    F = Flujo(phi, num_celdas, num_especies, diam, d, g, df, tF=2)
    phi = metodo(phi, F, num_celdas, num_especies, dt, dx)

    dt = 0.9 * dx # aquí se debe mejorar la cfl
    dt = min(dt, dtmax)
    dt = min(dt, ts - tiempo, tf - tiempo)
    tiempo = tiempo + dt

    if (n_iter % plot_every) == 0:
        for k in range(num_especies):
            lines[k].set_xdata(phi[k, :])

        ax.set_xlim(0.0, max(1e-6, 1.2 * np.max(phi)))
        time_text.set_text(f"t = {tiempo:.4f}")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)

    if abs(tiempo - ts) < 1e-12 or abs(tiempo - tf) < 1e-12:
        print("Tiempo :", tiempo)
        n_iter_save += 1
        ts += dts

        for k in range(num_especies):
            lines[k].set_xdata(phi[k, :])

        ax.set_xlim(0.0, max(1e-6, 1.2 * np.max(phi)))
        time_text.set_text(f"t = {tiempo:.4f}")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)

    n_iter += 1

plt.ioff()
plt.show()