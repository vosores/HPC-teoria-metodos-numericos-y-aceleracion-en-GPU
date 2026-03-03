import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def upwind_step(Q, u, dt, dx, bc="periodic"):
    """
    Un paso de FV upwind.
    Q: array (N,)
    bc: 'periodic' o 'outflow'
    """
    N = Q.size
    lam = dt / dx

    Qg = np.empty(N + 2)
    Qg[1:-1] = Q

    if bc == "periodic":
        Qg[0] = Q[-1]
        Qg[-1] = Q[0]
    elif bc == "outflow":
        Qg[0] = Q[0]
        Qg[-1] = Q[-1]
    else:
        raise ValueError("bc debe ser 'periodic' o 'outflow'")

    # Flujos en interfaces i+1/2 para i=0..N (sobre Qg)
    # F_{i+1/2} usa Q_i si u>0, usa Q_{i+1} si u<0
    if u > 0:
        F = u * Qg[:-1]
    else:
        F = u * Qg[1:]
    # Q_i^{n+1} = Q_i^n - lam (F_{i+1/2} - F_{i-1/2})
    Qn = Qg[1:-1] - lam * (F[1:] - F[:-1])
    return Qn


def simulate_upwind(u=1.0, L=1.0, N=400, CFL=0.9, T=1.0, bc="periodic"):
    x = np.linspace(0.0, L, N, endpoint=False)
    dx = L / N
    dt = CFL * dx / max(abs(u), 1e-14)
    nsteps = int(np.ceil(T / dt))
    dt = T / nsteps  # esto puede mejorar para que llegue exacto a T

    Q0 = np.exp(-((x - 0.25 * L) / (0.07 * L))**2)
    Q0 = Q0 + 0.5 * (x > 0.6 * L)  # salto (para ver difusión numérica)

    Q = Q0.copy()
    return x, Q0, Q, dx, dt, nsteps, bc


# Parámetros
u = 1.0
L = 1.0
N = 400
CFL = 0.9
T = 1.0
bc = "periodic"   # 'periodic' o 'outflow'

x, Q0, Q, dx, dt, nsteps, bc = simulate_upwind(u=u, L=L, N=N, CFL=CFL, T=T, bc=bc)

# Figura
fig, ax = plt.subplots()
ax.set_title("Advección 1D - FVM Upwind")
ax.set_xlabel("x")
ax.set_ylabel("q(x,t)")
ax.set_xlim(x.min(), x.max())
ax.set_ylim(min(Q0.min(), Q.min()) - 0.2, max(Q0.max(), Q.max()) + 0.2)
ax.grid(True, alpha=0.25)

(line,) = ax.plot(x, Q, lw=2, label="Upwind")
ax.plot(x, Q0, lw=1, alpha=0.7, label="Inicial")
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

ax.legend(loc="upper right")

steps_per_frame = max(1, nsteps // 300)

def init():
    line.set_ydata(Q)
    time_text.set_text("t = 0.000")
    return line, time_text

state = {"Q": Q.copy(), "n": 0}

def update(_frame):
    Qcur = state["Q"]
    ncur = state["n"]

    # avanzar steps_per_frame pasos
    for _ in range(steps_per_frame):
        if ncur >= nsteps:
            break
        Qcur = upwind_step(Qcur, u=u, dt=dt, dx=dx, bc=bc)
        ncur += 1

    state["Q"] = Qcur
    state["n"] = ncur

    tcur = ncur * dt
    line.set_ydata(Qcur)
    time_text.set_text(f"t = {tcur:.3f}  |  CFL = {abs(u)*dt/dx:.2f}")
    return line, time_text

anim = FuncAnimation(fig, update, init_func=init, interval=30, blit=True)
plt.show()