import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def flujo_godunov(qL, qR):
    """
    Flujo de Godunov exacto para Burgers:
      f(q)=q^2/2
    usando el estado interface q* del Riemann exacto.
    """
    # Selección de q*
    if qL <= qR:
        if qL <= 0.0 <= qR:
            q_star = 0.0
        elif qL > 0.0:
            q_star = qL
        else:
            q_star = qR
    else:               
        if 0.5 * (qL + qR) > 0.0:
            q_star = qL
        else:
            q_star = qR

    return 0.5 * q_star**2


def metodo_step_burgers(Q, dt, dx, bc="periodic"):
    """
    Un paso FV de Godunov (orden 1) para Burgers.
    Q: array (N,)
    bc: 'periodic' o 'outflow'
    """
    N = Q.size
    lam = dt / dx

    # Ghost cells
    Qg = np.empty(N + 2)
    Qg[1:-1] = Q

    if bc == "periodic":
        Qg[0]  = Q[-1]
        Qg[-1] = Q[0]
    elif bc == "outflow":
        Qg[0]  = Q[0]
        Qg[-1] = Q[-1]
    else:
        raise ValueError("bc debe ser 'periodic' o 'outflow'")

    # Flujos en interfaces i+1/2 para i=0..N (sobre Qg)
    F = np.zeros(N + 1)
    for k in range(N + 1):
        qL = Qg[k]
        qR = Qg[k + 1]
        F[k] = flujo_godunov(qL, qR)

    Qn = Qg[1:-1] - lam * (F[1:] - F[:-1])
    return Qn


def godunov_burgers(L=1.0, N=400, CFL=0.9, T=0.4, bc="periodic"):
    x = np.linspace(0.0, L, N, endpoint=False)
    dx = L / N

    Q0 = np.exp(-((x - 0.25 * L) / (0.07 * L))**2)
    # Q0 = 0.2 * (x < 0.4 * L) + 0.5 * (x > 0.6 * L)

    # dt por CFL usando max|q| (velocidad característica = q)
    max_speed0 = np.max(np.abs(Q0))
    dt = CFL * dx / max(max_speed0, 1e-14)
    nsteps = int(np.ceil(T / dt))
    dt = T / nsteps  # se puede ajustar para llegar exacto a T

    Q = Q0.copy()
    return x, Q0, Q, dx, dt, nsteps, bc


L = 1.0
N = 400
CFL = 0.9
T = 0.6
bc = "outflow"

x, Q0, Q, dx, dt, nsteps, bc = godunov_burgers(L=L, N=N, CFL=CFL, T=T, bc=bc)

fig, ax = plt.subplots()
ax.set_title("Burgers 1D - Godunov FVM")
ax.set_xlabel("x")
ax.set_ylabel("q(x,t)")
ax.set_xlim(x.min(), x.max())
ax.set_ylim(Q0.min() - 0.3, Q0.max() + 0.3)
ax.grid(True, alpha=0.25)

(line,) = ax.plot(x, Q, lw=2, label="Godunov")
ax.plot(x, Q0, lw=1, alpha=0.7, label="Inicial")
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
ax.legend(loc="upper right")

steps_per_frame = max(1, nsteps // 300)

state = {"Q": Q.copy(), "n": 0}

def init():
    line.set_ydata(state["Q"])
    time_text.set_text("t = 0.000")
    return line, time_text

def update(_frame):
    Qcur = state["Q"]
    ncur = state["n"]

    for _ in range(steps_per_frame):
        if ncur >= nsteps:
            break

        # max_speed = np.max(np.abs(Qcur))
        # dt_eff = min(dt, CFL*dx/max(max_speed,1e-14))
        # Qcur = metodo_step_burgers(Qcur, dt_eff, dx, bc=bc)

        Qcur = metodo_step_burgers(Qcur, dt, dx, bc=bc)
        ncur = ncur + 1

    state["Q"] = Qcur
    state["n"] = ncur

    tcur = ncur * dt
    cfl_now = (np.max(np.abs(Qcur)) * dt / dx) if np.max(np.abs(Qcur)) > 1e-14 else 0.0
    line.set_ydata(Qcur)
    time_text.set_text(f"t = {tcur:.3f}  |  CFL ≈ {cfl_now:.2f}")
    return line, time_text

anim = FuncAnimation(fig, update, init_func=init, interval=30, blit=True)
plt.show()