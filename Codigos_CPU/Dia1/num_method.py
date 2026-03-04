import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def flujo_godunov(qL, qR):
    """
    Flujo de Godunov exacto para Burgers.
    Resuelve el Riemann exacto en la interfaz.
    """
    if qL <= qR:          # rarefacción
        if qL <= 0.0 <= qR:
            q_star = 0.0
        elif qL > 0.0:
            q_star = qL
        else:
            q_star = qR
    else:                # choque
        if 0.5 * (qL + qR) > 0.0:
            q_star = qL
        else:
            q_star = qR

    return 0.5 * q_star**2


def flujo_roe(qL, qR):
    """
    Flujo de Roe para la ecuación de Burgers:
        q_t + (q^2/2)_x = 0
    """

    fL = 0.5 * qL**2
    fR = 0.5 * qR**2

    a_tilde = 0.5 * (qL + qR)
    F = 0.5 * (fL + fR) - 0.5 * abs(a_tilde) * (qR - qL)

    return F

def flujo_hll(qL, qR):
    """
    Flujo HLL para Burgers:
        q_t + (q^2/2)_x = 0

    Para Burgers, una elección estándar de velocidades de onda es:
        S_L = min(qL, qR)
        S_R = max(qL, qR)
    porque la velocidad característica es f'(q)=q.
    """
    fL = 0.5 * qL**2
    fR = 0.5 * qR**2

    SL = min(qL, qR)
    SR = max(qL, qR)

    if 0.0 <= SL:
        return fL
    elif SR <= 0.0:
        return fR
    else:
        return (SR * fL - SL * fR + SL * SR * (qR - qL)) / (SR - SL)

def flujo_lax_friedrichs(qL, qR, alpha):
    """
    Flujo de Lax-Friedrichs para Burgers:
        q_t + (q^2/2)_x = 0
    alpha >= max |f'(q)| = |q|
    """
    fL = 0.5 * qL**2
    fR = 0.5 * qR**2
    return 0.5 * (fL + fR) - 0.5 * alpha * (qR - qL)



def metodo_step(Q, dt, dx, bc="periodic", flux_method="godunov"):
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

    F = np.zeros(N + 1)
    for i in range(N + 1):
        if flux_method == "godunov":
            F[i] = flujo_godunov(Qg[i], Qg[i + 1])
        elif flux_method == "roe":
            F[i] = flujo_roe(Qg[i], Qg[i + 1])
        elif flux_method == "hll":
            F[i] = flujo_hll(Qg[i], Qg[i + 1])
        elif flux_method == "lax-friedrichs":
            # clasico alpha = \dx/dt, pero es muy difusivo. Mejor usar una estimación de max|q|
            alpha = max(abs(Qg[i]), abs(Qg[i + 1])) # para f convexa.
            alpha = np.max(np.abs(Q))  # estimación de max|q| global, tambien hay versión local
            F[i] = flujo_lax_friedrichs(Qg[i], Qg[i + 1], alpha)
        else:
            raise ValueError("flux_method debe ser 'godunov', 'roe', 'hll', o 'lax-friedrichs'")

    Qnew = Qg[1:-1] - lam * (F[1:] - F[:-1])
    return Qnew

L=1.0
N=500
CFL=0.8
T=0.2
bc="outflow"

x = np.linspace(0.0, L, N, endpoint=False)
dx = L / N

# Condición inicial
# Q0 = np.exp(-((x - 0.25 * L) / (0.07 * L))**2)
# Q0 += 0.5 * (x > 0.6 * L)
# Q0 = 1.0*(x <= 0.5 * L) + 0.2 * (x > 0.5 * L)
Q0 = 4.0*(x <= 0.3 * L) +0.1 * ((x > 0.3 * L) & (x <= 0.6 * L)) + 1.0 * (x > 0.6 * L)
# Q0 = 0.1*(x <= 0.3 * L) + 3.0 * ((x > 0.3 * L) & (x <= 0.6 * L)) - 2.0 * (x > 0.6 * L)


Q = Q0.copy()
t = 0.0
flux_method="roe"  # 'godunov', 'roe', 'hll', 'lax-friedrichs'



plt.ion()
fig, ax = plt.subplots(figsize=(8,4))
line0, = ax.plot(x, Q0, "--", lw=1.5, label="Inicial")
line,  = ax.plot(x, Q,  lw=2, label=flux_method)
txt = ax.text(0.02, 0.92, "", transform=ax.transAxes)

ax.set_xlabel("x"); ax.set_ylabel("q(x,t)")
ax.set_title(f"Eq. Burgers - {flux_method}")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plot_every = 5
k = 0

while t < T:
    max_speed = np.max(np.abs(Q))
    dt = CFL * dx / max(max_speed, 1e-14)
    if t + dt > T:
        dt = T - t

    Q = metodo_step(Q, dt, dx, bc=bc, flux_method=flux_method)
    t = t + dt
    k = k + 1

    if k % plot_every == 0 or t >= T:
        line.set_ydata(Q)
        txt.set_text(f"t = {t:.3f}   CFL = {max_speed*dt/dx:.2f}")
        ax.relim()
        ax.autoscale_view(scaley=True)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.1)

plt.ioff()
plt.show()

