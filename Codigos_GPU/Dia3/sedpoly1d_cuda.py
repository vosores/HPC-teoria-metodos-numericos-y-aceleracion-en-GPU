import numpy as np
from numba import cuda, float32, int32
import matplotlib.pyplot as plt
import time

# ============================================================
# GPU version: Polydisperse sedimentation 1D
# phi shape: (S, ncells+1)
# F   shape: (S, ncells+2)
#
# Aclaraciones:
# - El código GPU es más largo que el CPU porque hay que escribir explícitamente los kernels y 
#   las funciones device, y también hay que manejar la transferencia de datos entre host y device.
# - El codigo está lejos de ser óptimo, es una traducción directa de CPU a GPU. Hay muchas formas de optimizarlo, pero 
#   lo importante es que funcione primero.
# - El código GPU asume que el número de especies S es pequeño (<= MAX_SPECIES) y lo fija como una constante para poder 
#   usar arrays locales en los kernels. Si quieres manejar más especies, hay que cambiar la estrategia para manejar los arrays locales.
# ============================================================

MAX_SPECIES = 8

@cuda.jit(device=True, inline=True)
def absf(a):
    return a if a >= 0.0 else -a

@cuda.jit(device=True, inline=True)
def signnz(x):
    return 1.0 if x == 0.0 else (1.0 if x > 0.0 else -1.0)

@cuda.jit(device=True, inline=True)
def vMLB_dev(j, v, d, d_f, diam, num_especies, g):
    """
    j: índice especie
    v: vector local de tamaño MAX_SPECIES con v[0..S-1] válido
    d, diam: arrays length>=S
    """

    u_0 = 0.02416
    lam = 4.7
    phi_max = 0.6

    # sum(v) y min(v)
    s = 0.0
    vmin = 1e20
    for k in range(num_especies):
        vk = v[k]
        s += vk
        if vk < vmin:
            vmin = vk

    # Factor de dificultad
    if (vmin > 0.0) and (s < phi_max):
        v_f = (1.0 - s) ** (lam - 2.0)
    else:
        v_f = 0.0

    # vectrho = d - d_f
    # dot = vectrho * v
    # dot2 = (diam^2/diam0^2) * v * (vectrho - sum(dot))
    dot_sum = 0.0
    for k in range(num_especies):
        dot_sum += (d[k] - d_f) * v[k]

    diam0 = diam[0]
    sum_dot2 = 0.0
    for k in range(num_especies):
        vectrho_k = (d[k] - d_f)
        ratio = (diam[k] * diam[k]) / (diam0 * diam0)
        sum_dot2 += ratio * v[k] * (vectrho_k - dot_sum)

    vectrho_j = (d[j] - d_f)
    ratio_j = (diam[j] * diam[j]) / (diam0 * diam0)

    val = (-g * diam0) / (18.0 * u_0) * v_f * (
        ratio_j * (vectrho_j - dot_sum) - sum_dot2
    )
    return float32(val)

@cuda.jit(device=True, inline=True)
def flujovert_dev(j, phil, phir, d, d_f, diam, num_especies, g):
    # term1 = 0.5*(phil[j]*vMLB(j,phil)+phir[j]*vMLB(j,phir))
    v_l = vMLB_dev(j, phil, d, d_f, diam, num_especies, g)
    v_r = vMLB_dev(j, phir, d, d_f, diam, num_especies, g)

    term1 = 0.5 * (phil[j] * v_l + phir[j] * v_r)
    term2 = -0.5 * absf(v_r) * (phir[j] - phil[j])
    term3 = -0.5 * phil[j] * absf(v_l - v_r) * signnz(phir[j] - phil[j])

    return float32(term1 + term2 + term3)

@cuda.jit
def compute_F_MLB(phi, F, ncells, num_especies, d, diam, g, d_f, tF):
    """
    phi: (S, ncells+1)
    F:   (S, ncells+2)
    Interfaces:
      i=0 -> F[:,0]=0
      i=1..ncells -> usa phil=phi[:,i-1], phir=phi[:,i]
      i=ncells+1 -> usa phil=0, phir=0
    """
    i = cuda.grid(1)
    if i > ncells + 1:
        return

    # interfaz 0: flujo 0
    if i == 0:
        for j in range(num_especies):
            F[j, 0] = 0.0
        return

    phil = cuda.local.array(MAX_SPECIES, dtype=float32)
    phir = cuda.local.array(MAX_SPECIES, dtype=float32)

    if i <= ncells:
        for k in range(num_especies):
            phil[k] = phi[k, i - 1]
            phir[k] = phi[k, i]
    else:
        # i == ncells+1
        for k in range(num_especies):
            phil[k] = 0.0
            phir[k] = 0.0

    # calcular h[j] para cada especie y asignar F[:,i]
    if tF == 0:
        for j in range(num_especies):
            y = vMLB_dev(j, phil, d, d_f, diam, num_especies, g)
            # dd = [0, y] => max/min
            if y >= 0.0:
                F[j, i] = phil[j] * y + phir[j] * 0.0
            else:
                F[j, i] = phil[j] * 0.0 + phir[j] * y

    elif tF == 1:
        for j in range(num_especies):
            v_r = vMLB_dev(j, phir, d, d_f, diam, num_especies, g)
            v_l = vMLB_dev(j, phil, d, d_f, diam, num_especies, g)
            F[j, i] = (0.5 * (phir[j] * v_r + phil[j] * v_l)
                       - absf(v_r) * (phir[j] - phil[j])
                       - 0.5 * phil[j] * absf(v_l - v_r) * signnz(phir[j] - phil[j]))

    else:
        for j in range(num_especies):
            F[j, i] = flujovert_dev(j, phil, phir, d, d_f, diam, num_especies, g)

@cuda.jit
def fv_update_phi(phi, F, ncells, num_especies, dt, dx):
    """
    phi: (S, ncells+1)
    F:   (S, ncells+2)
    Update: phi[j,i] -= (dt/dx)*(F[j,i+1]-F[j,i])
    i=0..ncells
    """
    j, i = cuda.grid(2)  # especie, celda
    if j < num_especies and i <= ncells:
        phi[j, i] = phi[j, i] - (dt / dx) * (F[j, i + 1] - F[j, i])


# (host)
def solve_gpu(phi0, x, tf, dt0, dtmax, dts,
                  diam, d, d_f, g=9.81, tF=2, tpb=256,
                  plot_every=2000, do_plot=True):
    """
    phi0: (S, ncells+1) float32/float64
    x: (ncells+1)
    """
    phi0 = np.asarray(phi0, dtype=np.float32)
    diam = np.asarray(diam, dtype=np.float32)
    d    = np.asarray(d, dtype=np.float32)
    d_f  = np.float32(d_f)
    g    = np.float32(g)

    S, npts = phi0.shape
    ncells = npts - 1
    if S > MAX_SPECIES:
        raise ValueError(f"num_especies={S} excede MAX_SPECIES={MAX_SPECIES}. Súbelo.")
    if ncells < 1:
        raise ValueError("ncells debe ser >= 1")

    dx = np.float32(x[1] - x[0])

    d_phi  = cuda.to_device(phi0)
    d_F    = cuda.device_array((S, ncells + 2), dtype=np.float32)
    d_d    = cuda.to_device(d)
    d_diam = cuda.to_device(diam)

    blocks_F = (ncells + 2 + tpb - 1) // tpb
    threads_F = tpb

    TPBX, TPBY = 16, 16
    blocks_u = ((S + TPBX - 1) // TPBX, (ncells + 1 + TPBY - 1) // TPBY)
    threads_u = (TPBX, TPBY)

    if do_plot:
        z = x.astype(np.float32)
        plt.ion()
        fig, ax = plt.subplots()
        lines = []
        phi_h = phi0.copy()
        for k in range(S):
            (ln,) = ax.plot(phi_h[k, :], z, label=f"phi{k+1}")
            lines.append(ln)
        ax.set_xlabel("Concentración (phi)")
        ax.set_ylabel("Altura (z)")
        ax.set_title("Sedimentación (GPU): concentración vs altura")
        ax.legend(loc="best")
        ax.set_xlim(0.0, max(1e-6, 1.2 * np.max(phi_h)))
        ax.set_ylim(z.min(), z.max())
        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
        fig.canvas.draw()
        fig.canvas.flush_events()

    tiempo = 0.0
    ts = dts
    n_iter = 1

    dt = float(dt0)
    dtmax = float(dtmax)

    while tiempo < tf:
        compute_F_MLB[blocks_F, threads_F](d_phi, d_F, ncells, S, d_d, d_diam, g, d_f, int32(tF))
        fv_update_phi[blocks_u, threads_u](d_phi, d_F, ncells, S, np.float32(dt), dx)

        dt = 0.9 * float(dx) # aquí se debe mejorar la cfl
        dt = min(dt, dtmax)
        dt = min(dt, ts - tiempo, tf - tiempo)
        tiempo += dt

        if do_plot and (n_iter % plot_every) == 0:
            phi_h = d_phi.copy_to_host()
            for k in range(S):
                lines[k].set_xdata(phi_h[k, :])
            ax.set_xlim(0.0, max(1e-6, 1.2 * np.max(phi_h)))
            time_text.set_text(f"t = {tiempo:.4f}")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

        # snapshot cada dts
        if abs(tiempo - ts) < 1e-12 or abs(tiempo - tf) < 1e-12:
            print("Tiempo:", tiempo)
            ts = ts + dts
            if do_plot:
                phi_h = d_phi.copy_to_host()
                for k in range(S):
                    lines[k].set_xdata(phi_h[k, :])
                ax.set_xlim(0.0, max(1e-6, 1.2 * np.max(phi_h)))
                time_text.set_text(f"t = {tiempo:.4f}")
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)

        n_iter += 1

    phiT = d_phi.copy_to_host()
    if do_plot:
        plt.ioff()
        plt.show()
    return phiT



num_celdas =200
num_especies = 3

dtmax = 0.1
dt0   = 1.0e-5
tf    = 100.0
dts   = 50.0

LL = 1.0
dx = LL / num_celdas
g = 9.81

diam = np.array([4.96e-4, 3.25e-4, 1.0e-4], dtype=np.float32)
d    = (1.0 / 1208.0) * np.array([2790.0, 2790.0, 2790.0], dtype=np.float32)
df   = np.float32(1208.0 / 1208.0)

x = np.array([i * dx for i in range(num_celdas + 1)], dtype=np.float32)
phi0 = np.zeros((num_especies, num_celdas + 1), dtype=np.float32)
phi0[0, :] = 0.1
phi0[1, :] = 0.05
phi0[2, :] = 0.09

t0 = time.perf_counter()
phiT = solve_gpu(phi0, x, tf=tf, dt0=dt0, dtmax=dtmax, dts=dts,
                        diam=diam, d=d, d_f=df, g=g, tF=2, tpb=256,
                        plot_every=2000, do_plot=True)
t1 = time.perf_counter()
print("Tiempo total (GPU loop + copias para plot) [s]:", t1 - t0) # el copiar a host para plotear es lo que más tarda, no el loop GPU en sí. Si quieres medir solo el loop GPU, hazlo sin plotear o con plot_every muy grande.
print("phi final max:", np.max(phiT))

