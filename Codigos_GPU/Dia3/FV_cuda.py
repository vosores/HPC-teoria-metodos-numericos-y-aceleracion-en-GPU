import numpy as np
from numba import cuda, float32
import matplotlib.pyplot as plt
import time

# ============================================
# Ghost cells (ng=1): u size = n + 2
# physical: 1..n, ghosts: 0 and n+1
# ============================================

@cuda.jit
def apply_bc_outflow_ghost(u, n_phys):
    i = cuda.grid(1)
    if i == 0:
        u[0] = u[1]
    elif i == 1:
        u[n_phys + 1] = u[n_phys]

@cuda.jit
def reduce_max_abs(x, out, start, ncount):
    sm = cuda.shared.array(512, dtype=float32)
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x
    bid  = cuda.blockIdx.x

    k = bid * (2 * bdim) + tid

    m = float32(0.0)

    i = start + k
    if k < ncount:
        m = abs(x[i])

    k2 = k + bdim
    j = start + k2
    if k2 < ncount:
        v = abs(x[j])
        if v > m:
            m = v

    sm[tid] = m
    cuda.syncthreads()

    stride = bdim // 2
    while stride > 0:
        if tid < stride:
            v = sm[tid + stride]
            if v > sm[tid]:
                sm[tid] = v
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        out[bid] = sm[0]


def max_abs_device(x_dev, start=0, ncount=None, tpb=256):
    if ncount is None:
        ncount = x_dev.size - start

    blocks = (ncount + (2 * tpb) - 1) // (2 * tpb)
    curr = cuda.device_array(shape=blocks, dtype=np.float32)

    reduce_max_abs[blocks, tpb](x_dev, curr, start, ncount)

    curr_n = blocks
    while curr_n > 1:
        blocks = (curr_n + (2 * tpb) - 1) // (2 * tpb)
        nxt = cuda.device_array(shape=blocks, dtype=np.float32)
        reduce_max_abs[blocks, tpb](curr, nxt, 0, curr_n)
        curr = nxt
        curr_n = blocks

    cuda.synchronize()
    return float(curr.copy_to_host()[0])


# ============================================================
# 2) FV Burgers 1D: Rusanov / Roe / HLL / Lax-F
#    f(u) = 0.5 u^2,  a(u)=f'(u)=u
# ============================================================

M_RUSANOV = 0
M_ROE     = 1
M_HLL     = 2
M_LF      = 3   # Lax-Friedrichs global (usa dx/dt)

@cuda.jit(device=True, inline=True)
def f_burgers(u):
    return 0.5 * u * u

@cuda.jit(device=True, inline=True)
def absf(a):
    return a if a >= 0.0 else -a

@cuda.jit(device=True, inline=True)
def flux_num_burgers(uL, uR, method, alpha_lf):
    """
    Devuelve F = F(uL,uR) según method.
    alpha_lf = dx/dt (solo usado en Lax-F global).
    """
    fL = f_burgers(uL)
    fR = f_burgers(uR)

    if method == M_RUSANOV:
        # alpha = max(|uL|,|uR|)
        aL = absf(uL)
        aR = absf(uR)
        alpha = aL if aL > aR else aR
        return 0.5*(fL + fR) - 0.5*alpha*(uR - uL)

    elif method == M_ROE:
        # Roe escalar Burgers: a~ = (uL+uR)/2
        a = 0.5*(uL + uR)
        alpha = absf(a)
        return 0.5*(fL + fR) - 0.5*alpha*(uR - uL)

    elif method == M_HLL:
        SL = uL if uL < uR else uR
        SR = uR if uR > uL else uL

        if 0.0 <= SL:
            return fL
        elif 0.0 >= SR:
            return fR
        else:
            # HLL
            return (SR*fL - SL*fR + SL*SR*(uR - uL)) / (SR - SL)

    else:
        # Lax-Friedrichs global:
        # F = 0.5(fL+fR) - 0.5*(dx/dt)*(uR-uL)
        return 0.5*(fL + fR) - 0.5*alpha_lf*(uR - uL)


@cuda.jit
def fv_burgers(u, unew, dx, dt, n_phys, method):
    """
    u/unew size = n_phys + 2
    actualiza celdas físicas i=1..n_phys
    Usa flujos numéricos en i±1/2.
    method: int (M_RUSANOV, M_ROE, M_HLL, M_LF)
    """
    i = cuda.grid(1) + 1  # i=1..n_phys
    if i <= n_phys:
        uim1 = u[i - 1]
        ui   = u[i]
        uip1 = u[i + 1]

        # para Lax-F global: alpha = dx/dt
        alpha_lf = dx / dt

        FR = flux_num_burgers(ui,   uip1, method, alpha_lf)
        FL = flux_num_burgers(uim1, ui,   method, alpha_lf)

        unew[i] = ui - (dt / dx) * (FR - FL)


def solve_burgers_vf_gpu(u0_phys, dx, tfinal, cfl=0.9, tpb=256, method="rusanov"):
    """
    method: "rusanov", "roe", "hll", "laxf" (Lax-F global)
    """
    method_map = {
        "rusanov": M_RUSANOV,
        "roe":     M_ROE,
        "hll":     M_HLL,
        "laxf":    M_LF,
        "lf":      M_LF,
    }
    if method not in method_map:
        raise ValueError(f"method inválido: {method}. Usa: {list(method_map.keys())}")

    u0_phys = np.asarray(u0_phys, dtype=np.float32)
    n = u0_phys.size
    if n < 3:
        raise ValueError("Necesitas al menos 3 celdas físicas.")

    u0 = np.empty(n + 2, dtype=np.float32)
    u0[1:n+1] = u0_phys
    u0[0]     = u0[1]      # outflow
    u0[n+1]   = u0[n]

    d_u    = cuda.to_device(u0)
    d_unew = cuda.device_array_like(d_u)

    blocks = (n + tpb - 1) // tpb
    method_code = method_map[method]

    t = 0.0
    while t < tfinal:
        apply_bc_outflow_ghost[1, 2](d_u, n)
        amax = max_abs_device(d_u, start=1, ncount=n, tpb=tpb)

        if amax < 1e-12:
            dt = min(1e-6, tfinal - t)
        else:
            dt = min(cfl * dx / amax, tfinal - t)

        fv_burgers[blocks, tpb](d_u, d_unew, dx, dt, n, method_code)
        apply_bc_outflow_ghost[1, 2](d_unew, n)

        # swap
        d_u, d_unew = d_unew, d_u
        t = t + dt

    uT = d_u.copy_to_host()
    return uT[1:n+1]

# Uso
t0 = time.perf_counter()

n = 4096*6
L = 3.0
x = np.linspace(0, L, n, dtype=np.float32)

A, x0, sigma = 1.0, 0.5, 0.05
u0 = (A * np.exp(-0.5*((x - x0)/sigma)**2)).astype(np.float32)

dx = float(x[1] - x[0])
tfinal = 15.0

# Métodos disponibles: "rusanov", "roe", "hll", "laxf"
uT = solve_burgers_vf_gpu(u0, dx, tfinal, cfl=0.9, tpb=256, method="hll")

print("max|u| final:", np.max(np.abs(uT)))
t1 = time.perf_counter()
print("Tiempo GPU (s):", t1 - t0)

step = max(1, uT.shape[0] // 4000)
xP = x[::step]

plt.figure(figsize=(9, 4))
plt.plot(xP, u0[::step], label="u0")
plt.plot(xP, uT[::step], label="u(tfinal)")
plt.xlabel("x")
plt.ylabel("u")
plt.title("Burgers 1D (FV + método seleccionable, ng=1)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()