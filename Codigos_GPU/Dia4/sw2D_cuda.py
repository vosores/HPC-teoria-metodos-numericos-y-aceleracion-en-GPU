import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import cuda, float32, int32
import math
import time

SOLVER_HLL  = 0
SOLVER_HLLE = 1
SOLVER_ROE  = 2

SOLVER = SOLVER_ROE

BC_WALL     = 0
BC_PERIODIC = 1

@cuda.jit(device=True, inline=True)
def absf(a):
    return a if a >= 0.0 else -a

@cuda.jit(device=True, inline=True)
def primitives_cell(h, hu, hv, dry_tol):
    if h > dry_tol:
        u = hu / h
        v = hv / h
        return u, v, True
    else:
        return 0.0, 0.0, False

@cuda.jit(device=True, inline=True)
def flux_x_cell(h, hu, hv, g, dry_tol):
    u, v, wet = primitives_cell(h, hu, hv, dry_tol)
    if not wet:
        return 0.0, 0.0, 0.0
    return hu, hu*u + 0.5*g*h*h, hu*v

@cuda.jit(device=True, inline=True)
def flux_y_cell(h, hu, hv, g, dry_tol):
    u, v, wet = primitives_cell(h, hu, hv, dry_tol)
    if not wet:
        return 0.0, 0.0, 0.0
    return hv, hv*u, hv*v + 0.5*g*h*h

@cuda.jit(device=True, inline=True)
def hll_flux_face(qL0,qL1,qL2, qR0,qR1,qR2, g, direction, dry_tol):
    """
    HLL across one face.
    direction=0 -> x-face (normal u), use Fx
    direction=1 -> y-face (normal v), use Fy
    returns (F0,F1,F2) in conserved ordering (h,hu,hv).
    """
    uL, vL, wetL = primitives_cell(qL0, qL1, qL2, dry_tol)
    uR, vR, wetR = primitives_cell(qR0, qR1, qR2, dry_tol)

    hL = qL0 if wetL else 0.0
    hR = qR0 if wetR else 0.0

    if direction == 0:
        unL = uL; unR = uR
        FL0,FL1,FL2 = flux_x_cell(qL0,qL1,qL2,g,dry_tol)
        FR0,FR1,FR2 = flux_x_cell(qR0,qR1,qR2,g,dry_tol)
    else:
        unL = vL; unR = vR
        FL0,FL1,FL2 = flux_y_cell(qL0,qL1,qL2,g,dry_tol)
        FR0,FR1,FR2 = flux_y_cell(qR0,qR1,qR2,g,dry_tol)

    cL = math.sqrt(g*hL) if hL > 0.0 else 0.0
    cR = math.sqrt(g*hR) if hR > 0.0 else 0.0

    s1 = min(unL - cL, unR - cR)
    s2 = max(unL + cL, unR + cR)

    if (s2 - s1) < 1e-14:
        a = max(absf(unL)+cL, absf(unR)+cR)
        s1 = -a
        s2 =  a

    if s1 >= 0.0:
        return FL0,FL1,FL2
    if s2 <= 0.0:
        return FR0,FR1,FR2

    denom = (s2 - s1)
    dq0 = qR0 - qL0
    dq1 = qR1 - qL1
    dq2 = qR2 - qL2

    F0 = (s2*FL0 - s1*FR0 + (s1*s2)*dq0) / denom
    F1 = (s2*FL1 - s1*FR1 + (s1*s2)*dq1) / denom
    F2 = (s2*FL2 - s1*FR2 + (s1*s2)*dq2) / denom
    return F0,F1,F2


@cuda.jit(device=True, inline=True)
def hlle_flux_face(qL0,qL1,qL2, qR0,qR1,qR2, g, direction, dry_tol):
    """
    HLL across one face.
    direction=0 -> x-face (normal u), use Fx
    direction=1 -> y-face (normal v), use Fy
    returns (F0,F1,F2) in conserved ordering (h,hu,hv).
    """
    uL, vL, wetL = primitives_cell(qL0, qL1, qL2, dry_tol)
    uR, vR, wetR = primitives_cell(qR0, qR1, qR2, dry_tol)

    hL = qL0 if wetL else 0.0
    hR = qR0 if wetR else 0.0

    if direction == 0:
        unL, unR = uL, uR
        FL0, FL1, FL2 = flux_x_cell(qL0, qL1, qL2, g, dry_tol)
        FR0, FR1, FR2 = flux_x_cell(qR0, qR1, qR2, g, dry_tol)
    else:
        unL, unR = vL, vR
        FL0, FL1, FL2 = flux_y_cell(qL0, qL1, qL2, g, dry_tol)
        FR0, FR1, FR2 = flux_y_cell(qR0, qR1, qR2, g, dry_tol)

    cL = math.sqrt(g*hL) if hL > 0.0 else 0.0
    cR = math.sqrt(g*hR) if hR > 0.0 else 0.0

    sqrt_hL = math.sqrt(hL) if hL > 0.0 else 0.0
    sqrt_hR = math.sqrt(hR) if hR > 0.0 else 0.0

    if (sqrt_hL + sqrt_hR) > 0.0:
        uhat_n = (sqrt_hL*unL + sqrt_hR*unR) / (sqrt_hL + sqrt_hR)
        hbar = 0.5*(hL + hR)
        chat = math.sqrt(g*hbar)
    else:
        uhat_n = 0.0
        chat = 0.0

    s1 = min(unL - cL, uhat_n - chat)
    s2 = max(unR + cR, uhat_n + chat)

    eps = 1e-14 * max(1.0, abs(s1), abs(s2))
    if abs(s2 - s1) < eps:
        a = max(absf(unL) + cL, absf(unR) + cR, abs(uhat_n) + chat)
        s1 = -a
        s2 =  a

    if s1 >= 0.0:
        return FL0, FL1, FL2
    if s2 <= 0.0:
        return FR0, FR1, FR2

    denom = s2 - s1
    dq0 = qR0 - qL0
    dq1 = qR1 - qL1
    dq2 = qR2 - qL2

    F0 = (s2*FL0 - s1*FR0 + s1*s2*dq0) / denom
    F1 = (s2*FL1 - s1*FR1 + s1*s2*dq1) / denom
    F2 = (s2*FL2 - s1*FR2 + s1*s2*dq2) / denom
    return F0, F1, F2

@cuda.jit(device=True, inline=True)
def roe_flux_face(qL0,qL1,qL2, qR0,qR1,qR2, g, direction, dry_tol):
    """
    Roe flux for 2D shallow water across one face.

    direction = 0 -> x-face, normal velocity = u, flux = Fx
    direction = 1 -> y-face, normal velocity = v, flux = Fy

    Conserved variables: q = (h, hu, hv)
    Returns: (F0, F1, F2)
    """
    uL, vL, wetL = primitives_cell(qL0, qL1, qL2, dry_tol)
    uR, vR, wetR = primitives_cell(qR0, qR1, qR2, dry_tol)

    hL = qL0 if wetL else 0.0
    hR = qR0 if wetR else 0.0

    # ambos secos
    if (not wetL) and (not wetR):
        return 0.0, 0.0, 0.0

    if direction == 0:
        unL, unR = uL, uR
        utL, utR = vL, vR

        FL0, FL1, FL2 = flux_x_cell(qL0, qL1, qL2, g, dry_tol)
        FR0, FR1, FR2 = flux_x_cell(qR0, qR1, qR2, g, dry_tol)

        qnL, qnR = qL1, qR1   # normal momentum = hu
        qtL, qtR = qL2, qR2   # tangential momentum = hv
    else:
        unL, unR = vL, vR
        utL, utR = uL, uR

        FL0, FL1, FL2 = flux_y_cell(qL0, qL1, qL2, g, dry_tol)
        FR0, FR1, FR2 = flux_y_cell(qR0, qR1, qR2, g, dry_tol)

        qnL, qnR = qL2, qR2
        qtL, qtR = qL1, qR1


    # los promedios de Roe

    sqrt_hL = math.sqrt(hL) if hL > 0.0 else 0.0
    sqrt_hR = math.sqrt(hR) if hR > 0.0 else 0.0
    denom_h = sqrt_hL + sqrt_hR

    if denom_h > 0.0:
        uhat_n = (sqrt_hL*unL + sqrt_hR*unR) / denom_h
        uhat_t = (sqrt_hL*utL + sqrt_hR*utR) / denom_h
    else:
        uhat_n = 0.0
        uhat_t = 0.0

    hbar = 0.5 * (hL + hR)
    chat = math.sqrt(g * hbar) if hbar > 0.0 else 0.0

    dh  = hR  - hL
    dqn = qnR - qnL
    dqt = qtR - qtL

    # si estoy cerca de zona seca, llamo HLLE para evitar inestabilidades
    if chat < 1e-14:
        return hlle_flux_face(qL0,qL1,qL2, qR0,qR1,qR2, g, direction, dry_tol)

    alpha1 = ((uhat_n + chat)*dh - dqn) / (2.0 * chat)
    alpha3 = (dqn - (uhat_n - chat)*dh) / (2.0 * chat)
    alpha2 = dqt - uhat_t * dh

    # vp
    lam1 = uhat_n - chat
    lam2 = uhat_n
    lam3 = uhat_n + chat

    # entropía y suavización simple
    delta_ac = 0.1 * chat
    delta_ct = 0.1 * chat

    if absf(lam1) < delta_ac:
        lam1_abs = 0.5 * (lam1*lam1 / delta_ac + delta_ac)
    else:
        lam1_abs = absf(lam1)

    if absf(lam2) < delta_ct:
        lam2_abs = 0.5 * (lam2*lam2 / delta_ct + delta_ct)
    else:
        lam2_abs = absf(lam2)

    if absf(lam3) < delta_ac:
        lam3_abs = 0.5 * (lam3*lam3 / delta_ac + delta_ac)
    else:
        lam3_abs = absf(lam3)


    D0 = (
        lam1_abs * alpha1
        + lam2_abs * 0.0
        + lam3_abs * alpha3
    )

    D1 = (
        lam1_abs * alpha1 * (uhat_n - chat)
        + lam2_abs * alpha2 * 0.0
        + lam3_abs * alpha3 * (uhat_n + chat)
    )

    D2 = (
        lam1_abs * alpha1 * uhat_t
        + lam2_abs * alpha2 * 1.0
        + lam3_abs * alpha3 * uhat_t
    )

    Fh = 0.5 * (FL0 + FR0) - 0.5 * D0

    if direction == 0:
        Fqn = 0.5 * (FL1 + FR1) - 0.5 * D1
        Fqt = 0.5 * (FL2 + FR2) - 0.5 * D2
        return Fh, Fqn, Fqt
    else:

        Fqt = 0.5 * (FL1 + FR1) - 0.5 * D2
        Fqn = 0.5 * (FL2 + FR2) - 0.5 * D1
        return Fh, Fqt, Fqn

@cuda.jit
def apply_bc_periodic(q, ng):
    ny = q.shape[1]
    nx = q.shape[2]
    j, i = cuda.grid(2)
    if j >= ny or i >= nx:
        return

    if i < ng:
        src = nx - 2*ng + i
        q[0,j,i] = q[0,j,src]
        q[1,j,i] = q[1,j,src]
        q[2,j,i] = q[2,j,src]
    elif i >= nx - ng:
        src = ng + (i - (nx - ng))
        q[0,j,i] = q[0,j,src]
        q[1,j,i] = q[1,j,src]
        q[2,j,i] = q[2,j,src]

    if j < ng:
        src = ny - 2*ng + j
        q[0,j,i] = q[0,src,i]
        q[1,j,i] = q[1,src,i]
        q[2,j,i] = q[2,src,i]
    elif j >= ny - ng:
        src = ng + (j - (ny - ng))
        q[0,j,i] = q[0,src,i]
        q[1,j,i] = q[1,src,i]
        q[2,j,i] = q[2,src,i]

@cuda.jit
def apply_bc_wall(q, ng):
    """
    Reflective walls:
    - vertical walls flip hu
    - horizontal walls flip hv
    """
    ny = q.shape[1]
    nx = q.shape[2]
    j, i = cuda.grid(2)
    if j >= ny or i >= nx:
        return

    if i < ng:
        src = 2*ng - 1 - i
        q[0,j,i] = q[0,j,src]
        q[1,j,i] = -q[1,j,src]
        q[2,j,i] = q[2,j,src]
    elif i >= nx - ng:
        src = (nx - 2*ng - 1) - (i - (nx - ng))
        q[0,j,i] = q[0,j,src]
        q[1,j,i] = -q[1,j,src]
        q[2,j,i] = q[2,j,src]

    if j < ng:
        src = 2*ng - 1 - j
        q[0,j,i] = q[0,src,i]
        q[1,j,i] = q[1,src,i]
        q[2,j,i] = -q[2,src,i]
    elif j >= ny - ng:
        src = (ny - 2*ng - 1) - (j - (ny - ng))
        q[0,j,i] = q[0,src,i]
        q[1,j,i] = q[1,src,i]
        q[2,j,i] = -q[2,src,i]

@cuda.jit
def compute_speed_phys(q, speed_out, g, ng, dry_tol, nx_phys, ny_phys):
    tid = cuda.grid(1)
    if tid >= nx_phys*ny_phys:
        return
    j = tid // nx_phys
    i = tid - j*nx_phys

    jj = j + ng
    ii = i + ng

    h  = q[0, jj, ii]
    hu = q[1, jj, ii]
    hv = q[2, jj, ii]

    if h > dry_tol:
        u = hu / h
        v = hv / h
        c = math.sqrt(g*h)
        s = max(absf(u) + c, absf(v) + c)
    else:
        s = 0.0

    speed_out[tid] = s

@cuda.jit
def reduce_max(x, out, n):
    sm = cuda.shared.array(256, dtype=float32)  # assumes TPB=256
    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * cuda.blockDim.x + tid

    v = float32(0.0)
    if i < n:
        v = x[i]

    sm[tid] = v
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            if sm[tid + stride] > sm[tid]:
                sm[tid] = sm[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        out[cuda.blockIdx.x] = sm[0]

def max_device(x_dev, tpb=256):
    n = x_dev.size
    blocks = (n + tpb - 1) // tpb
    curr = cuda.device_array(blocks, dtype=np.float32)
    reduce_max[blocks, tpb](x_dev, curr, n)

    curr_n = blocks
    while curr_n > 1:
        blocks = (curr_n + tpb - 1) // tpb
        nxt = cuda.device_array(blocks, dtype=np.float32)
        reduce_max[blocks, tpb](curr, nxt, curr_n)
        curr = nxt
        curr_n = blocks

    cuda.synchronize()
    return float(curr.copy_to_host()[0])

@cuda.jit(device=True, inline=True)
def riemann_flux_face(qL0,qL1,qL2, qR0,qR1,qR2, g, direction, dry_tol, solver):
    if solver == SOLVER_HLL:
        return hll_flux_face(qL0,qL1,qL2, qR0,qR1,qR2, g, direction, dry_tol)
    elif solver == SOLVER_HLLE:
        return hlle_flux_face(qL0,qL1,qL2, qR0,qR1,qR2, g, direction, dry_tol)
    else:
        return roe_flux_face(qL0,qL1,qL2, qR0,qR1,qR2, g, direction, dry_tol)


@cuda.jit
def sweep_x(q, qnew, dx, dt, g, ng, dry_tol, solver):
    j, i = cuda.grid(2)
    ny = q.shape[1]
    nx = q.shape[2]

    j0 = ng
    i0 = ng
    j1 = ny - ng
    i1 = nx - ng

    if j < j0 or j >= j1 or i < i0 or i >= i1:
        return

    Fp0,Fp1,Fp2 = riemann_flux_face(
        q[0,j,i],   q[1,j,i],   q[2,j,i],
        q[0,j,i+1], q[1,j,i+1], q[2,j,i+1],
        g, 0, dry_tol, solver
    )
    Fm0,Fm1,Fm2 = riemann_flux_face(
        q[0,j,i-1], q[1,j,i-1], q[2,j,i-1],
        q[0,j,i],   q[1,j,i],   q[2,j,i],
        g, 0, dry_tol, solver
    )

    qnew[0,j,i] = q[0,j,i] - (dt/dx)*(Fp0 - Fm0)
    qnew[1,j,i] = q[1,j,i] - (dt/dx)*(Fp1 - Fm1)
    qnew[2,j,i] = q[2,j,i] - (dt/dx)*(Fp2 - Fm2)

@cuda.jit
def sweep_y(qnew, qtmp, dy, dt, g, ng, dry_tol, solver):
    j, i = cuda.grid(2)
    ny = qnew.shape[1]
    nx = qnew.shape[2]

    j0 = ng
    i0 = ng
    j1 = ny - ng
    i1 = nx - ng

    if j < j0 or j >= j1 or i < i0 or i >= i1:
        return

    Fp0,Fp1,Fp2 = riemann_flux_face(
        qnew[0,j,i],   qnew[1,j,i],   qnew[2,j,i],
        qnew[0,j+1,i], qnew[1,j+1,i], qnew[2,j+1,i],
        g, 1, dry_tol, solver
    )
    Fm0,Fm1,Fm2 = riemann_flux_face(
        qnew[0,j-1,i], qnew[1,j-1,i], qnew[2,j-1,i],
        qnew[0,j,i],   qnew[1,j,i],   qnew[2,j,i],
        g, 1, dry_tol, solver
    )

    qtmp[0,j,i] = qnew[0,j,i] - (dt/dy)*(Fp0 - Fm0)
    qtmp[1,j,i] = qnew[1,j,i] - (dt/dy)*(Fp1 - Fm1)
    qtmp[2,j,i] = qnew[2,j,i] - (dt/dy)*(Fp2 - Fm2)

@cuda.jit
def positivity_fix(q, ng, dry_tol):
    j, i = cuda.grid(2)
    ny = q.shape[1]
    nx = q.shape[2]
    j0 = ng
    i0 = ng
    j1 = ny - ng
    i1 = nx - ng

    if j < j0 or j >= j1 or i < i0 or i >= i1:
        return

    h = q[0,j,i]
    if h < 0.0:
        h = 0.0
    q[0,j,i] = h

    if h < dry_tol:
        q[1,j,i] = 0.0
        q[2,j,i] = 0.0


# Solo saco del divice a h. Tarea: sacar también hu, hv para análisis o animación.
@cuda.jit
def extract_h_phys(d_q, d_H, ng):
    """
    d_q: (3, ny+2ng, nx+2ng)
    d_H: (ny, nx) -- physical only
    """
    j, i = cuda.grid(2)
    ny = d_H.shape[0]
    nx = d_H.shape[1]
    if j < ny and i < nx:
        d_H[j, i] = d_q[0, j + ng, i + ng]

# Aquí defino la función principal que corre la simulación y guarda snapshots en tiempos controlados por el usuario.
def run_dambreak_2d_snapshots_gpu(
    nx=220, ny=220, x0=0.0, x1=1.0, y0=0.0, y1=1.0,
    tfinal=0.20, cfl=0.40, g=9.81,
    hL=2.0, hR=1.0,
    dam="circle",     # "vertical" or "circle"
    bc="wall",        # "wall" or "periodic"
    dry_tol=1e-14,
    tpb2d=(16,16),
    tpb1d=256,
    times_save=None,
):
    ng = 2
    bc_code = BC_WALL if bc.lower()=="wall" else BC_PERIODIC


    if times_save is None:
        times = np.array([0.0, tfinal], dtype=np.float32)
    elif isinstance(times_save, (int, np.integer)):
        n = int(times_save)
        if n <= 0:
            times = np.array([tfinal], dtype=np.float32)      # SOLO final
        elif n == 1:
            times = np.array([0.0], dtype=np.float32)         # solo t=0
        else:
            times = np.linspace(0.0, tfinal, n).astype(np.float32)
    else:
        times = np.asarray(times_save, dtype=np.float32)

        times = times[np.isfinite(times)]
        times = np.clip(times, 0.0, tfinal)
        times = np.unique(times)
        times.sort()
        # si el usuario pasó vacío, guardamos solo final
        if times.size == 0:
            times = np.array([tfinal], dtype=np.float32)

    # asegurar incluir tfinal
    if abs(float(times[-1]) - float(tfinal)) > 1e-12:
        times = np.append(times, np.float32(tfinal))

    # Nota: si times contiene 0.0, guardaremos t=0
    # Si times == [tfinal], guardamos solo el final.
    nsave = len(times)

    # grid
    x = np.linspace(x0, x1, nx).astype(np.float32)
    y = np.linspace(y0, y1, ny).astype(np.float32)
    dx = np.float32(x[1] - x[0])
    dy = np.float32(y[1] - y[0])
    X, Y = np.meshgrid(x, y)

    qh = np.zeros((3, ny+2*ng, nx+2*ng), dtype=np.float32)

    # IC (host)
    if dam == "vertical":
        mid = 0.5*(x0+x1)
        h0 = np.where(X <= mid, hL, hR).astype(np.float32)
    elif dam == "circle":
        cx, cy = 0.5*(x0+x1), 0.5*(y0+y1)
        r0 = 0.15*min(x1-x0, y1-y0)
        h0 = np.where((X-cx)**2 + (Y-cy)**2 <= r0*r0, hL, hR).astype(np.float32)
    else:
        raise ValueError("dam must be 'vertical' or 'circle'")

    qh[0, ng:-ng, ng:-ng] = h0
    qh[1, ng:-ng, ng:-ng] = 0.0
    qh[2, ng:-ng, ng:-ng] = 0.0

    # device arrays
    d_q    = cuda.to_device(qh)
    d_qnew = cuda.device_array_like(d_q)
    d_qtmp = cuda.device_array_like(d_q)

    # espacio en la GPU para h
    d_H = cuda.device_array((ny, nx), dtype=np.float32)

    # host snapshots
    Hhist = np.zeros((nsave, ny, nx), dtype=np.float32)

    # para CFL
    d_speed = cuda.device_array(nx*ny, dtype=np.float32)

    TPBX, TPBY = tpb2d
    blocks_full = ((nx + 2*ng + TPBX - 1)//TPBX, (ny + 2*ng + TPBY - 1)//TPBY)
    blocks_phys = blocks_full
    blocks_h    = ((nx + TPBX - 1)//TPBX, (ny + TPBY - 1)//TPBY)
    blocks1d    = (nx*ny + tpb1d - 1)//tpb1d

    g_f   = np.float32(g)
    dry_f = np.float32(dry_tol)

    t = 0.0
    k = 0

    # si el primer tiempo a guardar es 0, guardamos de inmediato
    if abs(float(times[0]) - 0.0) < 1e-12:
        extract_h_phys[blocks_h, (TPBX,TPBY)](d_q, d_H, ng)
        Hhist[0] = d_H.copy_to_host()
        k = 1

    while t < tfinal and k < nsave:
        # BC
        if bc_code == BC_WALL:
            apply_bc_wall[blocks_full, (TPBX,TPBY)](d_q, ng)
        else:
            apply_bc_periodic[blocks_full, (TPBX,TPBY)](d_q, ng)

        # CFL
        compute_speed_phys[blocks1d, tpb1d](d_q, d_speed, g_f, ng, dry_f, nx, ny)
        amax = max_device(d_speed, tpb=tpb1d)

        dt = float(cfl) * float(min(dx, dy)) / max(amax, 1e-12)

        dt = min(dt, float(times[k] - t))
        if dt <= 0.0:
            dt = 1e-12
        dt_f = np.float32(dt)

        # X sweep
        sweep_x[blocks_phys, (TPBX,TPBY)](d_q, d_qnew, dx, dt_f, g_f, ng, dry_f, int32(SOLVER))
        positivity_fix[blocks_phys, (TPBX,TPBY)](d_qnew, ng, dry_f)

        # BC before Y
        if bc_code == BC_WALL:
            apply_bc_wall[blocks_full, (TPBX,TPBY)](d_qnew, ng)
        else:
            apply_bc_periodic[blocks_full, (TPBX,TPBY)](d_qnew, ng)

        # Y sweep
        sweep_y[blocks_phys, (TPBX,TPBY)](d_qnew, d_qtmp, dy, dt_f, g_f, ng, dry_f, int32(SOLVER))
        positivity_fix[blocks_phys, (TPBX,TPBY)](d_qtmp, ng, dry_f)

        # swap
        d_q, d_qtmp = d_qtmp, d_q

        t = t + dt

        if t >= float(times[k]) - 1e-12:
            extract_h_phys[blocks_h, (TPBX,TPBY)](d_q, d_H, ng)
            Hhist[k] = d_H.copy_to_host()
            k += 1

    return X, Y, times[:k], Hhist[:k]


# Animation, esto lo hace lento, así que lo dejo aparte para no afectar el rendimiento de la simulación.
def animate_h_2d(X, Y, times, Hhist):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(Hhist[0], origin="lower",
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   aspect="auto")
    plt.colorbar(im, ax=ax)
    title = ax.set_title(f"t = {times[0]:.4f}")

    def update(frame):
        im.set_data(Hhist[frame])
        title.set_text(f"t = {times[frame]:.4f}")
        return im, title

    anim = FuncAnimation(fig, update, frames=len(times), interval=60, blit=False)
    plt.tight_layout()
    plt.show()
    return anim


# solo tiempo final
# times_save = 0

# N snapshots uniformes
times_save = 20

# tiempos exactos
# times_save = [0.0, 0.02, 0.05, 0.08, 0.2]

t0 = time.perf_counter()
X, Y, times, Hhist = run_dambreak_2d_snapshots_gpu(
    nx=320, ny=320, tfinal=0.2,
    hL=2.0, hR=1.0,
    dam="circle",
    bc="wall",
    cfl=0.40,
    times_save=times_save,
)
t1 = time.perf_counter()
print("Snapshots guardados:", len(times), "Tiempo total (s):", t1 - t0, "times:", times)

if len(times) > 1:
    animate_h_2d(X, Y, times, Hhist)
else:
    plt.figure(figsize=(6,5))
    plt.imshow(Hhist[0], origin="lower",
                extent=[X.min(), X.max(), Y.min(), Y.max()],
                aspect="auto")
    plt.colorbar()
    plt.title(f"h(t={times[0]:.4f})")
    plt.tight_layout()
    plt.show()