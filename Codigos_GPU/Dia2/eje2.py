import numpy as np
import time
from numba import cuda, njit, prange

def heavy_cpu_python(a, b, c, iters):
    n = c.size
    for i in range(n):
        x = float(a[i])
        y = float(b[i])
        for _ in range(iters):
            x = x * 1.000001 + y * 0.999999
            y = y * 1.000003 - x * 0.999997
        c[i] = x + y

@cuda.jit
def heavy_gpu(a, b, c, iters):
    i = cuda.grid(1)
    if i < c.size:
        x = a[i]
        y = b[i]
        for _ in range(iters):
            x = x * 1.000001 + y * 0.999999
            y = y * 1.000003 - x * 0.999997
        c[i] = x + y

@njit(parallel=True, fastmath=True)
def heavy_cpu_numba(a, b, c, iters):
    n = c.size
    for i in prange(n):
        x = a[i]
        y = b[i]
        for _ in range(iters):
            x = x * 1.000001 + y * 0.999999
            y = y * 1.000003 - x * 0.999997
        c[i] = x + y



# Prueba de rendimiento comparando GPU vs CPU (Numba) vs CPU puro
n = 10000000
iters = 500

a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.device_array(n, dtype=np.float32)

threads = 256
blocks = (n + threads - 1) // threads

# GPU warm-up (compila kernel)
heavy_gpu[blocks, threads](d_a, d_b, d_c, iters)
cuda.synchronize()

# GPU timing (solo kernel)
t0 = time.perf_counter()
heavy_gpu[blocks, threads](d_a, d_b, d_c, iters)
cuda.synchronize()
t1 = time.perf_counter()
gpu_time = t1 - t0
print("GPU kernel (s):", gpu_time)
c_gpu = d_c.copy_to_host()

# CPU (Numba)
c_cpu = np.empty_like(a)
heavy_cpu_numba(a, b, c_cpu, iters)

t0 = time.perf_counter()
heavy_cpu_numba(a, b, c_cpu, iters)
t1 = time.perf_counter()
cpu_numba_time = t1 - t0
print("CPU Numba (s):", cpu_numba_time)


c2 = np.empty_like(a)
t0 = time.perf_counter()
heavy_cpu_python(a, b, c2, iters)
t1 = time.perf_counter()
cpu_python_time = t1 - t0
print("CPU Python puro (s):", cpu_python_time)


# Comparación de tiempos
print("Speedup GPU vs CPU(Numba):", cpu_numba_time / gpu_time)
print("Speedup GPU vs CPU(Python):", cpu_python_time / gpu_time)

# Validación rápida (deberían ser muy parecidos; tolerancia por fastmath/orden)
max_err = np.max(np.abs(c_cpu - c_gpu))
print("Max abs error:", max_err)