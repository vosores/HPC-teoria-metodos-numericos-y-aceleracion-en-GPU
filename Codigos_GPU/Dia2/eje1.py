from numba import cuda, njit
import numpy as np
import time

device = cuda.get_current_device()

print("name:", device.name)
print("compute_capability:", device.compute_capability)
print("MULTIPROCESSOR_COUNT:", device.MULTIPROCESSOR_COUNT)
print("MAX_THREADS_PER_BLOCK:", device.MAX_THREADS_PER_BLOCK)
print("MAX_BLOCK_DIM_X:", device.MAX_BLOCK_DIM_X)
print("MAX_BLOCK_DIM_Y:", device.MAX_BLOCK_DIM_Y)
print("MAX_BLOCK_DIM_Z:", device.MAX_BLOCK_DIM_Z)
print("MAX_GRID_DIM_X:", device.MAX_GRID_DIM_X)
print("MAX_GRID_DIM_Y:", device.MAX_GRID_DIM_Y)
print("MAX_GRID_DIM_Z:", device.MAX_GRID_DIM_Z)
print("WARP_SIZE:", device.WARP_SIZE)
print("MAX_SHARED_MEMORY_PER_BLOCK:", device.MAX_SHARED_MEMORY_PER_BLOCK)
print("MAX_REGISTERS_PER_BLOCK:", device.MAX_REGISTERS_PER_BLOCK)
print("CLOCK_RATE:", device.CLOCK_RATE)

# Suma de 2 vectores con CPU puro vs GPU

n = 10000000
def suma(a,b,c):
    for i in range(c.size):
        c[i] = a[i] + b[i]
    return c

a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)
c = np.empty_like(a)

# warm-up (compilación)
suma(a,b,c)

t0 = time.perf_counter()
out = a+b
t1 = time.perf_counter()

print("Resultado:", out)
time_pure_cpu = t1 - t0
print("Tiempo CPU puro (s):", time_pure_cpu)


@cuda.jit
def add(a, b, c):
    i = cuda.grid(1)
    if i < c.size:
        c[i] = a[i] + b[i]

a = np.ones(n, dtype=np.float32)
b = np.ones(n, dtype=np.float32)
c = np.zeros(n, dtype=np.float32)

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.device_array_like(c)

threads = 256
blocks = (n + threads - 1) // threads
add[blocks, threads](d_a, d_b, d_c)

d_c.copy_to_host(c)
print("Resultado:", c[:5])


# warm-up (compila kernel)
add[blocks, threads](d_a, d_b, d_c)
cuda.synchronize()

t0 = time.perf_counter()
add[blocks, threads](d_a, d_b, d_c)
cuda.synchronize() 
t1 = time.perf_counter()
time_gpu = t1 - t0
print("Tiempo kernel GPU (s):", time_gpu)

print("Speedup GPU vs CPU puro:", time_pure_cpu / time_gpu)