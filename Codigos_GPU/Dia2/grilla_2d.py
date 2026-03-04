from numba import cuda

@cuda.jit
def grilla2d():
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    bdx = cuda.blockDim.x
    bdy = cuda.blockDim.y

    # ID único por componente (global 2D)
    idx = bx * bdx + tx
    idy = by * bdy + ty

    # ancho global en x para linealizar
    nx = cuda.gridDim.x * bdx
    idg = idy * nx + idx

    print("Hola Mundo!, soy el hilo (", tx, ",", ty, ") del bloque (",
          bx, ",", by, "), pero mi ID es (", idx, ",", idy,
          ") y mi IDG es ", idg, ".")



threads = (2, 2)   # bloque 2D
blocks  = (2, 2)   # grilla 2D
grilla2d[blocks, threads]()
cuda.synchronize()