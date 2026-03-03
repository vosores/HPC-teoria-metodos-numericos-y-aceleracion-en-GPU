# Leyes de conservación: teoría, métodos numéricos y aceleración en GPU

Este repositorio contiene el material del seminario **"Leyes de
conservación: teoría, métodos numéricos y aceleración en GPU"**,
orientado a la formación en modelamiento matemático, simulación numérica
y computación de alto rendimiento aplicada a leyes de conservación
hiperbólicas.


## Objetivo

Desarrollar competencias en **modelamiento, simulación numérica y
aceleración computacional** de leyes de conservación hiperbólicas,
integrando:

-   Fundamentos teóricos
-   Métodos de volúmenes finitos
-   Implementaciones en CPU
-   Aceleración en GPU mediante Python


## Resumen

Este seminario introduce el estudio y la simulación numérica de **leyes
de conservación hiperbólicas**, combinando teoría, métodos numéricos
clásicos y computación de alto rendimiento.

En una primera etapa, se estudian leyes de conservación unidimensionales
clásicas, como la ecuación de Burgers, junto con los fenómenos
característicos de estos modelos, incluyendo:

-   Soluciones débiles
-   Ondas de choque
-   Ondas de rarefacción

Posteriormente, se abordan **sistemas de leyes de conservación**
relevantes en aplicaciones físicas, tales como:

-   Ecuaciones de aguas someras
-   Dinámica de gases

Sobre esta base, se presentan métodos numéricos clásicos, en particular
**esquemas de volúmenes finitos**, junto con su implementación
computacional en Python.

El seminario enfatiza el diseño de código:

-   Claro
-   Modular
-   Eficiente
-   Reproducible

En la etapa final, se explora la **aceleración en GPU**, introduciendo
nociones básicas de paralelización y su implementación práctica desde
Python. Se analiza el impacto en desempeño y escalabilidad, comparando
implementaciones en CPU y GPU para distintos modelos.

## Tópicos

-   Leyes de conservación hiperbólicas unidimensionales\
-   Sistemas de leyes de conservación y modelos físicos\
-   Métodos numéricos clásicos para leyes de conservación\
-   Aceleración en GPU y computación de alto rendimiento

## Horario

**Martes a Viernes**\
**16:00 -- 18:00 hrs**

## 📂 Estructura del repositorio

    ├── Presentaciones/
    ├── Notebooks/
    ├── Codigos_CPU/
    ├── Codigos_GPU/
    └── README.md

### Presentaciones
- [Sesión 1: Leyes de conservación y métodos numéricos parte 1.](https://github.com/vosores/HPC-teoria-metodos-numericos-y-aceleracion-en-GPU/blob/main/Presentaciones/Dia1/seminario_dia1.pdf)

### Notebooks
- [Sesión 1: riemman_problem.py](https://colab.research.google.com/drive/1NIIyn2bV4rpME4oqXrBLhHadJEE2dXpm?usp=sharing)
- [Sesión 1: upwind.py](https://colab.research.google.com/drive/1daSbCb0yCY7dI9bLN7l1nFZ7FlVLlxtA?usp=sharing)
- [Sesión 1: burgers_godunov](https://colab.research.google.com/drive/1o950t9zbQiGzGj-h_xljx0g_o-PbJAcb?usp=sharing)

### Codigos_CPU
- [Sesión 1: Códigos CPU.]()
- [Sesión 2: Códigos CPU.]()
- [Sesión 3: Códigos CPU.]()
- [Sesión 4: Códigos CPU.]()

### Codigos_GPU
- [Sesión 1: Códigos GPU.]()
- [Sesión 2: Códigos GPU.]()
- [Sesión 3: Códigos GPU.]()
- [Sesión 4: Códigos GPU.]()

## Requisitos

-   Python 3.10+
-   NumPy
-   Matplotlib
-   Numba
-   CUDA Toolkit (para la parte GPU) o Google colab

Instalación sugerida:

``` bash
pip install numpy matplotlib numba
```

Verificar disponibilidad de GPU:

``` bash
nvidia-smi
```

## Enfoque del seminario

El seminario combina:

-   Teoría matemática rigurosa
-   Implementación computacional práctica
-   Comparación de desempeño CPU vs GPU
-   Discusión sobre escalabilidad y eficiencia

## Público objetivo

Estudiantes de pregrado avanzado, magíster o doctorado en:

-   Matemática
-   Ingeniería
-   Física
-   Ciencia de Datos

Con conocimientos básicos de:

-   Cálculo
-   Álgebra lineal
-   Programación en Python

**Repositorio oficial del seminario**\
Modelamiento · Simulación · GPU · Alto Rendimiento
