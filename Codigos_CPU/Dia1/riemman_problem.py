import numpy as np
import matplotlib.pyplot as plt
from riemman_tools import plot_riemann, plot_characteristics
from matplotlib.widgets import Slider, Button, RadioButtons

def slides_burger(init_ql=0.5, init_qr=0.0, init_t=0.2, init_xr=1.0):
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Burgers 1D - Riemann exacto", fontsize=13)

    ax = fig.add_axes([0.08, 0.22, 0.62, 0.70])  # plot principal

    ax_ql = fig.add_axes([0.08, 0.14, 0.62, 0.03])
    ax_qr = fig.add_axes([0.08, 0.10, 0.62, 0.03])
    ax_t  = fig.add_axes([0.08, 0.06, 0.62, 0.03])
    ax_xr = fig.add_axes([0.08, 0.02, 0.62, 0.03])

    ax_reset = fig.add_axes([0.75, 0.62, 0.20, 0.06])

    s_ql = Slider(ax_ql, r"$q_L$", 0.0, 2.0, valinit=init_ql, valstep=0.05)
    s_qr = Slider(ax_qr, r"$q_R$", 0.0, 2.0, valinit=init_qr, valstep=0.05)
    s_t  = Slider(ax_t,  r"$t$",   0.0,  10.0, valinit=init_t,  valstep=0.05)
    s_xr = Slider(ax_xr, "x-range", 0.5, 5.0, valinit=init_xr, valstep=0.5)

    btn_reset = Button(ax_reset, "Reset")

    def clear_ax(ax_):
        ax_.cla()
        ax_.grid(True, alpha=0.25)

    def draw():
        q_l = float(s_ql.val)
        q_r = float(s_qr.val)
        t = float(s_t.val)
        x_range = float(s_xr.val)

        clear_ax(ax)

        states, velocidads, reval, wave_types = sol_riemann_exacta(q_l, q_r)
        title = f"Exacta: {wave_types[0]} | qL={q_l:.2f}, qR={q_r:.2f}, t={t:.2f}"

        x = np.linspace(-x_range, x_range, 1200)
        if t <= 1e-12:
            u = np.where(x < 0, q_l, q_r)
        else:
            xi = x / t
            u = reval(xi)[0, :]

        ax.plot(x, u, lw=2)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.set_title(title)
        ax.set_xlim(-x_range, x_range)

        fig.canvas.draw_idle()

    def on_slider_change(_):
        draw()

    def on_reset(_event):
        # reset() ya dispara on_changed y redibuja,
        # así que NO hace falta llamar draw() explícitamente.
        s_ql.reset()
        s_qr.reset()
        s_t.reset()
        s_xr.reset()

    s_ql.on_changed(on_slider_change)
    s_qr.on_changed(on_slider_change)
    s_t.on_changed(on_slider_change)
    s_xr.on_changed(on_slider_change)
    btn_reset.on_clicked(on_reset)

    draw()
    plt.show()

def plot_burgers(case, q_l, q_r, t, x_range=1.0):
    if case == "Exact":
        states, velocidads, reval, wave_types = sol_riemann_exacta(q_l, q_r)
        title = f"Exacta (Burgers): {wave_types[0]}  |  q_l={q_l:.2f}, q_r={q_r:.2f}, t={t:.2f}"
    else:
        if abs(q_l - q_r) < 1e-14:
            q_r = q_l + 1e-6
        states, velocidads, reval, wave_types = unphysical_riemann_solution(q_l, q_r)
        title = f"No física (expansion shock)  |  q_l={q_l:.2f}, q_r={q_r:.2f}, t={t:.2f}"

    ax = plot_riemann(
        states, velocidads, reval, wave_types,
        t=t,
        layout="horizontal",
        t_pointer=0,
        extra_axes=False,
        variable_names=["q"],
        xmax=x_range
    )
    plot_characteristics(reval, velocidad, axes=ax[0])

    for a in ax:
        a.set_xlim(-x_range, x_range)
    ax[0].set_title(title)

    plt.show()


def velocidad(q, xi):
    return q

def sol_riemann_exacta(q_l, q_r):
    f = lambda q: 0.5*q*q
    states = np.array([[q_l, q_r]])

    if q_l > q_r:
        shock_velocidad = (f(q_l) - f(q_r)) / (q_l - q_r)
        velocidads = [shock_velocidad]
        wave_types = ['shock']

        def reval(xi):
            q = np.zeros((1, len(xi)))
            q[0, :] = (xi < shock_velocidad) * q_l + (xi >= shock_velocidad) * q_r
            return q

    else:
        c_l = q_l
        c_r = q_r
        velocidads = [[c_l, c_r]]
        wave_types = ['rarefaction']

        def reval(xi):
            q = np.zeros((1, len(xi)))
            q[0, :] = (xi <= c_l) * q_l + (xi >= c_r) * q_r + ((c_l < xi) & (xi < c_r)) * xi
            return q

    return states, velocidads, reval, wave_types


def unphysical_riemann_solution(q_l, q_r):
    r"""Unphysical solution (expansion shock) for Burgers' equation."""
    f = lambda q: 0.5*q*q
    states = np.array([[q_l, q_r]])

    shock_velocidad = (f(q_l) - f(q_r)) / (q_l - q_r)
    velocidads = [shock_velocidad]
    wave_types = ['shock']

    def reval(xi):
        q = np.zeros((1, len(xi)))
        q[0, :] = (xi < shock_velocidad) * q_l + (xi >= shock_velocidad) * q_r
        return q

    return states, velocidads, reval, wave_types

plot_burgers("Exact", q_l=0.8, q_r=0.2, t=0.25, x_range=1.5)
plot_burgers("Exact", q_l=0.2, q_r=0.8, t=0.25, x_range=1.5)
slides_burger(init_ql=0.1, init_qr=1.0, init_t=0.0, init_xr=1.0)
plot_burgers("No física", q_l=0.2, q_r=0.8, t=0.25, x_range=1.5)