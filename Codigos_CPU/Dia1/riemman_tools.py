import matplotlib
from matplotlib import pyplot as plt
import numpy as np

def plot_characteristics(reval, char_speed, aux=None, axes=None,
                         extra_lines=None, speeds=None, contact_index=None):

    if axes:
        xmin, xmax, tmin, tmax = axes.axis()
    else:
        xmin, xmax, tmin, tmax = (-1., 1., 0., 0.5)

    dx = xmax-xmin
    x = np.linspace(xmin-dx, xmax+dx, 60)
    t = np.linspace(tmin,tmax,500)
    chars = np.zeros((len(x),len(t)))  # x-t coordinates of characteristics, one curve per row
    chars[:,0] = x
    dt = t[1]-t[0]
    c = np.zeros(len(x))
    if contact_index is None:
        contact_speed = 0
    else:
        contact_speed = speeds[contact_index]
    for i in range(1,len(t)):
        xi = chars[:,i-1]/max(t[i-1],dt)
        q = np.array(reval(xi))
        for j in range(len(x)):
            if aux:
                c[j] = char_speed(q[:,j],xi[j],(xi[j]<=contact_speed)*aux[0]+(xi[j]>contact_speed)*aux[1])
            else:
                c[j] = char_speed(q[:,j],xi[j])
        chars[:,i] = chars[:,i-1] + dt*c  # Euler's method

    for j in range(len(x)):
        axes.plot(chars[j,:],t,'-k',linewidth=0.2,zorder=0)

    if extra_lines:
        for endpoints in extra_lines:
            begin, end = endpoints
            x = np.linspace(begin[0], end[0], 10)
            tstart = np.linspace(begin[1], end[1], 10)
            for epsilon in (-1.e-3, 1.e-3):
                for xx, tt in zip(x+epsilon,tstart):
                    t = np.linspace(tt,tmax,200)
                    dt = t[1]-t[0]
                    char = np.zeros(len(t))
                    char[0] = xx
                    for i in range(1,len(t)):
                        xi = char[i-1]/max(t[i-1],dt)
                        q = np.array(reval(np.array([xi])))
                        if aux:
                            c = char_speed(q,xi,(xi<=0)*aux[0]+(xi>0)*aux[1])
                        else:
                            c = char_speed(q,xi)
                        char[i] = char[i-1] + dt*c
                    axes.plot(char,t,'-k',linewidth=0.2,zorder=0)

    axes.axis((xmin, xmax, tmin, tmax))

def plot_waves(states, s, riemann_eval, wave_types, t=0.1, ax=None,
               color='multi', t_pointer=False, xmax=None):

    if wave_types is None:
        wave_types = ['contact']*len(s)

    colors = {}
    if color == 'multi':
        colors['shock'] = 'r'
        colors['raref'] = 'b'
        colors['contact'] = 'k'
    else:
        colors['shock'] = color
        colors['raref'] = color
        colors['contact'] = color

    if ax is None:
        fig, ax = plt.subplots()

    tmax = 1.0
    if xmax is None:
        xmax = 0.
    for i in range(len(s)):
        if wave_types[i] in ['shock','contact']:
            x1 = tmax * s[i]
            ax.plot([0,x1],[0,tmax],color=colors[wave_types[i]])
            xmax = max(xmax,abs(x1))
        else:  # plot rarefaction fan
            speeds = np.linspace(s[i][0],s[i][1],5)
            for ss in speeds:
                x1 = tmax * ss
                ax.plot([0,x1],[0,tmax],color=colors['raref'],lw=0.6)
                xmax = max(xmax,abs(x1))

    xmax = max(0.001, xmax)
    ax.set_xlim(-xmax,xmax)
    ax.plot([-xmax,xmax],[t,t],'--k',linewidth=0.5)
    if t_pointer:
        ax.text(-1.8*xmax,t,'t = %4.2f -->' % t)
    ax.set_title('Waves in x-t plane')
    ax.set_ylim(0,tmax)

def convert_to_list(x):
    if isinstance(x, (list, tuple)):
        return x
    else:
        return [x]

def plot_riemann(states, s, riemann_eval, wave_types=None, t=0.1, ax=None,
                 color='multi', layout='horizontal', variable_names=None,
                 t_pointer=False, extra_axes=False, fill=(),
                 derived_variables=None, xmax=None):

    num_vars, num_states = states.shape
    pstates = states.copy()
    if derived_variables:
        num_vars = len(derived_variables(states[:,0]))
        for i in range(num_states):
            pstates[:,i] = derived_variables(states[:,i])

    if ax is not None:
        assert len(ax) == num_vars + 1 + extra_axes

    if wave_types is None:
        wave_types = ['contact']*len(s)

    if variable_names is None:
        if num_vars == 1:
            variable_names = ['q']
        else:
            variable_names = ['$q_%s$' % i for i in range(1,num_vars+1)]

    if ax is None:  # Set up a new plot and axes
        existing_plots = False
        num_axes = num_vars+1
        if extra_axes: num_axes += extra_axes
        if layout == 'horizontal':
            # Plots side by side
            if num_axes >= 4:
                fig_width = 10
            else:
                fig_width = 3*num_axes
            fig, ax = plt.subplots(1,num_axes,figsize=(fig_width,3))
            plt.subplots_adjust(wspace=0.5)
            plt.tight_layout()
        elif layout == 'vertical':
            # Plots on top of each other, with shared x-axis
            fig_width = 8
            fig_height = 1.5*(num_axes-1)
            fig, ax = plt.subplots(num_axes,1,figsize=(fig_width,fig_height),sharex=True)
            plt.subplots_adjust(hspace=0)
            ax[-1].set_xlabel('x')
            ax[0].set_ylabel('t')
            ax[0].set_title('t = %6.3f' % t)
    else:
        assert len(ax) == num_vars + 1 + extra_axes
        existing_plots = True
        ylims = []
        for i in range(1,len(ax)):
            ylims.append(ax[i].get_ylim())

    # Make plot boundaries grey
    for axis in ax:
        for child in axis.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color('#dddddd')

    # Plot wave characteristics in x-t plane
    plot_waves(pstates, s, riemann_eval, wave_types, t=t, ax=ax[0], color=color,
               t_pointer=t_pointer, xmax=xmax)

    if xmax is None:
        xmax = ax[0].get_xlim()[1]

    # Plot conserved quantities as function of x for fixed t
    # Use xi values in [-10,10] unless the wave speeds are so large
    # that we need a larger range
    xi_range = np.linspace(min(-10, 2*np.min(s[0])), max(10, 2*np.max(s[-1])))
    q_sample = riemann_eval(xi_range)
    if derived_variables:
        q_sample = derived_variables(q_sample)

    for i in range(num_vars):
        # Set axis bounds to comfortably fit the values that will be plotted
        ax[i+1].set_xlim((-1,1))
        qmax = max(np.nanmax(q_sample[i][:]), np.nanmax(pstates[i,:]))
        qmin = min(np.nanmin(q_sample[i][:]), np.nanmin(pstates[i,:]))
        qdiff = qmax - qmin
        ax[i+1].set_xlim(-xmax, xmax)
        if qmin == qmax:
            qmin = qmin*0.9
            qmax = qmin*1.1+0.01
        if existing_plots:
            ax[i+1].set_ylim((min(ylims[i][0],qmin-0.1*qdiff), max(ylims[i][1],qmax+0.1*qdiff)))
        else:
            ax[i+1].set_ylim((qmin-0.1*qdiff, qmax+0.1*qdiff))

        if layout == 'horizontal':
            ax[i+1].set_title(variable_names[i]+' at t = %6.3f' % t)
        elif layout == 'vertical':
            ax[i+1].set_ylabel(variable_names[i])

    x = np.linspace(-xmax, xmax, 1000)
    if t>0:
        # Make sure we have a value of x between each pair of waves
        # This is important e.g. for nearly-pressureless gas,
        # in order to catch small regions
        wavespeeds = []
        for speed in s:
            wavespeeds += convert_to_list(speed)

        wavespeeds = np.array(wavespeeds)
        xm = 0.5*(wavespeeds[1:]+wavespeeds[:-1])*t
        iloc = np.searchsorted(x,xm)
        x = np.insert(x, iloc, xm)

    # Avoid dividing by zero
    q = riemann_eval(x/(t+1e-10))
    if derived_variables:
        q = derived_variables(q)

    for i in range(num_vars):
        if color == 'multi':
            ax[i+1].plot(x,q[i][:],'-k',lw=2)
        else:
            ax[i+1].plot(x,q[i][:],'-',color=color,lw=2)
        if i in fill:
            ax[i+1].fill_between(x,q[i][:],color='b')
            ax[i+1].set_ybound(lower=0)

    return ax


