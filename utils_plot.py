import numpy as np
import matplotlib.pyplot as plt

colors = ["#ED0000", "#FFF200", "#0000ED", "#FF7E27", "#00ED00", "#7F3F3F", "#B97A57", "#FFDBB7", "#ED00ED", "#00EDED",
          "#FF7F7F", "#FFC90D", "#6C6CFF", "#FFB364", "#22B14C", "#D1A5A5", "#008040", "#AEFFAE", "#FF84FF", "#95FFFF",           "#FFB7B7", "#808000", "#0080FF", "#C46200", "#008080", "#800000", "#B0B0FF", "#FFFFB3", "#FF3E9E", "#BBBB00",
          "#00ED00", "#00FFF2", "#ED0000", "#27FF7E", "#0000ED", "#3F7F3F", "#57B97A", "#B7FFDB", "#EDED00", "#ED00ED"]


def check_timeVector_series(time_vector, time_s):
    cnt = 0
    time_v = time_vector
    while len(time_vector) != len(time_s) and cnt < 3:
        time_v = time_vector[index]
        cnt += 1
    assert cnt < 3, "Mismatch of shapes between time_vector and signal"
    return time_v, time_s


def plot_res_mssm(time_vector, input_signal, reference_signal, label):
    fig = plt.figure(figsize=(12, 2.8))
    plt.suptitle(label)
    ax = fig.add_subplot(121)
    ax.plot(time_vector, input_signal, c='gray')
    ax.set_xlabel("time(s)")
    ax.set_ylabel("Spikes")
    ax.set_title("Input Spike train")
    ax.grid()
    ax = fig.add_subplot(122)
    for i in range(reference_signal.shape[0]): ax.plot(time_vector, reference_signal[i, :])
    ax.set_xlabel("time(s)")
    ax.set_ylabel("A")
    ax.set_title("Postsynaptic response")
    ax.grid()
    fig.tight_layout(pad=0.5, w_pad=0.1, h_pad=1.0)


def plot_syn_dyn(time_vector, plot_title, ind_plots, plots, plot_leg=None, color_plot=None, alphas=None,
                 subplot_title=None, xlabels=None, ylabels=None, ylim_ranges=None, subtitle_size=12,
                 subtitle_color='gray', fig_size=(17, 8.5), fig_pad=.8, save=False, plot=True, path_to_save="",
                 yerr=None, xerr=None, uplims=None, lolims=None, vlines=None, xscale='linear', std_plt1=None,
                 std_plt2=None, x_axis_log=False, y_axis_log=False):
    # Default graphical parameters
    alph = None
    labl = None
    col = None
    yer = None
    xer = None
    uplim = False
    lolim = False
    # Creating plot
    fig = plt.figure(figsize=fig_size)
    fig.suptitle(plot_title)

    # Looping through the plots
    for index in range(len(ind_plots)):

        # Adding a new axis for each subplot
        ax = fig.add_subplot(ind_plots[index])

        # if more than 1 graph should be plotted in ax
        if isinstance(plots[index], list) or (isinstance(plots[index], np.ndarray) and plots[index].ndim > 1):

            # Looping through the graphs
            for pl in range(len(plots[index])):

                # Specifying graphical parameters
                if alphas is not None:
                    alph = alphas[index][pl]
                if plot_leg is not None:
                    labl = plot_leg[index][pl]
                if color_plot is not None:
                    col = color_plot[index][pl]
                if yerr is not None:
                    yer = yerr[index][pl]
                if xerr is not None:
                    xer = xerr[index][pl]
                if uplims is not None:
                    uplim = uplims[index][pl]
                if lolims is not None:
                    lolim = lolims[index][pl]
                if std_plt1 is not None:
                    std_pl1 = std_plt1[index][pl]
                if std_plt2 is not None:
                    std_pl2 = std_plt2[index][pl]
                # Plotting
                time_v, time_s = check_timeVector_series(time_vector, plots[index][pl])
                ax.errorbar(time_v, time_s, label=labl, c=col, alpha=alph, yerr=yer, xerr=xer,
                            uplims=uplim, lolims=lolim, capsize=0.1)
                if std_plt1 is not None or std_plt2 is not None:
                    ax.fill_between(time_vector[index], plots[index][pl] - std_pl1, plots[index][pl] + std_pl2,
                                    color="grey", alpha=0.5)
            # Adding legends
            ax.legend(framealpha=0.3)

        # Otherwise plot only once
        else:

            # Specifying graphical parameters
            if alphas is not None:
                alph = alphas[index]
            if color_plot is not None:
                col = color_plot[index]
            if yerr is not None:
                yer = yerr[index]
            if xerr is not None:
                xer = xerr[index]
            if uplims is not None:
                uplim = uplims[index]
            if lolims is not None:
                lolim = lolims[index]
            if std_plt1 is not None:
                std_pl1 = std_plt1[index]
            if std_plt2 is not None:
                std_pl2 = std_plt2[index]
            # Plotting
            time_v, time_s = check_timeVector_series(time_vector, np.squeeze(plots[index]))
            ax.errorbar(time_v, time_s, alpha=alph, c=col, yerr=yer, xerr=xer,
                        uplims=uplim, lolims=lolim, capsize=0.1)
            if std_plt1 is not None or std_plt2 is not None:
                ax.fill_between(time_vector[index], plots[index] - std_pl1, plots[index] + std_pl2,
                                color="grey", alpha=0.5)
        ax.set_xscale(xscale)
        ax.grid()

        # Plotting vertical line (if necessary)
        if vlines is not None:
            if vlines[index]:
                ax.axvline(vlines[index], c='red')

        # Specifying graphical parameters
        if subplot_title is not None:
            ax.set_title(subplot_title[index], size=subtitle_size, c=subtitle_color)
        if ylabels is not None:
            ax.set_ylabel(ylabels[index])
        if xlabels is not None:
            ax.set_xlabel(xlabels[index])
        if ylim_ranges is not None:
            ax.set_ylim(ylim_ranges[index])
        if x_axis_log:
            ax.set_xscale('log')
        if y_axis_log:
            ax.set_yscale('log')

    # Adjust graph
    fig.tight_layout(pad=fig_pad)

    # If save is True, then save figure in the path_to_save
    if save:
        path = path_to_save
        path += ".png"
        plt.savefig(path)

    # Close figure if condition is True
    if not plot or save:
        plt.close(fig)

    # If True, show the figure
    if plot:
        plt.show()


def plot_hist_pdf(data, labels, title, medians=False, colors=None, ylimMax=None, ylimMin=None, rotationLabels=0,
                  sizeLabels=8, sizeTitle=12, figSize=None, plotFig=False, returnAx=False, fig=None, pos=None,
                  posInset=None, xAxisSci=False, yAxisFontSize=8, hatchs=None, binSize=30):
    # Create a figure instance
    if (plotFig and fig is None) or fig is None:
        if figSize is None:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=figSize)

    if returnAx:
        # Create an axes instance
        ax = fig.add_subplot(pos[0], pos[1], pos[2])
    else:
        # Create an axes instance
        ax = fig.add_subplot(111)

    # Getting rid of the nan values
    if isinstance(data, list):
        data = np.array(data).T
    # data_list = [[] for _ in range(data.shape[1])]
    # for cols in range(data.shape[1]):
    #     d_aux = data[:, cols]
    #     data_list[cols] = d_aux[~np.isnan(d_aux)]
    data_list = list(data)

    # Create the boxplot
    # colors_hist = [colors[i + 4] for i in range(data.shape[1])]  # [colors[i + 32] for i in range(data.shape[1])]
    colors_hist = colors[4]  # [colors[i + 32] for i in range(data.shape[1])]
    bp = ax.hist(data_list, bins=binSize, density=False, orientation='horizontal', cumulative=False,
                 histtype='stepfilled', label=labels, color=colors_hist, alpha=0.5)
    bp = ax.hist(data_list, bins=binSize, density=False, orientation='horizontal', cumulative=False, histtype='step',
                 color='#808080')  # colors_hist)
    #              color = ['#808080' for i in range(data.shape[1])])  # colors_hist)
    for bar, hatch in zip(ax.patches, hatchs):  # loop over bars and hatches to set hatches in correct order
        bar.set_hatch(hatch)
        # bar.set_edgecolor('#000000')
    # Add a horizontal grid to the plot, but make it very light in color
    ax.yaxis.grid(True, linestyle=':', which='major', color='lightgrey', alpha=0.5, label=labels)
    # Hide these grid behind plot objects
    ax.set_axisbelow(True)

    # Custom x-axis labels
    # ax.set_xticklabels(labels, rotation=rotationLabels, fontsize=sizeLabels)
    # ax.set_xticks([i for i in range(1, len(labels) + 1)], labels, rotation=rotationLabels, fontsize=sizeLabels)
    ax.tick_params(axis='x', labelsize=sizeLabels, colors='gray')
    ax.tick_params(axis='y', labelsize=sizeLabels, colors='gray')
    ax.set_title(title, size=sizeTitle)

    # Format of y-axis
    if xAxisSci:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.offsetText.set_color('black')  # set_position((1.3, 0))
        ax.yaxis.offsetText.set_fontsize(yAxisFontSize)

    # Setting limits to y-axis
    if ylimMax is not None:
        plt.ylim(top=ylimMax)
    if ylimMin is not None:
        plt.ylim(bottom=ylimMin)
