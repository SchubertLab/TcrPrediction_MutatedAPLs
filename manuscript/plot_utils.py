import matplotlib.pyplot as plt
import matplotlib as mpl


def add_axis_title_fixed_position(fig, ax, title, x_in=-0.1, y_in=0.05,
                                  size='large', weight='bold',
                                  ha='right', va='bottom',
                                  **kwargs):
    """
    Adds a title to the given axis, positioned with the given offset in inches
    with respect to the top left corner of the axis.

    Parameters
    ----------
    fig : The figure containing the axis
    ax : The axis to be titled.
    title : Text of the title
    x_in : Horizontal offset in inches with respect to the top left
        corner of the axis. Negative is to the left. The default is -0.1.
    y_in : Vertical offset in inches with respect  to the top left
        corner of the axis. Negative is downwards. The default is 0.05.
    size : Font of the title. The default is 'large'.
    weight : Font weight of the title. The default is 'bold'.
    ha : Horizontal alignment of the text with respect to the defined
        position. The default is 'right'.
    va : Vertical alignment of the text with respect to the devined
        position. The default is 'bottom'.
    **kwargs : Other arguments to ax.text

    Returns
    -------
    None.
    """
    
    tr = mpl.transforms.offset_copy(ax.transAxes, x=x_in, y=y_in, fig=fig)
    ax.text(1, 1, title, size=size, weight=weight,
            transform=tr, ha=ha, va=va, **kwargs)


def set_font_size(font_size):
    plt.rc('font', size=font_size)          # controls default text sizes
    plt.rc('axes', titlesize=font_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size, title_fontsize=font_size)    # legend fontsize
    plt.rc('figure', titlesize=font_size)   # fontsize of the figure title


def interpolate_transparency(foreground, alpha, background=(1,1,1)):
    (fr, fg, fb, *_), (br, bg, bb, *_) = foreground, background
    
    return (
        alpha * fr + (1 - alpha * br),    
        alpha * fg + (1 - alpha * bg),
        alpha * fb + (1 - alpha * bb),
    )