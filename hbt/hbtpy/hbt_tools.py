import argparse
from functools import wraps
import os
from time import time

from plottery.plotutils import savefig

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f'Running {func.__name__!r}...')
        to = time()
        value = func(*args, **kwargs)
        m = (time()-to) / 60
        s = (60*m) % 60
        m = int(m)
        print(f'Finished {func.__name__!r} in {m:02d}m{1:02d}s')
        return value
    return wrapper


def parse_args(parser=None):
    """Parse command-line arguments

    Parameters
    ----------
    parser : `argparse.ArgumentParser`, optional
        pass this parameter if, for instance, you add command-line
        arguments to your own script beyond the basic ones defined
        in `read_args`. Otherwise the basic arguments will be loaded.

    Returns
    -------
        args : output of `parser.parse_args`
    """
    if parser is None:
        parser = read_args()
    args = parser.parse_args()
    return args


def read_args():
    """Set up the base command-line arguments

    Call this function from any program if there are additional
    command-line arguments in it (to be added manually from that
    program)

    Returns
    -------
    parser : `argparse.ArgumentParser` object
    """
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('--ncores', dest='ncores', default=1, type=int)
    add('simulation', default='LR')
    return parser


def save_plot(fig, output, sim, **kwargs):
    if '/' in output:
        path = os.path.join(sim.plot_path, os.path.split(output)[0])
        if not os.path.isdir(path):
            os.makedirs(path)
    out = os.path.join(sim.plot_path, '{0}.pdf'.format(output))
    savefig(out, fig=fig, **kwargs)
    return
