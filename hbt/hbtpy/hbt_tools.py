import argparse
from functools import wraps
from icecream import ic
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


def parse_args(parser=None, args=None):
    """Parse command-line arguments

    Parameters
    ----------
    parser : `argparse.ArgumentParser`, optional
        pass this parameter if, for instance, you add command-line
        arguments to your own script beyond the basic ones defined
        in `read_args`. Otherwise the basic arguments will be loaded.
    args : `list`-like, optional
        additional arguments to include in the parser. Each element in
        ``args`` should contain two elements: the string(s) enabling the
        argument and the kwargs to add it to the parser. For instance,
            args=(('--foo', {'type': int, 'default': 1}),
                  ('--bar', {'action': 'store_true'}))

    Returns
    -------
        args : output of `parser.parse_args`
    """
    if parser is None:
        parser = read_args()
    if args is not None:
        if isinstance(args[0], str):
            args = (args,)
        for argname, kwargs in args:
            parser.add_argument(argname, **kwargs)
    args = parser.parse_args()
    if not args.debug:
        ic.disable()
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
    add('--debug', dest='debug', action='store_true')
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
    return out
