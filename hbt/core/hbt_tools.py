import argparse


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





