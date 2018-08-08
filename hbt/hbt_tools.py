import argparse


def parse_args():
    args = setup_parser()
    #args = setup_args(args)
    return args


def setup_parser():
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('simulation', default='LR')
    args = parser.parse_args()
    return args




