
from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.subhalo import Subhalos
from hbtpy.track import Track


def main():
    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation)
    reader = HBTReader(sim.path)