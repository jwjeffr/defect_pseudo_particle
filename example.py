#!/usr/bin/python

"""
Example script using the apdap library to create a new LAMMPS dump file with a vacancy as a pseudo-particle
"""


import ovito
from apdap import create_combined_pipeline


def main():

    combined_pipeline = create_combined_pipeline('transV.dump', rmsd_cutoff=0.12, threshold=5.0, defect_type=4)
    ovito.io.export_file(
        combined_pipeline,
        'transV_mod.dump',
        format='lammps/dump',
        multiple_frames=True,
        columns=['Particle Identifier', 'Particle Type', 'Position.X', 'Position.Y', 'Position.Z'],
    )


if __name__ == '__main__':

    main()
