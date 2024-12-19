import yt
import argparse
import numpy as np
import pandas as pd
from unyt import unyt_array, unyt_quantity
import matplotlib.pyplot as plt

import src.explorer as dge
from src.explorer.class_methods import load_ftable




def parse_args():
    parser = argparse.ArgumentParser(description="Process various input parameters for making profile plots of a given halo. You MUST provide the halo sub_tree_id and a list of snapshot numbers.\n The binning of the profiles is performed logarithmically over the (projected) radii from rmin to rmax")
    
    required = parser.add_argument_group('REQUIRED arguments')
    opt = parser.add_argument_group('OPTIONAL arguments')


    required.add_argument(
        "-i", "--input_file",
        help="Merger Tree file path. Must be csv.",
        required=True
    )
    required.add_argument(
        "-eq", "--equivalence_table",
        type=str,
        help="Equivalence table path. Must have correct formatting.",
        required=True
    )
    required.add_argument(
        "-sn", "--snapshot_numbers",
        nargs="*",
        help="List of snapshot numbers to plot. Maximum 4.",
        required=True
    )





    opt.add_argument(
        "-v", "--volumetric",
        nargs="*",
        default=["stars", "darkmatter", "gas"],
        help='Enable volumetric profiles with a list of components (default: ["stars", "darkmatter", "gas"]).'
    )
    opt.add_argument(
        "-s", "--surface",
        nargs="*",
        default=["stars", "darkmatter"],
        help='Enable surface profiles with a list of components (default: ["stars", "darkmatter"]).'
    )
    opt.add_argument(
        "-er", "--extra_radius",
        nargs="*",
        default=None,
        help="Extra radius as list of [float, str] (Default: halo virial radius extracted from data)."
    )
    opt.add_argument(
        "-n", "--Nproj",
        type=int,
        default=10,
        help="Number of projections to perform, uniformly, over half a sphere (Default: 10)."
    )
    opt.add_argument(
        "-pm", "--projection_mode",
        type=str,
        default="bins",
        help="Type of projection. Only affects projected velocity. 'bins' or 'apertures': 'bins' computes all quantities on radial bins while 'apertures does so in filled apertures' (Default: 'bins')."
    )
    opt.add_argument(
        "-g", "--gas_cm",
        type=str,
        default="darkmatter",
        help="Set center-of-mass properties of gas, given that its nature is clumpy and accurate values are hard to derive. (Default: darkmatter)."
    )

    
    opt.add_argument(
        "-o", "--output",
        type=str,
        default="./",
        help="Output folder (Default: ./)."
    )
    opt.add_argument(
        "-dbf", "--double_fit",
        type=bool,
        default=False,
        help="Wether to use the same fit for Volumetric and Surface, or produce two different ones. (Default: Uses Volumetric throguhout)."
    )
    opt.add_argument(
        "-rr", "--radii_range",
        nargs="*"
        type=float,
        default=[0.08, 50, ],
        help="Wether to use the same fit for Volumetric and Surface, or produce two different ones. (Default: Uses Volumetric throguhout)."
    )

    return parser.parse_args()




args = parse_args()


print(f"Volumetric Components: {args.volumetric}")
print(f"Surface Components: {args.surface}")
print(f"Extra Radius: {args.extra_radius}")
print(f"Nproj: {args.Nproj}")
print(f"Gas CM: {args.gas_cm}")
print(f"Input: {args.input_file}")
print(f"Output: {args.output}")


















































