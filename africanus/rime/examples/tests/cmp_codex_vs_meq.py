#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import itertools
from os.path import join as pjoin
from pathlib import Path
import sys
import tempfile

import numpy as np

from africanus.rime.examples.predict import (predict,
                                             create_parser as predict_parser)
from africanus.util.requirements import requires_optional
from africanus.testing.beam_factory import beam_factory

try:
    import pyrap.tables as pt
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None


@requires_optional('pyrap.tables', opt_import_error)
def inspect_polarisation_type(args):
    linear_corr_types = set([9, 10, 11, 12])
    circular_corr_types = set([5, 6, 7, 8])

    discovered_corr_types = set()

    with pt.table("::".join((args.ms, "POLARIZATION")), ack=False) as pol:
        for row in range(pol.nrows()):
            corr_types = pol.getcol("CORR_TYPE", startrow=row, nrow=1)
            discovered_corr_types.update(corr_types[0])

    if discovered_corr_types.issubset(linear_corr_types):
        return "linear"
    elif discovered_corr_types.issubset(circular_corr_types):
        return "circular"

    raise ValueError("MS Correlation types are not wholly "
                     "linear or circular: %s" % discovered_corr_types)


def cmp_script_factory(args, pol_type):
    beam_pattern = args.beam.replace("$", r"\$")

    return ["python",
            "cmp_codex_vs_meq.py",
            args.ms,
            "-sm " + args.sky_model,
            '-b "' + beam_pattern + '"',
            "--run-predict"]


def meqtrees_command_factory(args, pol_type):
    # Directory in which meqtree-related files are read/written
    meq_dir = 'meqtrees'

    # Scripts
    meqpipe = 'meqtree-pipeliner.py'

    # Meqtree profile and script
    cfg_file = pjoin(meq_dir, 'tdlconf.profiles')
    sim_script = pjoin(meq_dir, 'turbo-sim.py')

    meqtrees_vis_column = "CORRECTED_DATA"

    if pol_type == 'linear':
        cfg_section = '-'.join(('codex', 'compare', 'linear'))
    elif pol_type == 'circular':
        cfg_section = '-'.join(('codex', 'compare', 'circular'))
    else:
        raise ValueError("pol_type %s not in ('circular', 'linear')"
                         % pol_type)

    # $ is a special pattern is most shells, escape it
    beam_pattern = args.beam.replace("$", r"\$")

    # ========================================
    # Setup MeqTrees Call
    # ========================================

    cmd_list = [
        # Meqtree Pipeline script
        '$(which {})'.format(meqpipe),
        # Configuration File
        '-c', cfg_file,
        # Configuration section
        '"[{section}]"'.format(section=cfg_section),
        # Measurement Set
        'ms_sel.msname={ms}'.format(ms=args.ms),
        # Tigger sky file
        'tiggerlsm.filename={sm}'.format(sm=args.sky_model),
        # Output column
        'ms_sel.output_column={c}'.format(c=meqtrees_vis_column),
        # Enable the beam?
        'me.e_enable={e}'.format(e=1),
        # Enable feed rotation
        'me.l_enable={e}'.format(e=1),
        # Beam FITS file pattern
        'pybeams_fits.filename_pattern="{p}"'.format(p=beam_pattern),
        # FITS L and M AXIS
        'pybeams_fits.l_axis={lax}'.format(lax='X'),
        'pybeams_fits.m_axis={max}'.format(max='Y'),
        sim_script,
        '=simulate'
    ]

    return cmd_list


@requires_optional('pyrap.tables', opt_import_error)
def compare_columns(args, codex_column, meqtrees_column):
    with pt.table(args.ms) as T:
        codex_vis = T.getcol(codex_column)
        meqtrees_vis = T.getcol(meqtrees_column)

        # Compare
        close = np.isclose(meqtrees_vis, codex_vis)
        not_close = np.invert(close)
        problems = np.nonzero(not_close)

        # Everything agrees, exit
        if problems[0].size == 0:
            print("Codex Africanus visibilities agrees with MeqTrees")
            return True

        bad_vis_file = 'bad_visibilities.txt'

        # Some visibilities differ, do some analysis
        print("Codex Africanus differs from MeqTrees by {nc}/{t} "
              "visibilities. Writing them out to '{bvf}'"
              .format(nc=problems[0].size, t=not_close.size, bvf=bad_vis_file))

        mb_problems = codex_vis[problems]
        meq_problems = meqtrees_vis[problems]
        difference = mb_problems - meq_problems
        amplitude = np.abs(difference)

        pidx = np.asarray(problems).T

        # Create an iterator over the first 100 problematic visibilities
        t = (pidx, mb_problems, meq_problems, difference, amplitude)
        it = enumerate(zip(*t))
        it = itertools.islice(it, 0, 1000, 1)

        # Write out the problematic visibilities to file
        with open(bad_vis_file, 'w') as f:
            for i, (p, mb, meq, d, amp) in it:
                f.write("{i} {t} Codex Africanus: {mb} MeqTrees: {meq} "
                        "Difference {d} Absolute Difference {ad} \n"
                        .format(i=i, t=p, mb=mb, meq=meq, d=d, ad=amp))

        return False


def create_beams(schema, pol_type):
    td = tempfile.mkdtemp(prefix="beams-")

    path = Path(td, schema)
    filenames = beam_factory(polarisation_type=pol_type,
                             schema=path, npix=257)
    return path, filenames


def cmp_create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--run-predict", action="store_true")

    return p


def compare():
    cmp_args, remaining = cmp_create_parser().parse_known_args()
    args = predict_parser().parse_args(remaining)

    args.invert_uvw = True

    if cmp_args.run_predict:
        predict(args)
    else:
        print("SETTING UP COMPARISON")

        # Zero comparison columns
        with pt.table(args.ms, readonly=False, ack=False) as T:
            nrows = T.nrows()

            for r in range(0, nrows, 10000):
                nrow = min(r + 10000, nrows)

                exemplar = T.getcol("MODEL_DATA", startrow=r, nrow=nrow)
                T.putcol("MODEL_DATA", np.zeros_like(exemplar))
                T.putcol("CORRECTED_DATA", np.zeros_like(exemplar))

        pol_type = inspect_polarisation_type(args)
        beam_path, filenames = create_beams("beams_$(corr)_$(reim).fits",
                                            pol_type)
        args.beam = str(beam_path)
        meq_cmd = " ".join(meqtrees_command_factory(args, pol_type))
        cmp_cmd = " ".join(cmp_script_factory(args, pol_type))

        print("\nRUN THE FOLLOWING COMMAND IN A SEPARATE "
              "MEQTREES ENVIRONMENT, PREFERABLY BUILT FROM SOURCE\n"
              "https://github.com/ska-sa/meqtrees/wiki/BuildFromSource"
              "\n\n%s\n\n\n" % meq_cmd)

        print("\nTHEN RUN THIS IN THE CURRENT ENVIRONMENT"
              "\n\n%s\n\n\n" % cmp_cmd)

        return True

    if not compare_columns(args, "MODEL_DATA", "CORRECTED_DATA"):
        sys.exit(1)


if __name__ == "__main__":
    compare()
