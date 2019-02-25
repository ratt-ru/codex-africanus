#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from os.path import join as pjoin
import subprocess
import sys

import numpy as np

from africanus.rime.examples.predict import (create_parser,
                                             parse_sky_model,
                                             predict)
from africanus.util.requirements import requires_optional

try:
    import pyrap.tables as pt
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None


def create_meq_sky_model(filename, tigger_lsm):
    radec, stokes = parse_sky_model(filename)

    # Create the tigger sky model
    with open(tigger_lsm, 'w') as f:
        f.write('#format: ra_d dec_d i q u v spi freq0\n')

        radec = np.rad2deg(radec)

        for s in range(radec.shape[0]):
            ra, dec = radec[s]
            i, q, u, v = stokes[s]

            # Format line
            line = ('{ra:.20f} {dec:.20f} '
                    '{i} {q} {u} {v} {spi} '
                    '{rf:.20f}\n'.format(ra=ra, dec=dec,
                                         i=i, q=q, u=u, v=v,
                                         spi=0.0, rf=1.0))

            f.write(line)


def run_meqtrees(args):
    # Directory in which meqtree-related files are read/written
    meq_dir = 'meqtrees'

    # Scripts
    meqpipe = 'meqtree-pipeliner.py'

    # Meqtree profile and script
    cfg_file = pjoin(meq_dir, 'tdlconf.profiles')
    sim_script = pjoin(meq_dir, 'turbo-sim.py')
    tigger_lsm = pjoin(meq_dir, 'tigger_lsm.txt')

    cfg_section = '-'.join(('codex', 'compare', args.feed_type))
    meqtrees_vis_column = "CORRECTED_DATA"

    # Find the location of the meqtree pipeliner script
    meqpipe_actual = subprocess.check_output(['which', meqpipe]).strip()

    create_meq_sky_model(args.sky_model, tigger_lsm)

    print(meqpipe_actual)

    # ========================================
    # Call MeqTrees
    # ========================================

    cmd_list = [
        'python',
        # Meqtree Pipeline script
        meqpipe_actual,
        # Configuration File
        '-c', cfg_file,
        # Configuration section
        '[{section}]'.format(section=cfg_section),
        # Measurement Set
        'ms_sel.msname={ms}'.format(ms=args.ms),
        # Tigger sky file
        'tiggerlsm.filename={sm}'.format(sm=tigger_lsm),
        # Output column
        'ms_sel.output_column={c}'.format(c=meqtrees_vis_column),
        # Imaging Column
        'img_sel.imaging_column={c}'.format(c=meqtrees_vis_column),
        # Enable the beam?
        'me.e_enable = {e}'.format(e=0),
        # Beam FITS file pattern
        # 'pybeams_fits.filename_pattern={p}'.format(p=beam_file_schema),
        # FITS L and M AXIS
        # 'pybeams_fits.l_axis={l}'.format(l=l_axis),
        # 'pybeams_fits.m_axis={m}'.format(m=m_axis),
        sim_script,
        '=simulate'
    ]

    # Call the meqtrees simulation script,
    # dumping visibilities into meqtrees_vis_column
    subprocess.call(cmd_list)


@requires_optional('pyrap.tables', opt_import_error)
def compare_columns(args, codex_column, meqtrees_column):
    with pt.table(args.ms) as T:
        codex_vis = T.getcol(codex_column)
        # Meqtrees visibilities are conjugated compared to codex
        # Codex used e^(-2*pi*1j*(l*u + m*v + (n-1)*w)
        # Meqtrees used e^(2*pi*1j*(l*u + m*v + (n-1)*w)
        meqtrees_vis = np.conj(T.getcol(meqtrees_column))

        # Compare
        close = np.isclose(meqtrees_vis, codex_vis)
        not_close = np.invert(close)
        problems = np.nonzero(not_close)

        # Everything agrees, exit
        if problems[0].size == 0:
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
        it = enumerate(itertools.izip(*t))
        it = itertools.islice(it, 0, 1000, 1)

        # Write out the problematic visibilities to file
        with open(bad_vis_file, 'w') as f:
            for i, (p, mb, meq, d, amp) in it:
                f.write("{i} {t} Codex Africanus: {mb} MeqTrees: {meq} "
                        "Difference {d} Absolute Difference {ad} \n"
                        .format(i=i, t=p, mb=mb, meq=meq, d=d, ad=amp))

        return False


if __name__ == "__main__":
    args = create_parser().parse_args()
    run_meqtrees(args)
    predict(args)
    if not compare_columns(args, "MODEL_DATA", "CORRECTED_DATA"):
        sys.exit(1)
