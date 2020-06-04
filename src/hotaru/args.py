import argparse


def get_args(args=None):
    parser = argparse.ArgumentParser(prog='hotaruk')

    g = parser.add_argument_group('main')
    g.add_argument('--job-dir', metavar='DIR', default='.', help='(default: %(default)s)')
    g.add_argument('--start', default=-1, type=int, help='for restart (default: %(default)s)')
    g.add_argument('--debug', action='store_true')

    g = parser.add_argument_group('data')
    g.add_argument('--imgs-file', metavar='FILE', default='imgs.tif')
    g.add_argument('--hz', type=float, default=20.0, help='sampling rate of imgs-file (default: %(defalut)s)')
    g.add_argument('--mask-type', default='0.pad', metavar='TYPE', help='(default: %(default)s)')

    g = parser.add_argument_group('gamma')
    g.add_argument('--tau1', type=float, default=0.08, help='(default: %(default)s)')
    g.add_argument('--tau2', type=float, default=0.16, help='(default: %(default)s)')
    g.add_argument('--ltau', type=float, default=1.00, help='(default: %(default)s)')

    g = parser.add_argument_group('penalty')
    g.add_argument('--la', type=float, default=1.5, help='(default: %(default)s)')
    g.add_argument('--lu', type=float, default=5.0, help='(default: %(default)s)')
    g.add_argument('--sa', type=float, default=100.00, help='(default: %(default)s)')
    g.add_argument('--ru', type=float, default=1.00, help='(default: %(default)s)')
    g.add_argument('--bx', type=float, default=0.00, help='(default: %(default)s)')
    g.add_argument('--bt', type=float, default=0.00, help='(default: %(default)s)')
    g.add_argument('--lu-finish', type=float, default=10.0, metavar='LU', help='(default: %(default)s)')

    g = parser.add_argument_group('image processing')
    g.add_argument('--gauss', type=float, default=2.0, metavar='G', help='(default: %(default)s)')
    g.add_argument('--min-radius', type=float, default=2.0, metavar='R', help='(default: %(default)s)')
    g.add_argument('--max-radius', type=float, default=8.0, metavar='R', help='(default: %(default)s)')
    g.add_argument('--num-radius', type=int, default=10, metavar='N', help='(default: %(default)s)')

    g = parser.add_argument_group('restruction')
    g.add_argument('--thr-gl', type=float, default=0.4, metavar='THR', help='remove weak peaks (default: %(default)s)')
    g.add_argument('--thr-dist', type=float, default=1.8, metavar='THR', help='remove close peaks (default: %(default)s)')
    g.add_argument('--thr-usim', type=float, default=0.90, metavar='THR', help='remove similar signal components (default: %(default)s)')
    g.add_argument('--thr-asim', type=float, default=0.70, metavar='THR', help='remove overlapped components (default: %(default)s)')
    g.add_argument('--thr-weak', type=float, default=0.01, metavar='THR', help='remove weak signal components (default: %(default)s)')
    g.add_argument('--thr-blur', type=float, default=0.01, metavar='THR', help='remove blur shape components (default: %(default)s)')

    g = parser.add_argument_group('algorithm')
    g.add_argument('--epochs-ustep', type=int, default=10, metavar='N', help='(default: %(default)s)')
    g.add_argument('--epochs-astep', type=int, default=10, metavar='N', help='(default: %(default)s)')
    g.add_argument('--steps-ustep', type=int, default=100, metavar='N', help='(default: %(default)s)')
    g.add_argument('--steps-astep', type=int, default=100, metavar='N', help='(default: %(default)s)')
    g.add_argument('--tol-ustep', type=float, default=1e-3, metavar='TOL', help='(default: %(default)s)')
    g.add_argument('--tol-astep', type=float, default=1e-3, metavar='TOL', help='(default: %(default)s)')
    g.add_argument('--scale-ustep', type=float, default=20.0, metavar='S', help='(default: %(default)s)')
    g.add_argument('--scale-astep', type=float, default=20.0, metavar='S', help='(default: %(default)s)')
    g.add_argument('--fac-lr-ustep', type=float, default=0.7, metavar='F', help='(default: %(default)s)')
    g.add_argument('--fac-lr-astep', type=float, default=0.7, metavar='F', help='(default: %(default)s)')
    g.add_argument('--batch-data', type=int, default=100, metavar='B', help='(default: %(default)s)')
    g.add_argument('--batch-clean', type=int, default=100, metavar='B', help='(default: %(default)s)')

    args = parser.parse_args(args)

    def gen_params(**kwargs):
        return {k: getattr(args, v) for k, v in kwargs.items()}

    args.data_params = gen_params(
        imgs_file='imgs_file', hz='hz', mask_type='mask_type', batch='batch_data',
    )
    args.gamma_params = gen_params(
        tau1='tau1', tau2='tau2', ltau='ltau',
    )
    args.penalty_params = gen_params(
        la='la', lu='lu', sa='sa', ru='ru', bx='bx', bt='bt',
    )
    args.penalty_finish = gen_params(
        lu='lu_finish'
    )
    args.peak_params = gen_params(
        gauss='gauss', thr_gl='thr_gl', thr_dist='thr_dist',
        min_radius='min_radius', max_radius='max_radius', num_radius='num_radius',
    )
    args.segment_params = gen_params(
        start='start',
    )
    args.ustep_params = gen_params(
        epochs='epochs_ustep', steps='steps_ustep', scale='scale_ustep',
        tol='tol_ustep', fac_lr='fac_lr_ustep',
        thr_weak='thr_weak', thr_usim='thr_usim',
    )
    args.astep_params = gen_params(
        epochs='epochs_astep', steps='steps_astep', scale='scale_astep',
        tol='tol_astep', fac_lr='fac_lr_astep',
    )
    args.clean_params = gen_params(
        thr_blur='thr_blur', thr_asim='thr_asim', batch='batch_clean',
    )

    return args
