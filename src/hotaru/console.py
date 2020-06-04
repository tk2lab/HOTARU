def main():

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    devs = tf.config.experimental.list_physical_devices('GPU')
    for d in devs:
        tf.config.experimental.set_memory_growth(d, True)

    from .args import get_args
    args = get_args()
    if args.debug:
        tf.config.experimental_run_functions_eagerly(True)

    from .core.model import HotaruModel
    from .util.timer import Timer


    tf.io.gfile.makedirs(args.job_dir + '/out')
    if args.debug:
        model = HotaruModel.build(
            args.job_dir, **args.data_params, **args.gamma_params,
        )
    else:
        model = HotaruModel.load_or_build(
            args.job_dir, **args.data_params, **args.gamma_params,
        )
        with Timer('save'):
            tf.saved_model.save(model, args.job_dir)

    model.get_peak(**args.peak_params)
    with Timer('save'):
        tf.saved_model.save(model, args.job_dir)

    model.get_segment(**args.segment_params)
    with Timer('save'):
        tf.saved_model.save(model, args.job_dir)

    num = lambda i: tf.size(model.history[i]['size']).numpy()
    model.set_penalty(**args.penalty_params)
    while (len(model.history) <= 2) or (num(-3) != num(-2)) or (num(-2) != num(-1)):
        model.ustep(**args.ustep_params)
        with Timer('output'):
            model.save(f'{args.job_dir}/out/{len(model.history):03d}')
        model.astep(**args.astep_params)
        model.clean(**args.clean_params)
        with Timer('save'):
            print([num(i) for i in range(len(model.history))])
            tf.saved_model.save(model, args.job_dir)

    model.set_penalty(**args.penalty_finish)
    model.ustep(**args.ustep_params)
    with Timer('output'):
        model.save(f'{args.job_dir}/out/final')
