def app():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    devs = tf.config.list_physical_devices('GPU')
    for d in devs:
        tf.config.experimental.set_memory_growth(d, True)

    from .main import main
    return main()
