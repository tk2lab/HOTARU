def main():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    # for debug
    tf.debugging.set_log_device_placement(True)
    # tf.config.experimental_run_functions_eagerly(True)
    #tf.autograph.set_verbosity(1, True)

    devs = tf.config.list_physical_devices('GPU')
    # for d in devs:
    #    tf.config.experimental.set_memory_growth(d, True)
    tf.config.set_logical_device_configuration(
        devs[0], [
            tf.config.LogicalDeviceConfiguration(memory_limit=1024*7),
        ],
    )

    from .application import Application
    return Application().run()
