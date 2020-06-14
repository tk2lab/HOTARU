def main():
    import os
    import tensorflow as tf

    # for debug
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('WARN')
    #tf.debugging.set_log_device_placement(True)
    # tf.config.experimental_run_functions_eagerly(True)
    #tf.autograph.set_verbosity(1, True)

    devs = tf.config.list_physical_devices('GPU')
    for d in devs:
        tf.config.experimental.set_memory_growth(d, True)

    # limit GPU memory
    tf.config.set_logical_device_configuration(
        devs[0], [
            tf.config.LogicalDeviceConfiguration(memory_limit=1024*7),
        ],
    )

    from .application import Application
    return Application().run()
