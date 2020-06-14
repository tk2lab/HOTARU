def main():
    import os
    import tensorflow as tf

    limit_gpu_memory = None

    # for debug
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('WARN')
    #tf.debugging.set_log_device_placement(True)
    # tf.config.experimental_run_functions_eagerly(True)
    #tf.autograph.set_verbosity(1, True)

    # limit GPU memory
    devs = tf.config.list_physical_devices('GPU')
    if limit_gpu_memory is not None:
        tf.config.set_logical_device_configuration(
            devs[0], [
                tf.config.LogicalDeviceConfiguration(memory_limit=1024*7),
            ],
        )
    else:
        for d in devs:
            tf.config.experimental.set_memory_growth(d, True)

    from .application import Application
    return Application().run()
