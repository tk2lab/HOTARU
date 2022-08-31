def app():
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    from .main import main

    return main()
