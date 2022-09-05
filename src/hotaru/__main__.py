def app():
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    import sys

    from .console.main import main

    sys.exit(main())
    return main()


if __name__ == "__main__":
    app()
