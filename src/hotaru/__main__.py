def app():
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    from .console.main import main

    import sys
    sys.exit(main())
    return main()


if __name__ == "__main__":
    app()
