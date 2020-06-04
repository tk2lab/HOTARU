from .application import Application


def main():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    return Application().run()
