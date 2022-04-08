from hotaru.sim.mpeg import make_mpeg


if __name__ == '__main__':
    import sys
    make_mpeg(sys.argv[1] if len(sys.argv) >= 2 else 'default')
