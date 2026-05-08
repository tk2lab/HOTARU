from hotaru.data import MovieWithStats


def test_data():
    MovieWithStats.load('sample/imgs.tif', 20.0, batch_size=100)
