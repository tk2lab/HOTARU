from hotaru.data import MovieWithStats


def test_movie():
    data = MovieWithStats.load(
        'sample/imgs.tif',
        20.0,
        batch_size=100,
        cache_path='sample/stats.h5',
    )
    data.normalized_movie('sample/normalized.mp4')
