from hotaru.data import CalciumImagingDataWithStats


def test_movie():
    data = CalciumImagingDataWithStats(
        path='sample/imgs.tif',
        hz=20.0,
    )
    data.calc(batch_size=100)
    data.to_movie('sample/reg.mp4')
