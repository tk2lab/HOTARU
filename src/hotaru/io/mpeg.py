import av


class MpegStream:
    def __init__(self, w, h, hz, outfile):
        self.args = w, h, hz, outfile

    def __enter__(self):
        w, h, hz, outfile = self.args
        container = av.open(outfile, mode="w")
        stream = container.add_stream("h264", rate=20.0)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        self.container = container
        self.stream = stream
        return self

    def write(self, img):
        frame = av.VideoFrame.from_ndarray(img, format="rgba")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def __exit__(self, exc_type, exc_value, traceback):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
