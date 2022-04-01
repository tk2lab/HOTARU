import numpy as np
import ffmpeg


class MpegStream:

    def __init__(self, w, h, hz, outfile, filters=None):
        self.args = w, h, str(int(hz)), outfile
        self.filters = filters or []

    def write(self, data):
        self.process.stdin.write(data.tobytes())

    def __enter__(self):
        w, h, hz, outfile = self.args
        p = ffmpeg.input('pipe:', f='rawvideo', pix_fmt='rgba', s=f'{w}x{h}', r=hz)
        for a, b in self.filters:
            if a == 'drawtext':
                p = p.drawtext(**b)
        self.process = (
            p
            .output(outfile, vcodec='libx264', pix_fmt='yuv420p', r=hz)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.process.stdin.close()
        self.process.wait()
