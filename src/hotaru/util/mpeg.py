import numpy as np
import ffmpeg


class MpegStream:

    def __init__(self, w, h, hz, outfile):
        self.args = w, h, str(int(hz)), outfile

    def write(self, data):
        self.process.stdin.write(data.tobytes())

    def __enter__(self):
        w, h, hz, outfile = self.args
        self.process = (
            ffmpeg
            .input('pipe:', f='rawvideo', pix_fmt='rgba', s=f'{w}x{h}')
            .output(outfile, vcodec='libx264', pix_fmt='yuv420p', r=hz)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.process.stdin.close()
        self.process.wait()
