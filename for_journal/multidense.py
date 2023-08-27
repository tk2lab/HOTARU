from hotaru.plot.densesig import dense_sig_multi_fig


paths = dict(
    u0="outputs/Kd32/miniature/min31/clip2x2/u0/default/dup9/",
    u40="outputs/Kd32/miniature/min31/clip2x2/u40/default/dup9/",
)
stages = [[0, 1, 17], [0, 1, 14]]


fig = dense_sig_multi_fig(paths, stages, thr_udense=1.0)
fig.write_image("figs/Kd32-dense-sig.pdf")
