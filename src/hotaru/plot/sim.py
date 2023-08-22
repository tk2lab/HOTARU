for k0i, k1i in zip(k0[:10], k1[:10]):
    print(k0i, k1i, d[k0i, k1i])
    fig = go.Figure()
    y0, x0 = y[k0i], x[k0i]
    y1, x1 = y[k1i], x[k1i]
    ym, xm = (y0 + y1) // 2, (x0 + x1) // 2
    print(y0, x0, y1, x1)
    fig.add_trace(
        go.Heatmap(z=1.0*(seg[k0i, max(ym-30, 0):ym+31, max(xm-30, 0):xm+31]>0.5), opacity=0.8, colorscale="Blues", zmin=0, zmax=1, showscale=False)
    )
    fig.add_trace(
        go.Heatmap(z=1.0*(seg[k1i, max(ym-30, 0):ym+31, max(xm-30, 0):xm+31]>0.5), opacity=0.6, colorscale="Reds", zmin=0, zmax=1, showscale=False)
    )
    fig.update_xaxes(
        showticklabels=False,
    )
    fig.update_yaxes(
        showticklabels=False,
    )
    fig.update_layout(
        width=500,
        height=500,
    )
    fig.show()
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=spk[[k0i, k1i]], colorscale="Reds", showscale=False)
    )
    fig.add_trace(
        go.Scatter(x=[x0-xm, x1-xm], y=[y0-ym, y1-ym], text=[k0i, k1i]),
    )
    """
    fig.update_xaxes(
        #title_text="time (sec)",
        tickmode="array",
        tickvals=[0, 400, 800, 1200],
        ticktext=["0", "20", "40", "60"],
    )
    """
    fig.update_yaxes(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=[str(k0i), str(k1i)],
    )
    fig.update_layout(
        width=500,
        height=200,
    )
    fig.show()