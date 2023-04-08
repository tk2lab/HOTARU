from dash import Dash, html
from dash_slicer import VolumeSlicer
import tifffile


app = Dash(__name__, update_title=None)
vol = tifffile.imread("Data3/imgs.tif", out="memmap")
slicer = VolumeSlicer(app, vol)
slicer.graph.config["scrollZoom"] = False
app.layout = html.Div([slicer.graph, slicer.slider, *slicer.stores])
app.run_server(debug=True, port=8888)
