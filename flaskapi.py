import dash
from flask import Flask, request
from flask_cors import CORS

import Recommander
from models import Encoders
from utils import pickleIO, params

server = Flask(__name__)
CORS(server)
app = dash.Dash(__name__, server=server)


clip = Encoders.MultiModalClip()
meme_features = pickleIO.loadFeatures(params.meme_feature_path)
@server.route('/meme', methods=['GET'])
def getMeme():
    print('[Flask]: getMeme Starts')
    msg = request.get_json()['message']
    return dict(result = Recommander.recommand(msg))


# Helper functions
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

# dataroot = '/home/seungchan/Desktop/Projs/kakao/Kakao_ML/backend/datasets/1-year_en/images'
dataroot = './datasets/2022_02_all/images'
# galaxy_path = './result/test_512/galaxy.pkl'
galaxy_path = os.path.join('./result', expr_name, 'plotly.pkl')
galaxy = vis_hover_img_test(galaxy_path, split_ratio=10)
print('Clustering: start')
# tsne = TSNE(n_components=3, random_state=3, n_iter=2500, learning_rate='auto', n_jobs=4).fit_transform(galaxy['feature'])
tsne = TSNE(n_components=3, random_state=3, n_iter=4000, learning_rate='auto', n_jobs=4).fit_transform(galaxy['feature'])
print('Clustering: complete')
del galaxy['feature']

home = deepcopy(galaxy['color'][:])
fig = go.Figure(data=[
    go.Scatter3d(
        x=tsne[:, 0],
        y=tsne[:, 1],
        z=tsne[:, 2],
        mode='markers',
        marker=dict(
            size=10,
            color=galaxy['color'],
        ),
    )
])
# del tsne
print('Rendering: complete')

fig.update_traces(
    hoverinfo="none",
    hovertemplate=None,
)

app.layout = html.Div(
    className="container",
    children=[
        dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True, style={'height': '100vh'}),
        dcc.Tooltip(id="graph-tooltip-5", direction='bottom', style={'margin': '30px'}),
        html.Div(id='hidden-div', style={'display': 'none'})
    ],
)

@app.callback(
    Output("graph-5", "figure"),
    Input("graph-5", "clickData"),
)
def update_chart(clickData):
    # print(clickData)
    print('clicked!')
    if clickData is None:
        return no_update
    if clickData["points"][0]['marker.color']=='rgba(255,255,255,0.3)':
        fig.update_traces(
            dict(marker=dict(
                color=home
            ))
        )
        return fig
    if clickData is not None:
        c = list(fig.data[0].marker.color)
        clan = clickData["points"][0]['marker.color']
        other = 'rgba(255,255,255,0.3)'
        new_c = [color if color==clan else other for color in c]
        fig.update_traces(
            dict(marker=dict(
                color=new_c,
            ))
        )
        return fig


@app.callback(
    Output("graph-tooltip-5", "show"),
    Output("graph-tooltip-5", "bbox"),
    Output("graph-tooltip-5", "children"),
    Input("graph-5", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    num = hover_data["pointNumber"]

    im_matrix = np.array( Image.open(os.path.join(dataroot, galaxy['fname'][num])) )
    im_url = np_image_to_base64(im_matrix)
    children = [
        html.Div([
            html.Img(
                src=im_url,
                style={"height": "256px", 'display': 'block', 'margin': '0 auto'},
            ),
            html.P("Cluster " + str(galaxy['cluster_id'][num]), style={'fontWeight': 'bold'})
        ]),
    ]

    return True, bbox, children

if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=6006)
