import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from matplotlib import offsetbox


G = nx.Graph()
countries = ["US", "CN", "IN", "RU", "BR"]
G.add_nodes_from(countries)
edges = [("US", "CN"), ("US", "IN"), ("CN", "RU"), ("IN", "BR"), ("BR", "RU")]
G.add_edges_from(edges)

flag_urls = {
    "US": "https://raw.githubusercontent.com/hjnilsson/country-flags/master/png100px/us.png",
    "CN": "https://raw.githubusercontent.com/hjnilsson/country-flags/master/png100px/cn.png",
    "IN": "https://raw.githubusercontent.com/hjnilsson/country-flags/master/png100px/in.png",
    "RU": "https://raw.githubusercontent.com/hjnilsson/country-flags/master/png100px/ru.png",
    "BR": "https://raw.githubusercontent.com/hjnilsson/country-flags/master/png100px/br.png",
}

flags = {}
for country, url in flag_urls.items():
    response = requests.get(url)
    flags[country] = Image.open(BytesIO(response.content)).resize((40, 30), Image.Resampling.LANCZOS)


pos = nx.spring_layout(G)

fig, ax = plt.subplots(figsize=(10, 10))

nx.draw(G, pos, ax=ax, with_labels=False, node_size=0)


for node in G.nodes:
    flag = flags[node]
    imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(flag, zoom=1), pos[node], frameon=False)
    ax.add_artist(imagebox)


for node, (x, y) in pos.items():
    ax.text(x, y+0.06, node, fontsize=12, ha='center', va='center')

plt.show()
