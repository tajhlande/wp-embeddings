import logging

from pyvis.network import Network

from database import get_cluster_final_topics, get_sql_conn

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

"""
Requires graphviz to be installed

    sudo apt install graphviz

or

    brew install graphviz

then pygraphviz has to be installed with specific config settings
e.g. on Mac:

    uv pip install --no-cache-dir --verbose pygraphviz \
    --config-settings=--global-option=build_ext \
    --config-settings=--global-option="-I$(brew --prefix graphviz)/include" \
    --config-settings=--global-option="-L$(brew --prefix graphviz)/lib"

"""

namespace = "enwiki_namespace_0"

sqlconn = get_sql_conn(namespace)
rows = get_cluster_final_topics(sqlconn, namespace)

colors = [
    "#FF0000",  # Red
    "#33FF00",  # Spring‑green
    "#0066FF",  # Azure
    "#FF0099",  # Rose
    "#CCFF00",  # Chartreuse
    "#00FFFF",  # Cyan
    "#CC00FF",  # Magenta
    "#FF9900",  # Orange
    "#00FF66",  # Lime green
    "#3300FF"   # Violet
]

net = Network(height="1600px", width="100%", notebook=False, directed=True)
for parent_id, node_id, final_label, depth in rows:
    if parent_id is None:
        final_label = "English Wikipedia"
    color_index = depth % len(colors)
    color = colors[color_index]
    net.add_node(node_id, label=final_label or f"<{node_id}>", size=10, color=color)
    if parent_id is not None:
        net.add_edge(parent_id, node_id)

html_content = net.generate_html()
with open("cluster_tree.html", "w") as f:
    f.write(html_content)

print("Open cluster_tree.html in your browser")
