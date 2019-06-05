import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd
import libpysal
from libpysal.weights.contiguity import Queen
from splot.libpysal import plot_spatial_weights

path = 'powiaty/powiaty.shp'
# https://github.com/Toblerity/Fiona/issues/548
gdf = gpd.read_file(path)
print(gdf.head())

weights = Queen.from_dataframe(gdf)

plot_spatial_weights(weights, gdf)
plt.show()

contig_matrix = libpysal.weights.Rook.from_shapefile(path)

# build list of edges - this will create edges going both ways from connected nodes, so you might need to remove duplicates
nodes = contig_matrix.weights.keys()  # to get dict of keys, alternatively use contig_matrix.id2i.keys()
edges = [(node, neighbour) for node in nodes for neighbour in contig_matrix[node]]
my_graph = nx.Graph(edges)

nx.draw(my_graph)  # networkx draw()
plt.draw()  # pyplot draw()
plt.show()
