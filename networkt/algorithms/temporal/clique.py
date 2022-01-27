from networkt.core.tarray import TArray
import networkx
import numpy as np
import pandas as pd


def count_cliques(edges, edgetime, timestamps, max_clique_size=8, max_count=-1, count_temporal=False,
                  count_terminal=False):
    from .clique_counter import count_cliques as countcliques

    # nodes = set([node for edge in edges for node in edge])
    # node_index = dict(zip(nodes, range(len(nodes))))
    # indexed_edgelist = [(node_index[edge[0]], node_index[edge[1]])
    #                     for edge in edges]

    results = countcliques(edges, edgetime, timestamps,
                           max_clique_size, max_count, count_temporal, count_terminal)

    if count_temporal:
        results = pd.DataFrame(results)
        results = np.nancumsum(results, axis=0)

    return results


def count_cliques_(edgelist, max_clique_size=8, count_directed=False):
    from .clique_counter import count_cliques as countcliques
    G = networkx.from_edgelist(edgelist, networkx.DiGraph)
    G = networkx.relabel.convert_node_labels_to_integers(G)

    indexed_edgelist = list(G.edges)
    res = countcliques(indexed_edgelist,
                       max_clique_size, count_directed)
    # while res and res[-1] == 0:
    #     res.pop()
    return res

