import os
import sys
sys.path.append("../")
import pandas as pd
import networkt
# from utils import *
import numpy as np
import uuid


def pivoter(edgelist, prefix='', check_early_stop=None):
    import time
    name = prefix + uuid.uuid4().hex
    graph_filepath = './graphs/graph_pivoter_' + str(name) + '.txt'
    if 'Pivoter' not in os.getcwd():
        os.chdir('./Pivoter/')

    num_edge = sanitize(edgelist, name)
    import subprocess
    res_ = [1., 1.]

    args = ["./bin/degeneracy_cliques",
            "-i", graph_filepath, "-t", "A", '-k', '0', '-d', '0']

    try:
        if check_early_stop:
            p = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE)
            while True:
                if check_early_stop():
                    return None
                if p.poll() is not None:
                    break
                time.sleep(1)
            res = p.communicate()[0]
        else:
            res = subprocess.check_output(args, shell=False)
        res = res.split(b'\n\n')[1]
        res = res.split(b'\n')
        res_ = []
        for line in res:
            res_.append(float(line.split(b', ')[1]))
        res_.pop(0)
    except:
        pass

    if res_[1] != float(num_edge):
        os.chdir('../')
        raise Exception('{}, {}'.format(graph_filepath, res_[:5]))

    try:
        os.remove(graph_filepath)
    except:
        print('failed to remove {}'.format(graph_filepath))
    finally:
        os.chdir('../')
    return res_


def sanitize(edgelist, name):
    out_name = './graphs/graph_pivoter_' + \
        str(name) + '.txt'  # Output file name
    f_output = open(out_name, 'w')   # Output file ptr
    out_list = set()
    ind = 0
    mapping = dict()
    for tokens in edgelist:
        if tokens[0] == tokens[1]:
            continue
        if tokens[0] in mapping:    # If first node name is already mapped
            # Get numeric map value of first node name
            node1 = mapping[tokens[0]]
        else:
            mapping[tokens[0]] = ind    # Introduce new node to mapping
            node1 = ind
            ind = ind + 1                 # Increment the index
        if tokens[1] in mapping:    # Perform same task for second node name
            node2 = mapping[tokens[1]]
        else:
            mapping[tokens[1]] = ind
            node2 = ind
            ind = ind + 1

        out_list.add((min(node1, node2), max(node1, node2)))

    f_output.write(str(ind) + ' ')
    f_output.write(str(len(out_list)) + '\n')

    for node1, node2 in out_list:
        f_output.write(str(node1) + ' ' + str(node2) + '\n')

    f_output.close()
    return len(out_list)


def clique_num_to_df(results):
    name_mapping = dict(zip(list(range(10000)), ['#nodes'] +
                            ["#clique-" + str(i) for i in range(2, 10000)]))
    df = pd.DataFrame(results)
    df.rename(columns=name_mapping, inplace=True)
    return df


def get_clique_num(df):
    df.fillna(0, inplace=True)
    df = df.iloc[:, :]
    x = df.iloc[:, 0]
    y = df.iloc[:, 1:]

    return np.array(x), y.to_numpy()


def temporal_pivoter(G, node_indexes=None, prefix=''):
    from time import time
    for i, tspan in enumerate(G.batch_iterator(node_indexes=node_indexes)):
        subG = G[tspan].labels_to_indices()
        num_of_cliques = pivoter(subG.tedges, prefix)
        # row = np.array(num_of_cliques)
        # mean_size = sum(row *
        #                 np.array(range(2, len(row) + 2))) / sum(row)
        yield num_of_cliques
