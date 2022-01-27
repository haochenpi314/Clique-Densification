import os
import numpy as np
import json
import networkt
from networkt import *
import pandas as pd
from ast import literal_eval as make_tuple
# import modin.pandas as pd


def __infer_sep(filename):
    if filename.endswith(".tsv"):
        return "\t"
    elif filename.endswith(".csv"):
        return ","
    else:
        return r"(?:\s+)"


def load(path_to_dataset):
    import time
    t0 = time.time()
    metadata = None
    nodes = None
    edges = None

    filenames = os.listdir(path_to_dataset)
    if "metadata.json" in filenames:
        filenames.remove("metadata.json")
        with open(os.path.join(path_to_dataset, "metadata.json"), "r+") as json_file:
            metadata = json.load(json_file)
    elif len(filenames) == 1:
        metadata = dict()
        metadata[filenames[0]] = {
            "src_node": 0,
            "dst_node": 1,
            "timestamp": 2
        }
    elif len(filenames) == 2:
        raise Exception("metadata.json is needed to identify file format")

    for filename in filenames:
        if filename not in metadata:
            continue
        full_path = os.path.join(path_to_dataset, filename)
        temp = pd.read_csv(full_path, header=None, sep=__infer_sep(filename),
                           comment='#', engine='python', usecols=list(metadata[filename].values()))

        temp.columns = metadata[filename].keys()
        if "node" in metadata[filename]:
            nodes = temp
        else:
            edges = temp

    numeric_edgetime = (
        "timestamp" not in edges) or edges["timestamp"].apply(np.isreal).all()
    numeric_nodetime = (nodes is None) or (
        "timestamp" not in nodes) or nodes["timestamp"].apply(np.isreal).all()

    if not (numeric_edgetime and numeric_nodetime):
        if "timestamp" in edges:
            edges["timestamp"] = edges["timestamp"].apply(
                lambda x: int(pd.to_datetime(x).timestamp()) if pd.notnull(x) else x)
        if nodes is not None and "timestamp" in nodes:
            nodes["timestamp"] = nodes["timestamp"].apply(
                lambda x: int(pd.to_datetime(x).timestamp()) if pd.notnull(x) else x)

    t1 = time.time()
    print("loading " + path_to_dataset + " took " + str(t1 - t0))
    tgraph = TGraph(edges, nodes)
    t2 = time.time()
    print("building " + path_to_dataset + " took " + str(t2 - t1))
    return tgraph


def save(folderpath, tgraph):
    if isinstance(tgraph, TGraph):
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        # df1 = pd.DataFrame(tgraph.tnodes)
        # df1.to_csv(os.path.join(folderpath, "nodes.tsv"),
        #            header=False, index=False, sep="\t")
        df2 = pd.DataFrame(tgraph.tedges)
        df2['time'] = tgraph.tedges.timestamps
        df2.to_csv(os.path.join(folderpath, "edges.tsv"),
                   header=False, index=False, sep="\t")


def save_hdf(folderpath, tgraph):
    if isinstance(tgraph, TGraph):
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        df1 = pd.DataFrame(tgraph.tnodes)
        df1.columns = ["node"]
        df1["timestamp"] = tgraph.tnodes.timestamps
        df2 = pd.DataFrame(tgraph.tedges)
        df2.columns = ["src_node", "dst_node"]
        df2["timestamp"] = tgraph.tedges.timestamps
        df1.to_hdf(os.path.join(folderpath, "net"), key="nodes")
        df2.to_hdf(os.path.join(folderpath, "net"), key="edges")

# load as is


def load_hdf(path_to_dataset):
    df1 = pd.read_hdf(os.path.join(path_to_dataset, "net"), key="nodes")
    df2 = pd.read_hdf(os.path.join(path_to_dataset, "net"), key="edges")
    G = TGraph(df2, df1)
    return G
