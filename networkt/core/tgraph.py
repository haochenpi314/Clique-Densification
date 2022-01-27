import itertools
import numpy as np
import pandas as pd
# import modin.pandas as pd
from collections import defaultdict
from .tarray import TArray
import time


class TGraph:
    def __init__(self, edges, nodes=None):
        if isinstance(edges, TArray) and isinstance(nodes, TArray):
            self.tnodes = nodes
            self.tedges = edges
            return

        edges = pd.DataFrame(edges)
        nodes = pd.DataFrame(nodes)
        if len(edges.columns) == 3 and len(nodes.columns) == 2 and not nodes.iloc[:, -1].isnull().any() and not edges.iloc[:, -1].isnull().any():
            # standard form: timestamps & original timestamps
            # fast load
            nodes.columns = ['node', 'timestamp']
            edges.columns = ["src_node", "dst_node", 'timestamp']
            self.tnodes = TArray(
                nodes["node"], nodes["timestamp"])
            self.tedges = TArray(
                edges[["src_node", "dst_node"]], edges["timestamp"])
        elif len(edges.columns) == 3:
            edges.columns = ["src_node", "dst_node", 'timestamp']
            self.tedges = TArray(
                edges[["src_node", "dst_node"]], edges["timestamp"])
            self.tnodes = self.__infer_tnodes(edges)
            self.tnodes.reset_index(inplace=True)
            self.tnodes = TArray(
                self.tnodes["node"], self.tnodes["timestamp"])
        else:
            self.__build(edges, nodes)

    # convert original timestamps to sortable timestamps
    def __build(self, edges, nodes=None):
        try:
            edges.rename(columns={0: "src_node", 1: "dst_node",
                                  2: "timestamp"}, inplace=True, errors='raise')
        except:
            edges.rename(
                columns={0: "src_node", 1: "dst_node", 2: "timestamp"}, inplace=True)
            if "timestamp" not in edges:
                edges["timestamp"] = np.nan
        edges["edge"] = list(zip(edges["src_node"], edges["dst_node"]))
        # edges["timestamp"] = edges["timestamp"].apply(
        #     lambda x: natsort.natsort_keygen()(x) if pd.notnull(x) else x)
        # edges = edges.groupby(["edge"]).min()

        try:
            nodes.rename(columns={0: "node", 1: "timestamp"},
                         inplace=True, errors='raise')
        except:
            nodes.rename(columns={0: "node", 1: "timestamp"}, inplace=True)
            if "timestamp" not in nodes:
                nodes["timestamp"] = np.nan
                nodes["node"] = np.nan
        # nodes["timestamp"] = nodes["timestamp"].apply(
        #     lambda x: natsort.natsort_keygen()(x) if pd.notnull(x) else x)
        nodes = nodes.groupby(["node"]).min()

        if nodes["timestamp"].isnull().all() and edges["timestamp"].isnull().all():
            raise Exception("no temporal information is provided")

        inferred_tnodes = self.__infer_tnodes(edges)
        merged_tnodes = self.__merge_tnodes(nodes, inferred_tnodes)
        self.tedges = self.__infer_tedges(edges, merged_tnodes)
        if nodes["timestamp"].isnull().all():
            self.tnodes = merged_tnodes
        else:
            self.tnodes = self.__infer_tnodes(self.tedges)

        self.tnodes.reset_index(inplace=True)
        self.tnodes = self.tnodes.astype({"timestamp": int})
        edges = edges.astype({"timestamp": int})

        self.tnodes = TArray(
            self.tnodes["node"], self.tnodes["timestamp"])

        self.tedges = TArray(
            edges[["src_node", "dst_node"]], edges["timestamp"])

    # infer nodes' timestamps from timestamped edges
    @staticmethod
    def __infer_tnodes(edges):
        tnodes_c0 = edges.loc[:, ["src_node", "timestamp"]]
        tnodes_c1 = edges.loc[:, ["dst_node", "timestamp"]]
        tnodes_c0.rename(columns={"src_node": "node"}, inplace=True)
        tnodes_c1.rename(columns={"dst_node": "node"}, inplace=True)
        inferred_tnodes = pd.concat([tnodes_c0, tnodes_c1], ignore_index=True)
        inferred_tnodes = inferred_tnodes.groupby("node").min()
        return inferred_tnodes

    # update timestamps of inferred_tnodes using nodes provided
    # and fill all nodes whose times are not given with the earliest timestamp
    @staticmethod
    def __merge_tnodes(nodes, inferred_tnodes):
        common_tnodes = nodes.merge(inferred_tnodes, on=["node"])
        if (common_tnodes["timestamp_x"] > common_tnodes["timestamp_y"]).any():
            raise Exception("inconsistent temporal information")
        inferred_tnodes.loc[common_tnodes.index] = nodes.loc[common_tnodes.index]

        # assume nodes of not timestamped edges came first
        min_timestamp = 0
        num_of_nans = len(inferred_tnodes[inferred_tnodes.isnull()])
        df = pd.DataFrame(
            {"timestamp": [min_timestamp] * num_of_nans, "node": inferred_tnodes[inferred_tnodes.isnull()].index})
        df.set_index("node", inplace=True)
        inferred_tnodes[inferred_tnodes.isnull()] = df
        # original_dtype = nodes.dtypes["timestamp"]
        # print(original_dtype)
        # inferred_tnodes = inferred_tnodes.astype({"timestamp": original_dtype})
        return inferred_tnodes

    # infer timestamps of not-timestamped edges' from timestamped nodes
    @staticmethod
    def __infer_tedges(edges, tnodes):
        if edges["timestamp"].isnull().any():
            not_timestamped_edges = edges.loc[edges["timestamp"].isnull()]
            inferred_edges = pd.DataFrame(index=not_timestamped_edges.index)
            inferred_edges["src_time"] = tnodes.lookup(
                not_timestamped_edges["src_node"], len(not_timestamped_edges["timestamp"]) * ["timestamp"])
            inferred_edges["dst_time"] = tnodes.lookup(
                not_timestamped_edges["dst_node"], len(not_timestamped_edges["timestamp"]) * ["timestamp"])

            inferred_edges = inferred_edges.max(axis=1)
            inferred_edges = pd.DataFrame(inferred_edges)
            inferred_edges.rename(columns={0: "timestamp"}, inplace=True)
            edges.fillna(inferred_edges, inplace=True)

        del edges['edge']
        return edges

    def __getitem__(self, slice_):
        if slice_.start != None:
            raise Exception
        tedges = self.tedges[slice_]
        tnodes = self.tnodes[slice_]
        graph = TGraph(tedges, tnodes)
        return graph

    # def labels_to_integers(self):

    # return indices that divide length exponentially
    @staticmethod
    def get_exp_division_indexes(length, base=1.1):
        a1 = length
        a0 = 1
        num_of_sections = np.ceil(
            np.log(a1 / a0) / np.log(base)).astype(int)
        indexes = np.arange(start=1, stop=num_of_sections - 1)
        indexes = a0 * base**indexes
        indexes = np.around(indexes)
        indexes = np.unique(indexes)
        indexes = indexes.astype(int)
        indexes = indexes[indexes < length]
        indexes = np.insert(indexes, 0, 0)
        return indexes

    # return a chunk iterator
    # for temporal batch-processing
    def batch_iterator(self, difference=False, base=1.1, node_indexes=None):
        tedges = self.tedges
        tnodes = self.tnodes

        max_time = max(self.tedges.timestamps) + 1
        if node_indexes is None:
            node_indexes = self.get_exp_division_indexes(len(tnodes), base)
        else:
            node_indexes = [0] + node_indexes

        timestamps = [tnodes.index_to_time(index) for index in node_indexes]
        timestamps = [
            timestamp for timestamp in timestamps if timestamp is not None]
        timestamps = sorted(list(set(timestamps)))

        if difference:
            timestamps = list(zip(timestamps, timestamps[1:] + [max_time, ]))
        else:
            timestamps = list(
                zip(len(timestamps) * [None], timestamps[1:] + [max_time, ]))

        timeslices = [slice(start, stop) for start, stop in timestamps]
        return timeslices

    def labels_to_indices(self):
        nodes = set(self.tnodes)
        node_index = dict(zip(nodes, range(len(nodes))))

        tedges = TArray(np.copy(self.tedges), np.copy(self.tedges.timestamps))
        tnodes = TArray(np.copy(self.tnodes), np.copy(self.tnodes.timestamps))

        for i in range(len(tedges)):
            tedges[i, 0] = node_index[tedges[i, 0]]
            tedges[i, 1] = node_index[tedges[i, 1]]

        for i in range(len(tnodes)):
            tnodes[i] = node_index[tnodes[i]]

        return TGraph(tedges, tnodes)
