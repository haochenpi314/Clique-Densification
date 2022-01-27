import matplotlib
import numpy as np
import pandas as pd
import networkx as nx
import sys
sys.path.append("../")
import networkt
import os
import json
import uuid
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, NullFormatter, MultipleLocator, MaxNLocator, FormatStrFormatter, LogLocator


def snap_forest_fire_graph(p, r, n, target_m):
    import subprocess
    FNULL = open(os.devnull, 'w')

    name = uuid.uuid4().hex
    fname = 'bins/graph_' + name + '.txt'

    args = ["./bins/forestfire", "-o:" + fname, "-n:" +
            str(int(n)), '-f:' + str(p), '-b:' + str(r), '-m:' + str(int(target_m))]

    subprocess.call(args, stdout=FNULL)
    f = open(fname, 'r')
    edges = f.readlines()[4:]
    edges = [list(map(int, line.split())) for line in edges]
    f.close()
    os.remove(fname)
    return edges


MODELS = [networkt.redner_graph,
          networkt.threshold_padme_graph, snap_forest_fire_graph]
MODELS = {v.__name__: v for v in MODELS}

DISPLAY_NAME = {'redner_graph': 'Lambiotte et al., 2016',
                'threshold_padme_graph': 'PCM',
                'snap_forest_fire_graph': 'Forest Fire \n(Leskovec et al., 2007)'}

COLOR_MAPPING = {'redner_graph': 'skyblue',
                 'threshold_padme_graph': 'seagreen',
                 'snap_forest_fire_graph': 'indianred'}


def format_name(dataset_name, skip_non_paras=False, skip_paras=False):
    found = 0
    dataset_name_original = dataset_name
    for key in DISPLAY_NAME:
        if dataset_name.startswith(key):
            found += 1
            dataset_name = dataset_name.replace(key, DISPLAY_NAME[key])
    if found != 1:
        return dataset_name_original
    import re
    if skip_non_paras:
        dataset_name = re.sub(r'_n_\d+', '', dataset_name)
        dataset_name = re.sub(r'_repeat_\d+', '', dataset_name)
    if skip_paras:
        dataset_name = re.sub(r'_[a-z]_\d+.*\d*', '', dataset_name)
    dataset_name = re.sub(
        r'_(([a-z]+){1})_', ', \\1=', dataset_name).replace('_', ' ').replace(', p', ' (p')

    return dataset_name


def get_color(dataset_name):
    for key in COLOR_MAPPING:
        if dataset_name.startswith(key):
            return COLOR_MAPPING[key]
    return "grey"


def parse_model_type(dataset_name):
    for key in DISPLAY_NAME:
        if key in dataset_name:
            return key
    return dataset_name


def parse_model_paras(dataset_name):
    import re
    paras = dict(re.findall("(?:_([a-z])_)(\d+\.*\d+)", dataset_name))
    paras = dict(sorted(paras.items(), key=lambda item: item[0]))
    paras = {k: float(v) for k, v in paras.items()}
    return paras


def format_para_str(paras):
    weights = {'p': 0, 'r': 1, 'n': 2}
    return '_'.join(['_'.join(str(val) for val in it)
                     for it in sorted(paras.items(), key=lambda args:weights[args[0]], reverse=False)])


def generic_gen_func(args):
    func, paras, used_filenames, lock = args
    path_to_raw = get_path_to("raw")
    from glob import glob
    paras_ = dict(paras)
    del paras_['n']
    fn = os.path.join(path_to_raw, func.func.__name__ +
                      '_' + format_para_str(paras_)) + '_*'
    if len(glob(fn)) >= 5:
        return
    edges = func(**paras)
    paras['n'] = len({v for e in edges for v in e})
    edges = [list(edge) + [max(edge)] for edge in edges]
    repeat = 0
    while True:
        fn = os.path.join(path_to_raw, func.func.__name__ +
                          '_' + format_para_str(paras)) + '_repeat_{}'.format(repeat)
        print(fn)
        lock.acquire()
        if not os.path.exists(fn) and fn not in used_filenames:
            used_filenames.append(fn)
            lock.release()
            break
        repeat += 1
        lock.release()

    tG = networkt.TGraph(edges)
    networkt.save(fn, tG)


def generate_based_on_best_paras(log_dir='logA'):
    from multiprocessing import Pool, Manager
    from functools import partial
    best_fits = json.load(open(os.path.join(log_dir, 'best_fits.json')))
    unique_model_paras_ = set()
    unique_model_paras = []
    to_generate = []
    manager = Manager()
    lock = manager.Lock()
    used_filenames = manager.list()
    for dname in best_fits:
        for mname in best_fits[dname]:
            model_name = parse_model_type(mname)
            paras = parse_model_paras(mname)
            if (model_name, str(paras)) in unique_model_paras_:
                continue
            unique_model_paras_.add((model_name, str(paras)))
            unique_model_paras.append((model_name, paras))

    for model_name, paras in unique_model_paras:
        paras['n'] = 10**6
        for _ in range(5):
            to_generate.append(
                [partial(MODELS[model_name], target_m=10**6), paras, used_filenames, lock])
    pool = Pool()
    list(pool.imap(generic_gen_func, to_generate))


def counter(dataset_path):
    path_to_stats = get_path_to("statistics")
    name = os.path.basename(dataset_path)
    if os.path.exists(os.path.join(path_to_stats, name)):
        return
    print("processing " + name)
    G = networkt.load(dataset_path)
    from clique_counter import temporal_pivoter, clique_num_to_df
    results = []
    import time
    t0 = time.perf_counter()
    for nums in temporal_pivoter(G):
        results.append(nums)
        if (time.perf_counter() - t0) / (60) > 10:
            break
        t0 = time.perf_counter()
    df = clique_num_to_df(results)
    results_filename = os.path.join(
        path_to_stats, name, "num_of_cliques" + ".csv")
    if not os.path.exists(os.path.dirname(results_filename)):
        os.makedirs(os.path.dirname(results_filename))
    df.to_csv(results_filename, index=False)


def count_cliques():
    import multiprocessing
    pool = multiprocessing.Pool()
    pool.map(counter, get_datasets("raw"))


def get_all_instances(model_para_str, path_type='raw'):
    path_to_raw = get_path_to(path_type)
    from glob import glob
    suffix = '' if len(parse_model_paras(model_para_str)) == 0 else '_'
    return glob(os.path.join(path_to_raw, model_para_str + suffix + '*'))


def cal_delta_deg(G, delta, batch_size):
    from scipy.stats.mstats import gmean, hmean
    G_t0 = G[: delta.start]
    if batch_size > 0:
        batch_idx = batch_size - 1
        if len(G.tnodes[delta.start:].timestamps) < batch_size:
            batch_idx = len(G.tnodes[delta.start:].timestamps) - 1
        t1 = min(G.tnodes[delta.start:].timestamps[batch_idx], delta.stop)
    else:
        t1 = delta.stop

    G_t1 = G[: t1]
    new_nodes = G.tnodes[delta.start:t1]

    nxG_t1 = nx.from_edgelist(G_t1.tedges)

    connected_mean_deg = np.mean([np.mean([d for n, d in nxG_t1.degree(
        list(nxG_t1.neighbors(node)))]) for node in new_nodes])
    sampled_nodes = np.random.choice(
        nxG_t1.nodes, min(len(nxG_t1.nodes), max(100, int(0.1 * len(nxG_t1.nodes)) + 1)))

    mean_deg = np.mean([np.mean([d for n, d in nxG_t1.degree(
        list(nxG_t1.neighbors(node)) + [node])]) for node in sampled_nodes])

    node_num = len(G[:delta.start].tnodes)
    return node_num, mean_deg, connected_mean_deg


def cal_mean_nbr_degree(args):
    filepath, emp_G = args
    d = {}
    instance_name = os.path.basename(filepath)
    if os.path.exists(os.path.join('logA', 'mean_nbr_degree', f'{instance_name}.json')):
        with open(os.path.join('logA', 'mean_nbr_degree', f'{instance_name}.json')) as f:
            d = json.load(f)
            node_nums_, mean_degs_, connected_mean_degs_ = d['res']
            return os.path.basename(filepath), node_nums_, mean_degs_, connected_mean_degs_

    G = networkt.load(filepath)
    results = []
    delta_0 = G.tnodes.timestamps[0]

    for delta_emp in emp_G.batch_iterator(True):
        if len(emp_G.tnodes[:emp_G.tnodes.timestamps[len(emp_G.tnodes[:delta_emp.stop]) - 1]]) > len(G.tnodes):
            break
        delta = slice(G.tnodes.timestamps[len(emp_G.tnodes[:delta_emp.start])],
                      G.tnodes.timestamps[len(emp_G.tnodes[:delta_emp.stop]) - 1])
        if len(G.tnodes[slice(delta_0, delta.stop)]) > 30:
            delta_ = slice(delta_0, delta.stop)
            delta_deg = cal_delta_deg(
                G, delta_, 0)

            results.append(delta_deg)
            delta_0 = delta.stop

        else:
            delta_0 = delta.start

    node_nums_, mean_degs_, connected_mean_degs_ = list(
        zip(*results))
    with open(os.path.join('logA', 'mean_nbr_degree', f'{instance_name}.json'), 'w') as f:
        d['res'] = node_nums_, mean_degs_, connected_mean_degs_
        json.dump(d, f)
    return os.path.basename(filepath), node_nums_, mean_degs_, connected_mean_degs_


def mean_nbr_degree_combined(log_dir='logA'):
    import multiprocessing
    pool = multiprocessing.Pool()
    agg = []
    dnames = []
    best_fits = json.load(open(os.path.join(log_dir, 'best_fits.json')))

    plt.rcParams["figure.figsize"] = (9, 9)
    from matplotlib.pyplot import rcParams
    rcParams['font.family'] = "sans-serif"
    rcParams['font.sans-serif'] = "Arial"
    rcParams['font.size'] = "15"
    for dataset_path in get_datasets("raw"):
        if len(parse_model_paras(os.path.basename(dataset_path))) != 0:
            continue
        emp_name = os.path.basename(dataset_path)
        dnames.append(emp_name)
        print("processing " + emp_name)
        emp_G = networkt.load(dataset_path)
        model_names = best_fits[emp_name]
        dataset_names = [emp_name] + model_names
        instances = [
            (instance, emp_G) for dataset_name in dataset_names for instance in get_all_instances(dataset_name)]
        results = list(pool.map(cal_mean_nbr_degree, instances))
        agg.append([])
        for name in model_names + [emp_name]:
            node_nums = []
            mean_degs = []
            connected_mean_degs = []
            for instance_name, node_nums_, mean_degs_, connected_mean_degs_ in results:
                if parse_model_type(instance_name) == parse_model_type(name):
                    node_nums.append(node_nums_)
                    mean_degs.append(mean_degs_)
                    connected_mean_degs.append(connected_mean_degs_)

            min_len = np.min([len(e) for e in node_nums])
            node_nums = node_nums[np.argmin([len(e) for e in node_nums])]
            mean_degs = pd.DataFrame(mean_degs).values[:, :min_len]
            connected_mean_degs = pd.DataFrame(
                connected_mean_degs).values[:, :min_len]
            ratios = connected_mean_degs / mean_degs
            ratio_ste = np.nanstd(pd.DataFrame(ratios), axis=0)
            ratio_ste = ratio_ste / np.sqrt(len(ratios))
            mean_ratio = np.mean(ratios, axis=0)

            plt.errorbar(node_nums, mean_ratio, yerr=ratio_ste, fmt='-', color=get_color(name),
                         label='{}'.format(format_name(name)))

            agg[-1].append([np.mean(ratios), np.mean(ratio_ste)])

        legend0 = plt.legend(loc='upper left')
        plt.gca().add_artist(legend0)

        plt.xscale('log')
        plt.xlim([0.9 * node_nums[0], 1.1 * node_nums[-1]])
        plt.xlabel('N')

        plt.ylabel(
            '$Deg_{connected}/Deg_{random}$')
        plt.clf()

    width = 0.15
    plt.figure(figsize=(14, 9))
    x = np.arange(len(dnames))
    for i, name in enumerate(model_names + [emp_name]):
        plt.bar(x + (2 * i - 5) * width / 2, [e[i][0] for e in agg],
                width,
                yerr=[e[i][1] for e in agg],
                label=format_name(name, True, True),
                color=get_color(name), ecolor='black')
    plt.legend()
    plt.xticks(x, dnames, rotation=70)
    plt.ylim(bottom=0.75)
    plt.ylabel(
        '$Deg_{connected}/Deg_{random}$')
    plt.tight_layout()
    os.makedirs(
        '{}/figures/mean_of_mean_nbr_deg'.format(log_dir), exist_ok=True)
    plt.savefig('{}/figures/mean_of_mean_nbr_deg/all.pdf'.format(log_dir))


def mean_nbr_degree(log_dir='logA'):
    import multiprocessing
    from glob import glob
    pool = multiprocessing.Pool()
    agg = []
    dnames = []
    best_fits = json.load(open(os.path.join(log_dir, 'best_fits.json')))

    plt.rcParams["figure.figsize"] = (9.5, 10.5)
    from matplotlib.pyplot import rcParams
    rcParams['font.family'] = "sans-serif"
    rcParams['font.sans-serif'] = "Arial"
    rcParams['font.size'] = "11"
    fig, axs = plt.subplots(4, 3)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False,
                    bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("\nNumber of Nodes")
    plt.ylabel("$Deg_{connected}/Deg_{random}$\n")
    indexes = [(x, y) for x in range(4) for y in range(3)]
    for dataset_path in get_datasets("raw"):
        print(dataset_path)
        if len(parse_model_paras(os.path.basename(dataset_path))) != 0:
            continue
        emp_name = os.path.basename(dataset_path)
        dnames.append(emp_name)
        print("processing " + emp_name)
        emp_G = networkt.load(dataset_path)
        model_names = best_fits[emp_name]
        dataset_names = [emp_name] + model_names
        instances = [
            (instance, emp_G) for dataset_name in dataset_names for instance in get_all_instances(dataset_name)]
        results = list(pool.map(cal_mean_nbr_degree, instances))
        agg.append([])
        for name in model_names + [emp_name]:
            style = "--" if len(parse_model_paras(name)) != 0 else "-"
            node_nums = []
            mean_degs = []
            connected_mean_degs = []
            for instance_name, node_nums_, mean_degs_, connected_mean_degs_ in results:
                if parse_model_type(instance_name) == parse_model_type(name):
                    node_nums.append(node_nums_)
                    mean_degs.append(mean_degs_)
                    connected_mean_degs.append(connected_mean_degs_)

            min_len = np.min([len(e) for e in node_nums])
            node_nums = node_nums[np.argmin([len(e) for e in node_nums])]
            mean_degs = pd.DataFrame(mean_degs).values[:, :min_len]
            connected_mean_degs = pd.DataFrame(
                connected_mean_degs).values[:, :min_len]
            ratios = connected_mean_degs / mean_degs
            ratio_ste = np.nanstd(pd.DataFrame(ratios), axis=0)
            ratio_ste = ratio_ste / np.sqrt(len(ratios))
            mean_ratio = np.mean(ratios, axis=0)

            d_index = len(dnames) - 1
            ax = axs[indexes[d_index][0], indexes[d_index][1]]
            ax.errorbar(node_nums, mean_ratio, yerr=ratio_ste, fmt=style, color=get_color(name),
                        label='{}'.format(format_name(name)))
            ax.set_title(format_name(emp_name), y=1.0, pad=-14)
            ax.set_xscale('log')

            ax.set_xlim([99, max(1010, len(emp_G.tnodes))])

            ax.xaxis.set_minor_formatter(NullFormatter())

            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax = axs[-1, -1]
        from matplotlib.lines import Line2D
        colors = ['grey', 'skyblue', 'seagreen', 'indianred']
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--')
                 for c in colors]
        lines[0] = Line2D([0], [0], color=colors[0],
                          linewidth=3, linestyle='-')
        labels = ['Empirical Data', 'Lambiotte et al., 2016',
                  'PCM', 'Forest Fire \n(Leskovec et al., 2007)']
        ax.legend(lines, labels, loc=2)

    axs[-1, -1].axis('off')
    plt.tight_layout()
    plt.show()


def mean_distance_edge(G):
    from scipy.stats.mstats import gmean
    agg_func = gmean
    nxG = nx.Graph()
    node_nums = []
    mean_dist_before = []
    mean_dist = []
    for i, delta in enumerate(G.batch_iterator(True)):
        timestamps = sorted(list(set(G.tedges[delta].timestamps)))
        mean_dist_before_interval = []
        mean_dist_interval = []
        node_num = len(nxG.nodes)
        cnt = 0
        for t0, t1 in zip([delta.start] + timestamps[1:], timestamps[1:] + [delta.stop]):
            mean_dist_before_ = []
            mean_dist_random_ = []
            for edge in G.tedges[t0:t1]:
                if edge in nxG.edges:
                    continue
                cnt += 1
                for v in edge:
                    nxG.add_node(v)
                dist_before = np.inf
                if edge[0] == edge[1]:
                    continue
                try:
                    dist_before = nx.shortest_path_length(
                        nxG, edge[0], edge[1])
                except:
                    pass
                nxG.add_edge(*edge)
                nodes = list(nxG.nodes)
                while True:
                    sample_pair = (nodes[np.random.randint(
                        0, len(nodes))], nodes[np.random.randint(0, len(nodes))])
                    if sample_pair[0] != sample_pair[1]:
                        break

                dist_random = np.inf
                try:
                    dist_random = nx.shortest_path_length(
                        nxG, sample_pair[0], sample_pair[1])
                except:
                    pass

                if dist_random != np.inf and dist_before != np.inf:
                    mean_dist_random_.append(dist_random)
                    mean_dist_before_.append(dist_before)

            if cnt > 1000:
                break

            if len(mean_dist_before_) > 0 and len(mean_dist_random_) > 0:
                mean_dist_before_ = agg_func(mean_dist_before_)
                mean_dist_before_interval.append(mean_dist_before_)
                mean_dist_random_ = agg_func(mean_dist_random_)
                mean_dist_interval.append(mean_dist_random_)
        nxG.add_edges_from(G.tedges[t1:delta.stop])

        if len(mean_dist_before_interval) > 5 and len(mean_dist_interval) > 5:
            mean_dist_before.append(agg_func(mean_dist_before_interval))
            node_nums.append(node_num)
            mean_dist.append(agg_func(mean_dist_interval))
        if node_num > 10**6:
            break
    return node_nums, mean_dist_before, mean_dist


def distance_temporal(name, emp_name, path_to_raw, G):
    from glob import glob
    node_nums = []
    distance_before = []
    distance_random = []
    if name != emp_name:
        for filepath in glob(os.path.join(path_to_raw, name + '_*'))[:]:
            print(filepath)
            modelG = networkt.load(filepath)
            t1 = modelG.tnodes.timestamps[min(
                len(modelG.tnodes), len(G.tnodes)) - 1]
            modelG = modelG[:t1 + 1]
            node_nums_, distance_before_, distance_random_ = mean_distance_edge(
                modelG)
            node_nums.append(node_nums_)
            distance_before.append(distance_before_)
            distance_random.append(distance_random_)
    else:
        node_nums_, distance_before_, distance_random_ = mean_distance_edge(
            G)
        node_nums.append(node_nums_)
        distance_before.append(distance_before_)
        distance_random.append(distance_random_)

    node_nums = node_nums[np.argmin([len(e) for e in node_nums])]
    distance_before = np.mean(pd.DataFrame(distance_before).dropna(
        axis='columns').values, axis=0)
    distance_random = np.mean(pd.DataFrame(
        distance_random).dropna(axis='columns').values, axis=0)

    return node_nums, distance_before, distance_random


def distance_edge_best(log_dir='logA'):

    from matplotlib.pyplot import rcParams
    plt.rcParams["figure.figsize"] = (9.5, 10.5)
    rcParams['font.family'] = "sans-serif"
    rcParams['font.sans-serif'] = "Arial"
    rcParams['font.size'] = "11"
    dnames = []
    path_to_raw = get_path_to('raw')
    best_fits = json.load(open(os.path.join(log_dir, 'best_fits.json')))
    os.makedirs(os.path.join(log_dir, 'distance'), exist_ok=True)
    fig, axs = plt.subplots(4, 3)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False,
                    bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("\nNumber of Nodes")
    plt.ylabel("Distance between To-Be-Connected Nodes\n")
    indexes = [(x, y) for x in range(4) for y in range(3)]
    for dataset_path in get_datasets("raw"):
        if len(parse_model_paras(os.path.basename(dataset_path))) != 0:
            continue
        emp_name = os.path.basename(dataset_path)
        print("processing " + emp_name)
        dnames.append(emp_name)
        G = networkt.load(dataset_path)
        model_names = [n for n in best_fits[emp_name]]

        distances = {}
        if os.path.exists(os.path.join(log_dir, 'distance', f'{emp_name}.json')):
            with open(os.path.join(log_dir, 'distance', f'{emp_name}.json')) as f:
                distances = json.load(f)

        for name in model_names + [emp_name]:
            style = "--" if len(parse_model_paras(name)) != 0 else "-"
            print(name)

            if name in distances:
                node_nums = distances[name]['node_nums']
                distance_before = distances[name]['distance_before']
                distance_random = distances[name]['distance_random']
            else:
                node_nums, distance_before, distance_random = distance_temporal(
                    name, emp_name, path_to_raw, G)
                distances[name] = {}
                distances[name]['node_nums'] = list(node_nums)
                distances[name]['distance_before'] = list(distance_before)
                distances[name]['distance_random'] = list(distance_random)

            d_index = len(dnames) - 1
            ax = axs[indexes[d_index][0], indexes[d_index][1]]
            ax.plot(node_nums, distance_before, style,
                    color=get_color(name), label=format_name(name) + ', before link formed')

            if name == emp_name:
                ax.plot(node_nums, distance_random,
                        '-.', color=get_color(name), label=format_name(name) + ', random pair')
                if max(node_nums) < 1000:
                    ax.set_xlim(right=1100)
            ax.set_title(format_name(emp_name), y=1.0, pad=-14)
            ax.set_xscale('log')
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())

            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        with open(os.path.join(log_dir, 'distance', f'{emp_name}.json'), 'w') as f:
            json.dump(distances, f)

    ax = axs[-1, -1]
    from matplotlib.lines import Line2D
    colors = ['skyblue', 'seagreen', 'indianred']
    lines = [Line2D([0], [0], color='grey', linewidth=2, linestyle='-.')]
    lines += [Line2D([0], [0], color='grey', linewidth=2, linestyle='-')]
    lines += [Line2D([0], [0], color=c, linewidth=2, linestyle='--')
              for c in colors]

    labels = ['Distance between\nRandomly Chosen Pair']
    labels += ['Empirical Data', 'Lambiotte et al., 2016',
               'PCM', 'Forest Fire \n(Leskovec et al., 2007)']
    ax.legend(lines, labels)

    axs[-1, -1].axis('off')
    plt.tight_layout()
    plt.show()


def distance_edge(log_dir='logA'):

    from matplotlib.pyplot import rcParams
    from glob import glob
    plt.rcParams["figure.figsize"] = (6.5, 6.5)
    rcParams['font.family'] = "sans-serif"
    rcParams['font.sans-serif'] = "Arial"
    rcParams['font.size'] = "15"
    dnames = []
    path_to_raw = get_path_to('raw')
    best_fits = json.load(open(os.path.join(log_dir, 'best_fits.json')))

    for dataset_path in get_datasets("raw"):
        if len(parse_model_paras(os.path.basename(dataset_path))) != 0:
            continue
        emp_name = os.path.basename(dataset_path)
        print("processing " + emp_name)
        dnames.append(emp_name)
        plt.xlabel("\nNumber of Nodes")
        plt.ylabel("Distance between To-Be-Connected Nodes\n")

        G = networkt.load(dataset_path)
        model_names = [n for n in best_fits[emp_name]]
        for name in model_names + [emp_name]:
            style = "--" if len(parse_model_paras(name)) != 0 else "-"
            print(name)
            node_nums = []
            distance_before = []
            distance_random = []
            if name != emp_name:
                for filepath in glob(os.path.join(path_to_raw, name + '_*'))[:]:
                    print(filepath)
                    modelG = networkt.load(filepath)
                    t1 = modelG.tnodes.timestamps[min(
                        len(modelG.tnodes), len(G.tnodes)) - 1]
                    modelG = modelG[:t1 + 1]
                    node_nums_, distance_before_, distance_random_ = mean_distance_edge(
                        modelG)
                    node_nums.append(node_nums_)
                    distance_before.append(distance_before_)
                    distance_random.append(distance_random_)
            else:
                node_nums_, distance_before_, distance_random_ = mean_distance_edge(
                    G)
                node_nums.append(node_nums_)
                distance_before.append(distance_before_)
                distance_random.append(distance_random_)

            node_nums = node_nums[np.argmin([len(e) for e in node_nums])]
            distance_before = np.mean(pd.DataFrame(distance_before).dropna(
                axis='columns').values, axis=0)
            distance_random = np.mean(pd.DataFrame(
                distance_random).dropna(axis='columns').values, axis=0)

            plt.plot(node_nums, distance_before, style,
                     color=get_color(name), label=format_name(name) + ', before link formed')
            if name == emp_name:
                plt.plot(node_nums, distance_random,
                         '-.', color=get_color(name), label=format_name(name) + ', random pair')

        ax = plt.gca()
        from matplotlib.lines import Line2D
        colors = ['skyblue', 'seagreen', 'indianred']
        lines = [Line2D([0], [0], color='grey', linewidth=2, linestyle='-.')]
        lines += [Line2D([0], [0], color='grey', linewidth=2, linestyle='-')]
        lines += [Line2D([0], [0], color=c, linewidth=2, linestyle='--')
                  for c in colors]

        labels = ['Distance between\nRandomly Chosen Pair']
        labels += [format_name(emp_name), 'Lambiotte et al., 2016',
                   'PCM', 'Forest Fire \n(Leskovec et al., 2007)']
        ax.legend(lines, labels)
        plt.tight_layout()
        plt.show()


def get_slope(seriesdata):
    seriesdata = seriesdata.to_numpy()
    _, clique_size = seriesdata.shape
    x = seriesdata[:, 0]
    slopes = []
    for i in range(2, clique_size + 1):
        # x = seriesdata[:, i - 2]
        x = seriesdata[:, 0]
        y = seriesdata[:, i - 1]
        logx = np.log(x)
        logy = np.log(y)
        p = longest_poly_fit(logx, logy, 1)
        # p = np.polyfit(logx, logy, 1)
        slopes.append(p[0])
    slopes = np.array(slopes)
    slopes = slopes[~np.isnan(slopes)]
    return slopes


def scaling_laws_combined(log_dir='logA'):
    from collections import defaultdict
    from matplotlib.pyplot import rcParams
    rcParams["figure.figsize"] = (9.5, 10.5)
    rcParams['font.family'] = "sans-serif"
    rcParams['font.sans-serif'] = "Arial"
    rcParams['font.size'] = "11"
    best_fits = json.load(open(os.path.join(log_dir, 'best_fits.json')))
    dnames = []
    fig, axs = plt.subplots(4, 3)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False,
                    bottom=False, left=False, right=False)
    plt.grid(False)
    indexes = [(x, y) for x in range(4) for y in range(3)]

    for dataset_path in get_datasets("statistics"):
        if len(parse_model_paras(os.path.basename(dataset_path))) != 0:
            continue
        max_num = 10**5
        slope_dict = defaultdict(list)
        emp_name = os.path.basename(dataset_path)
        dnames.append(emp_name)
        print("processing " + emp_name)
        model_names = best_fits[emp_name]
        dataset_names = [emp_name] + model_names
        instances = [instance for dataset_name in dataset_names
                     for instance in get_all_instances(dataset_name, 'statistics')]
        for instance_name in instances:
            series_filepath = os.path.join(instance_name, "num_of_cliques.csv")
            seriesdata = pd.read_csv(series_filepath)
            max_num = min(max_num, seriesdata.iloc[-1, 0])

        for instance_name in instances:
            series_filepath = os.path.join(instance_name, "num_of_cliques.csv")
            seriesdata = pd.read_csv(series_filepath)
            idx = np.searchsorted(seriesdata.iloc[:, 0], max_num)
            seriesdata = seriesdata.iloc[:idx, :]
            slopes = get_slope(seriesdata)
            instance_name = instance_name.split('_n_')[0]
            instance_name = os.path.basename(instance_name)
            slope_dict[instance_name].append(slopes)
        print(slope_dict.keys())
        slopes = slope_dict[emp_name]
        slopes = slopes[0]
        max_clique_size = len(slopes)

        for dataset_name in slope_dict:
            style = "--" if len(parse_model_paras(dataset_name)) != 0 else "-"
            slopes = slope_dict[dataset_name]
            ste = np.nanstd(pd.DataFrame(slopes).to_numpy(), axis=0)[:8]
            ste = ste / np.sqrt(len(slopes))
            slopes = np.mean(pd.DataFrame(slopes).to_numpy(), axis=0)[:8]
            slen = min(max_clique_size, len(slopes))
            d_index = len(dnames) - 1
            ax = axs[indexes[d_index][0], indexes[d_index][1]]
            ax.errorbar(range(2, slen + 2), slopes[:slen], fmt=style, yerr=ste[:slen], color=get_color(
                dataset_name), label=format_name(dataset_name))
            ax.set_title(format_name(emp_name), y=1.0, pad=-14)
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(MultipleLocator(1.))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%1.0f'))

            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_major_locator(MultipleLocator(1.))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%1.0f'))

        ax = axs[-1, -1]
        from matplotlib.lines import Line2D
        colors = ['grey', 'skyblue', 'seagreen', 'indianred']
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--')
                 for c in colors]
        lines[0] = Line2D([0], [0], color=colors[0],
                          linewidth=3, linestyle='-')
        labels = ['Empirical Data', 'Lambiotte et al., 2016',
                  'PCM', 'Forest Fire \n(Leskovec et al., 2007)']
        ax.legend(lines, labels, loc=2)

    axs[-1, -1].axis('off')
    plt.tight_layout()
    plt.show()


def scaling_laws(log_dir='logA'):
    from collections import defaultdict
    from matplotlib.pyplot import rcParams
    rcParams["figure.figsize"] = (6.5, 6.5)
    rcParams['font.family'] = "sans-serif"
    rcParams['font.sans-serif'] = "Arial"
    rcParams['font.size'] = "15"
    best_fits = json.load(open(os.path.join(log_dir, 'best_fits.json')))
    dnames = []

    for dataset_path in get_datasets("statistics"):
        if len(parse_model_paras(os.path.basename(dataset_path))) != 0:
            continue
        max_num = 10**5
        slope_dict = defaultdict(list)
        emp_name = os.path.basename(dataset_path)
        dnames.append(emp_name)
        print("processing " + emp_name)
        model_names = best_fits[emp_name]
        dataset_names = [emp_name] + model_names
        instances = [instance for dataset_name in dataset_names
                     for instance in get_all_instances(dataset_name, 'statistics')]
        for instance_name in instances:
            series_filepath = os.path.join(instance_name, "num_of_cliques.csv")
            seriesdata = pd.read_csv(series_filepath)
            max_num = min(max_num, seriesdata.iloc[-1, 0])

        for instance_name in instances:
            series_filepath = os.path.join(instance_name, "num_of_cliques.csv")
            seriesdata = pd.read_csv(series_filepath)
            idx = np.searchsorted(seriesdata.iloc[:, 0], max_num)
            seriesdata = seriesdata.iloc[:idx, :]
            slopes = get_slope(seriesdata)
            instance_name = instance_name.split('_n_')[0]
            instance_name = os.path.basename(instance_name)
            slope_dict[instance_name].append(slopes)

        slopes = slope_dict[emp_name]
        slopes = slopes[0]
        max_clique_size = len(slopes)

        for dataset_name in slope_dict:
            style = "--" if len(parse_model_paras(dataset_name)) != 0 else "-"
            slopes = slope_dict[dataset_name]
            ste = np.nanstd(pd.DataFrame(slopes).to_numpy(), axis=0)[:8]
            ste = ste / np.sqrt(len(slopes))
            slopes = np.mean(pd.DataFrame(slopes).to_numpy(), axis=0)[:8]
            slen = min(max_clique_size, len(slopes))

            plt.errorbar(range(2, slen + 2), slopes[:slen], fmt=style, yerr=ste[:slen], color=get_color(
                dataset_name), label=format_name(dataset_name, skip_paras=True))

        plt.legend(loc="upper left", markerscale=0.5)
        plt.xlabel("Clique Size")
        plt.ylabel("Scaling Exponent")
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()


def get_mean_clique_size(seriesdata):
    seriesdata.fillna(0, inplace=True)
    mlen = len(seriesdata)
    if len(np.where(np.diff(seriesdata.iloc[:, 2]) < 0)[0]) > 0:
        mlen = np.min(np.where(np.diff(seriesdata.iloc[:, 2]) < 0)) + 1

    seriesdata = seriesdata.iloc[:mlen, :]
    x = seriesdata.iloc[:, 0]
    y = seriesdata.iloc[:, 1:].apply(
        lambda row: sum(row.to_numpy() * np.array(range(2, len(row) + 2))) / sum(row), axis=1)

    return x, y


def mean_clique_size_combined(log_dir='logA'):
    from matplotlib.pyplot import rcParams
    rcParams["figure.figsize"] = (9.5, 10.5)
    rcParams['font.family'] = "sans-serif"
    rcParams['font.sans-serif'] = "Arial"
    rcParams['font.size'] = "11"
    from collections import defaultdict
    best_fits = json.load(open(os.path.join(log_dir, 'best_fits.json')))
    dnames = []
    fig, axs = plt.subplots(4, 3)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False,
                    bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("\nNumber of Nodes")
    plt.ylabel("Mean Clique Size")
    indexes = [(x, y) for x in range(4) for y in range(3)]
    for dataset_path in get_datasets("statistics"):
        if len(parse_model_paras(os.path.basename(dataset_path))) != 0:
            continue
        # max_num = 10**5
        size_dict = defaultdict(list)
        node_num_dict = defaultdict(list)
        emp_name = os.path.basename(dataset_path)
        dnames.append(emp_name)
        print("processing " + emp_name)
        model_names = best_fits[emp_name]
        dataset_names = [emp_name] + model_names
        instances = [instance for dataset_name in dataset_names
                     for instance in get_all_instances(dataset_name, 'statistics')]
        for instance_name in instances[:1]:
            series_filepath = os.path.join(instance_name, "num_of_cliques.csv")
            seriesdata = pd.read_csv(series_filepath)
            # max_num = min(max_num, seriesdata.iloc[-1, 0])
            max_num = seriesdata.iloc[-1, 0]

        for instance_name in instances:
            series_filepath = os.path.join(instance_name, "num_of_cliques.csv")
            seriesdata = pd.read_csv(series_filepath)
            idx = np.searchsorted(seriesdata.iloc[:, 0], max_num)
            seriesdata = seriesdata.iloc[:idx, :]
            x, y = get_mean_clique_size(seriesdata)
            model_name = instance_name.split('_n_')[0]
            model_name = os.path.basename(model_name)
            size_dict[model_name].append(y)
            node_num_dict[model_name].append(x)

        for dataset_name in size_dict:
            style = "--" if len(parse_model_paras(dataset_name)) != 0 else "-"
            mean_sizes = size_dict[dataset_name]
            ste = np.nanstd(pd.DataFrame(mean_sizes).to_numpy(), axis=0)
            ste = ste / np.sqrt(len(mean_sizes))
            mean_sizes = np.mean(pd.DataFrame(
                mean_sizes).to_numpy(), axis=0)
            node_nums = np.mean(pd.DataFrame(
                node_num_dict[dataset_name]).to_numpy(), axis=0)
            d_index = len(dnames) - 1
            ax = axs[indexes[d_index][0], indexes[d_index][1]]
            ax.errorbar(node_nums, mean_sizes[:], fmt=style, yerr=ste[:], color=get_color(
                dataset_name), label=format_name(dataset_name))
            ax.set_title(format_name(emp_name), y=1.0, pad=-14)
            ax.set_xscale('log')
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax = axs[-1, -1]
        from matplotlib.lines import Line2D
        colors = ['grey', 'skyblue', 'seagreen', 'indianred']
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--')
                 for c in colors]
        lines[0] = Line2D([0], [0], color=colors[0],
                          linewidth=3, linestyle='-')
        labels = ['Empirical Data', 'Lambiotte et al., 2016',
                  'PCM', 'Forest Fire \n(Leskovec et al., 2007)']
        ax.legend(lines, labels, loc=2)
    axs[-1, -1].axis('off')
    plt.tight_layout()
    plt.show()


def mean_clique_size(log_dir='logA'):
    from matplotlib.pyplot import rcParams
    rcParams["figure.figsize"] = (6.5, 6.5)
    rcParams['font.family'] = "sans-serif"
    rcParams['font.sans-serif'] = "Arial"
    rcParams['font.size'] = "15"

    from collections import defaultdict
    best_fits = json.load(open(os.path.join(log_dir, 'best_fits.json')))
    dnames = []
    for dataset_path in get_datasets("statistics"):
        if len(parse_model_paras(os.path.basename(dataset_path))) != 0:
            continue
        size_dict = defaultdict(list)
        node_num_dict = defaultdict(list)
        emp_name = os.path.basename(dataset_path)
        dnames.append(emp_name)
        print("processing " + emp_name)
        model_names = best_fits[emp_name]
        dataset_names = [emp_name] + model_names
        instances = [instance for dataset_name in dataset_names
                     for instance in get_all_instances(dataset_name, 'statistics')]
        for instance_name in instances[:1]:
            series_filepath = os.path.join(instance_name, "num_of_cliques.csv")
            seriesdata = pd.read_csv(series_filepath)
            max_num = seriesdata.iloc[-1, 0]

        for instance_name in instances:
            series_filepath = os.path.join(instance_name, "num_of_cliques.csv")
            seriesdata = pd.read_csv(series_filepath)
            idx = np.searchsorted(seriesdata.iloc[:, 0], max_num)
            seriesdata = seriesdata.iloc[:idx, :]
            x, y = get_mean_clique_size(seriesdata)
            model_name = instance_name.split('_n_')[0]
            model_name = os.path.basename(model_name)
            size_dict[model_name].append(y)
            node_num_dict[model_name].append(x)

        for dataset_name in size_dict:
            style = "--" if len(parse_model_paras(dataset_name)) != 0 else "-"
            mean_sizes = size_dict[dataset_name]
            ste = np.nanstd(pd.DataFrame(mean_sizes).to_numpy(), axis=0)
            ste = ste / np.sqrt(len(mean_sizes))
            mean_sizes = np.mean(pd.DataFrame(
                mean_sizes).to_numpy(), axis=0)
            node_nums = np.mean(pd.DataFrame(
                node_num_dict[dataset_name]).to_numpy(), axis=0)

            plt.errorbar(node_nums, mean_sizes[:], fmt=style, yerr=ste[:], color=get_color(
                dataset_name), label=format_name(dataset_name, skip_paras=True))

        plt.legend()
        plt.xlabel("Number of Nodes")
        plt.ylabel("Mean Clique Size")
        plt.xscale('log')
        plt.show()
        plt.clf()


def num_cliques_combined():
    from matplotlib.pyplot import rcParams
    rcParams["figure.figsize"] = (9.5, 10.5)
    rcParams['font.family'] = "sans-serif"
    rcParams['font.sans-serif'] = "Arial"
    rcParams['font.size'] = "11"

    dnames = []
    fig, axs = plt.subplots(4, 3)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False,
                    bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("\nNumber of Nodes")
    plt.ylabel("Number of k-Cliques")
    indexes = [(x, y) for x in range(4) for y in range(3)]
    for dataset_path in get_datasets("statistics"):
        if len(parse_model_paras(os.path.basename(dataset_path))) != 0:
            continue
        emp_name = os.path.basename(dataset_path)
        dnames.append(emp_name)
        print("processing " + emp_name)
        series_filepath = os.path.join(dataset_path, "num_of_cliques.csv")
        seriesdata = pd.read_csv(series_filepath).to_numpy()
        d_index = len(dnames) - 1
        ax = axs[indexes[d_index][0], indexes[d_index][1]]
        _, num_of_series = seriesdata.shape
        x = seriesdata[:, 0]
        cmap = plt.get_cmap('cool')
        for i in range(1, num_of_series):
            c_t = seriesdata[:, i]
            ax.plot(x[c_t != 0], c_t[c_t != 0], c=cmap(
                np.log10(i + 0) / np.log10(55)), label=format_name(emp_name))
        ax.set_title(format_name(emp_name), y=1.0, pad=-14)
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.xaxis.set_minor_locator(LogLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())

        ax.yaxis.set_minor_locator(LogLocator(
            base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
        ax.yaxis.set_minor_formatter(NullFormatter())

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=matplotlib.colors.Normalize(vmin=2, vmax=55))
    cax = fig.add_axes([0.9, 0.1, 0.04, 0.17])
    fig.colorbar(sm, ticks=[2, 55], cax=cax)
    axs[-1, -1].text(0.05, 0.5, 'Size of Clique - k')
    axs[-1, -1].axis('off')
    plt.tight_layout()
    plt.show()


def num_cliques():
    from matplotlib.pyplot import rcParams
    rcParams["figure.figsize"] = (6.5, 6.5)
    rcParams['font.family'] = "sans-serif"
    rcParams['font.sans-serif'] = "Arial"
    rcParams['font.size'] = "15"

    dnames = []

    for dataset_path in get_datasets("statistics"):
        if len(parse_model_paras(os.path.basename(dataset_path))) != 0:
            continue
        emp_name = os.path.basename(dataset_path)
        dnames.append(emp_name)
        print("processing " + emp_name)
        series_filepath = os.path.join(dataset_path, "num_of_cliques.csv")
        seriesdata = pd.read_csv(series_filepath).to_numpy()

        _, num_of_series = seriesdata.shape
        x = seriesdata[:, 0]
        cmap = plt.get_cmap('cool')
        for i in range(1, num_of_series):
            c_t = seriesdata[:, i]
            plt.plot(x[c_t != 0], c_t[c_t != 0], c=cmap(
                np.log10(i + 0) / np.log10(num_of_series)), label=format_name(emp_name))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("\nNumber of Nodes")
        plt.ylabel("Number of k-Cliques")
        plt.tight_layout()
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=matplotlib.colors.Normalize(vmin=2, vmax=num_of_series))
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins1 = inset_axes(plt.gca(),
                            bbox_to_anchor=(0.01, -0.01, 1, 1),
                            bbox_transform=plt.gca().transAxes,
                            width="5%",
                            height="20%",
                            loc='upper left')
        cb = plt.colorbar(sm, ticks=[2, num_of_series], cax=axins1)
        cb.set_label('k', rotation=0, va="center")
        plt.show()
        plt.clf()


def parse_parameters(dataset_name):
    import re
    all_paras = dict(re.findall("(?:_([a-z])_)(\d+\.*\d+)", dataset_name))
    return all_paras


def get_path_to(dataset_type):
    with open("config.json", "r+") as json_file:
        configs = json.load(json_file)
        path_to_ = configs[dataset_type]["path"]
    return path_to_


def get_datasets(dataset_type):
    with open("config.json", "r+") as json_file:
        configs = json.load(json_file)

    path_to_datasets = configs[dataset_type]["path"]

    dataset_paths = []
    for name in os.listdir(path_to_datasets):
        full_path = os.path.join(path_to_datasets, name)
        if os.path.isdir(full_path):
            dataset_paths.append(full_path)

    paths = []
    allowed_names = configs["allowlist"]
    if len(allowed_names) != 0:
        for allowed_name in allowed_names:
            filtered_paths = list(filter(
                lambda x: allowed_name in x, dataset_paths))
            paths.extend(filtered_paths)
    else:
        paths = dataset_paths

    banned_names = configs["banlist"]

    for ban_name in banned_names:
        paths = list(filter(
            lambda x: ban_name not in x, paths))
    return sorted(set(paths))


def find_linear_section(x, y, deg):
    def pf(z, *params):
        # n0, n1 = z.astype(int)
        n1 = len(x)
        n0 = z[0].astype(int)
        if n1 - n0 < len(x) * 0.1 or n0 < 0 or n1 > len(x):
            return np.Inf
        p = np.polyfit(x[n0:n1], y[n0:n1], deg)
        pred_y = np.polyval(p, x[n0:n1])
        err = np.mean((pred_y - y[n0:n1])**2) * 800**((len(x) / (n1 - n0)))

        return err
    from scipy import optimize

    ns = optimize.brute(
        pf, [slice(0, len(x))] * 1, args=(x, y))
    ns = ns.astype(int)
    return ns[0]


def longest_poly_fit(x, y, deg):
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    if len(x) <= 15:
        return [np.nan]
    # n0 = find_linear_section(x, y, deg)
    n0 = 0
    ps = np.polyfit(x[n0:], y[n0:], deg)

    return ps


def log_hist_2d(xs, ys):

    import matplotlib.colors as colors
    xbins = np.e**np.linspace(0, np.log(np.max(xs)), 50)
    ybins = np.e**np.linspace(0, np.log(np.max(ys)), 50)
    counts, _, _ = np.histogram2d(xs, ys, bins=(xbins, ybins))
    # counts = np.log(counts)
    # counts[counts == -np.inf] = 0
    plt.pcolormesh(xbins, ybins, counts.T, norm=colors.LogNorm())
    plt.colorbar()
    plt.xscale('log')
    plt.yscale('log')
