import sys
sys.path.append("../")
import networkt
from utils import snap_forest_fire_graph
import multiprocessing
import inspect
import numpy as np
import random
from functools import partial
import os
from clique_counter import get_clique_num, clique_num_to_df
from collections import defaultdict
import json
from clique_counter import pivoter
import time
import bisect
from utils import MODELS, DISPLAY_NAME


def load_datasets(max_n):
    import os
    import pandas as pd
    from utils import get_datasets, parse_parameters
    from clique_counter import get_clique_num
    emp_clique_nums = {}
    for empirical_dataset_path in get_datasets("statistics"):
        if len(parse_parameters(os.path.basename(empirical_dataset_path))) > 0:
            continue
        print(os.path.basename(empirical_dataset_path))
        series_filepath = os.path.join(
            empirical_dataset_path, "num_of_cliques.csv")
        emp_clique_num = pd.read_csv(series_filepath)
        key = os.path.basename(empirical_dataset_path)
        emp_clique_nums[key] = get_clique_num(emp_clique_num)
        if max_n > 0:
            emp_clique_nums[key] = cap_node_num(
                emp_clique_nums[key], max_n)

    return emp_clique_nums


class ABCModelChoice:
    def __init__(self, datasets, models=MODELS,
                 knn_percentile=0.1, k=None, N=2500 * 2,
                 h_timeout=6,
                 log_dir='./log'):
        self.datasets = datasets
        self.models = models
        self.k = int(knn_percentile * N * len(models))
        if k is not None:
            self.k = k
        self.N = N
        self.h_timeout = h_timeout

        self.log_dir = log_dir

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        for m in models.values():
            if not os.path.exists(os.path.join(self.log_dir, m.__name__)):
                os.mkdir(os.path.join(self.log_dir, m.__name__))
            if not os.path.exists(os.path.join(self.log_dir, m.__name__ + '_num')):
                os.mkdir(os.path.join(self.log_dir, m.__name__ + '_num'))

            if not os.path.exists(os.path.join(self.log_dir, 'processes')):
                os.mkdir(os.path.join(self.log_dir, 'processes'))

    def select(self):
        args = {}
        for name in self.models:
            try_func = self.models[name]
            args[name] = set(inspect.getfullargspec(
                try_func).args).difference(['target_m', 'n'])

        particles = []
        for _ in range(self.N * len(self.models)):
            model = random.choice(list(self.models))
            paras = {}
            for para_name in sorted(args[model]):
                paras[para_name] = np.round(np.random.rand(), 2)
            particles.append((model, paras))

        # logger = multiprocessing.log_to_stderr()
        # logger.setLevel(multiprocessing.SUBDEBUG)

        manager = multiprocessing.Manager()
        min_dists = {}
        locks = {}
        for key in self.datasets:
            min_dists[key] = manager.list()
            locks[key] = manager.Lock()

        global_vals = {}
        global_vals['start'] = time.time()
        global_vals['max'] = self.h_timeout * 60 * 60
        global_vals['done'] = manager.list()
        global_vals['lock'] = manager.Lock()

        print(len(particles))

        particles = [(*e, particles[:i].count(e))
                     for i, e in enumerate(particles)]
        work = [[self.models[m], paras, occurrence,
                 self.datasets,
                 self.k,
                 locks,
                 min_dists,
                 global_vals,
                 self.log_dir] for m, paras, occurrence in particles]

        particles = ["{}, {}, {}".format(
            m, ABCModelChoice.to_para_str(p), o) for m, p, o in particles]
        with open(os.path.join(self.log_dir, 'particles.txt'), 'w') as f:
            for p in particles:
                print(p, file=f)

        print("k={}".format(self.k))
        pool = multiprocessing.Pool(multiprocessing.cpu_count() * 2)
        t0 = time.time()
        dists = list(pool.imap(self.simulate, work))
        print("time taken: {}".format(time.time() - t0))
        # dists = list(map(self.simulate, work))
        json.dump(dists, open(os.path.join(self.log_dir, 'abc_N_{}_k_{}.json'.format(
            self.N, self.k)), 'w'), indent=4)

    # @staticmethod
    # def cal_dist(D_sim, D_emp):
    #     # Euclidean Distance
    #     D_sim = np.array(D_sim)
    #     D_emp = np.array(D_emp)
    #     r, c0 = D_emp.shape
    #     _, c1 = D_sim.shape
    #     c = max(c0, c1)
    #     padded0 = np.zeros((r, c)) #+ 1
    #     padded0[:, :c0] = D_emp
    #     padded1 = np.zeros((r, c)) #+ 1
    #     padded1[:, :c1] = D_sim
    #     return np.sum((np.log(padded1 + 1) - np.log(padded0 + 1))**2)

    @staticmethod
    def cal_dist(D_sim, D_emp):
        # KL Divergence
        np.seterr(all='raise')
        D_sim = np.array(D_sim)
        D_emp = np.array(D_emp)
        r, c0 = D_emp.shape
        _, c1 = D_sim.shape
        c = max(c0, c1)
        padded0 = np.zeros((r, c))
        padded0[:, :c0] = D_emp
        padded1 = np.zeros((r, c))
        padded1[:, :c1] = D_sim
        padded1 = padded1 + 1
        padded0 = padded0 + 1
        # padded0 = np.log(padded0) + 1
        # padded1 = np.log(padded1) + 1
        padded0 = padded0 / np.sum(padded0, axis=1)[:, None]
        padded1 = padded1 / np.sum(padded1, axis=1)[:, None]
        # return np.sum(padded1 * np.log(padded1 / padded0))
        return np.sum(padded0 * np.log(padded0 / padded1))

    @staticmethod
    def get_max_edge_num(datasets):
        return int(max([v[1][-1, 0] for k, v in datasets.items()]))

    @staticmethod
    def to_para_str(paras):
        return '_'.join(['_'.join(str(val) for val in it)
                         for it in paras.items()])

    @staticmethod
    def simulate(args):
        try_func, paras, occurrence, datasets, k, locks, min_dists, global_vals, log_dir = args
        model_name = try_func.__name__
        try_func = partial(try_func, **paras)
        para_str = '_' + ABCModelChoice.to_para_str(paras)
        log_file = os.path.join(log_dir, model_name,
                                '{}_{}.txt'.format(para_str, occurrence))
        num_file = os.path.join(
            log_dir, model_name + '_num', '{}_{}.txt'.format(para_str, occurrence))

        f = open(log_file, 'w')
        fnum = open(num_file, 'w')
        gen_func = partial(try_func, target_m=0)

        all_node_nums = []
        node_num_to_dataset = defaultdict(list)
        for key in datasets:
            node_nums, _ = datasets[key]
            for node_num in node_nums:
                node_num_to_dataset[node_num].append(key)
            all_node_nums.extend(node_nums)
        all_node_nums = sorted(list(set(all_node_nums)))

        max_edge_num = ABCModelChoice.get_max_edge_num(datasets)

        ns = []
        for _ in range(1):
            retry = True
            while retry:
                try:
                    edges = try_func(n=min(max(all_node_nums), 100000),
                                     target_m=max_edge_num)
                    retry = False
                except Exception as e:
                    print('try func error: {}'.format(e), file=f)
            nodes = {node for edge in edges for node in edge}
            ns.append(len(nodes))

        mid_node_num = 0.5 * (max(all_node_nums) - min(ns)) + \
            min(ns) if min(ns) < 1 / 2 * max(all_node_nums) else min(ns)

        search_list = [min(ns),
                       mid_node_num,
                       max(all_node_nums)]
        search_list = [int(e) for e in search_list if e <= max(all_node_nums)]
        search_list = sorted(list(set(search_list)))
        target_ms = [2 * max_edge_num, 5 * 10 ** 6, 10 ** 7]
        # target_ms = [2 * max_edge_num, 4 * max_edge_num, 6 * max_edge_num]
        target_ms = target_ms[-len(search_list):]
        dists = {}

        print('n\'s to search: {}'.format(search_list), file=f)
        print('expected m\'s: {}'.format(target_ms), file=f)
        f.flush()
        # os.fsync(f)
        start = time.perf_counter()
        stop = False
        runtime_in_hours = 0
        early_exit = set()

        for n, tm in zip(search_list, target_ms):
            clique_nums = defaultdict(list)
            print("{},{}".format(paras, n), file=f)

            if n < all_node_nums[0]:
                continue

            retry = True
            while retry:
                try:
                    edges = gen_func(n=n, target_m=tm)
                    retry = False
                except Exception as e:
                    print('gen func error: {}'.format(e), file=f)
            edges = [list(edge) + [max(edge)] for edge in edges]
            tG = networkt.TGraph(edges)

            n = len(tG.tnodes)
            print('actual n: {}'.format(len(tG.tnodes)), file=f)
            print('actual n: {}'.format(len(tG.tnodes)), file=fnum)

            sub_node_nums_ = [int(num)
                              for num in all_node_nums if num <= len(tG.tnodes)]
            sub_node_nums_gen = [num - 1 for num in sub_node_nums_]

            for node_num, tspan in zip(sub_node_nums_, tG.batch_iterator(node_indexes=sub_node_nums_gen)):
                if len(set(node_num_to_dataset[node_num]).difference(early_exit)) == 0:
                    continue

                if time.perf_counter() - start > global_vals['max']:
                    stop = True
                    pass
                subG = tG[tspan].labels_to_indices()
                kwargs_ = dict(paras)
                kwargs_['sub'] = len(tG[tspan].tnodes)
                kwargs_['n'] = len(tG.tnodes)
                prefix = '_'.join(['{}_{}'.format(k, v)
                                   for k, v in kwargs_.items()])

                num_of_cliques = None
                import traceback
                try:
                    num_of_cliques = pivoter(subG.tedges, prefix, None)
                except Exception as e:
                    print('Exception: {}'.format(e), file=f)
                    stop = True
                if num_of_cliques is None:
                    stop = True
                    break

                print(num_of_cliques, file=fnum)

                remaining_keys = set(
                    node_num_to_dataset[node_num]).difference(early_exit)

                for key in remaining_keys:
                    clique_nums[key].append(num_of_cliques)
                    clique_df = clique_num_to_df(clique_nums[key])
                    datasets_model = get_clique_num(clique_df)
                    dist = ABCModelChoice.cal_dist(
                        datasets_model[1], datasets[key][1][:len(datasets_model[1])])

                    locks[key].acquire()
                    sorted_dists = list(min_dists[key])
                    locks[key].release()

                    if not all(sorted_dists[i] <= sorted_dists[i + 1] for i in range(len(sorted_dists) - 1)):
                        print('{}: not sorted'.format(log_file), file=f)
                        raise Exception('{}: not sorted'.format(log_file))

                    rank = np.searchsorted(sorted_dists, dist) + 1

                    print('for {}, {}, sub={}: at #nodes={}, dist({}) ranked {}-th'.format(
                        key, paras, len(tG.tnodes), int(node_num),
                        dist, rank), file=f)

                    if node_num >= datasets[key][0][-1]:
                        locks[key].acquire()
                        min_dists[key].append(dist)
                        min_dists[key].sort()
                        if len(min_dists[key]) > k:
                            min_dists[key].pop(k)
                        locks[key].release()

                        print('    complete', file=f)
                        dists[key] = ('complete={}'.format(
                            int(node_num)), dist)
                    else:
                        dists[key] = ('sub={}'.format(int(node_num)), dist)

                    if k <= len(sorted_dists) and dist > sorted_dists[k - 1]:
                        early_exit.add(key)
                        print('    knn', file=f)

                update_freq = 1 / (60 * 60)
                time_elapsed = int(
                    (time.time() - global_vals['start']) * update_freq)
                perf_time = int(
                    (time.perf_counter() - start) * update_freq)
                if time_elapsed > runtime_in_hours:
                    print('time {}h, {}h time'.format(
                        time_elapsed, perf_time), file=f)
                runtime_in_hours = time_elapsed * update_freq

                if stop:
                    break

                f.flush()
                # os.fsync(f)

                fnum.flush()
                # os.fsync(fnum)

            if stop:
                print('early stop', file=f)
                break

            if len(set(datasets.keys()).difference(early_exit)) == 0:
                print('stop exploring {}...'.format(paras), file=f)
                break
            f.flush()
            # os.fsync(f)

        print('end', file=f)
        global_vals['done'].append(1)
        print("done = {}".format(len(global_vals['done'])))

        try:
            os.rename(log_file, os.path.join(os.path.dirname(
                log_file), 'done' + os.path.basename(log_file)))
        except:
            print('failed to rename {}'.format(log_file))

        return model_name, paras, dists

    @staticmethod
    def apply_custom_dist_one(args):
        fn, datasets = args
        with open(fn) as f:
            lines = f.readlines()
            # clique_nums = [json.loads(l) for l in lines[1:]]
            clique_nums = []
            start_pos = []
            for i in range(len(lines)):
                if 'actual' in lines[i]:
                    start_pos.append(i + 1)

            dists = {}
            for start, end in zip(start_pos, start_pos[1:] + [len(lines) + 1]):
                for i, l in enumerate(lines[start:end - 1]):
                    clique_nums.append(json.loads(l))
                clique_df = clique_num_to_df(clique_nums)
                datasets_model = get_clique_num(clique_df)

                for key in datasets:
                    # for node_num, clique_nums in zip(*datasets[key]):
                    node_nums = []
                    clique_nums = []
                    for node_num, clique_num in zip(*datasets_model):
                        if node_num in datasets[key][0]:
                            node_nums.append(node_num)
                            clique_nums.append(clique_num)

                    datasets_model_ = [node_nums, clique_nums]

                    min_len = min(len(datasets[key][1]),
                                  len(datasets_model_[1]))
                    if min_len == 0:
                        continue
                    # dist = ABCModelChoice.cal_dist(datasets[key][1][:min_len],
                    #                                datasets_model_[1][:min_len])
                    dist = ABCModelChoice.cal_dist(datasets_model_[1][:min_len],
                                                   datasets[key][1][:min_len])
                    full_emp_len = len(datasets[key][1])

                    state = 'sub' if full_emp_len > min_len else 'complete'
                    num_node = int(datasets[key][0][min_len - 1])
                    dists[key] = [f"{state}={num_node}", dist]

            model_name = os.path.basename(
                os.path.dirname(fn)).replace('_num', '')
            para_str = os.path.basename(fn)
            from utils import parse_parameters
            paras = dict(sorted(parse_parameters(para_str).items()))

        return [model_name, paras, dists]

    @staticmethod
    def apply_custom_dist(log_dir: dict({'default': 'logA'}), max_n: dict({'default': 100000})):
        from glob import glob
        import json
        logs = glob(os.path.join(log_dir, '*_num', '_*'))
        datasets = load_datasets(max_n)
        from multiprocessing import Pool
        pool = Pool()
        # results = []
        args = [(log, datasets) for log in logs[:]]
        results = pool.map(ABCModelChoice.apply_custom_dist_one, args)
        # results = list(map(ABCModelChoice.apply_custom_dist_one, args))
        with open(os.path.join(log_dir, 'abc_custom.json'), 'w') as outf:
            json.dump(results, outf, indent=4)

    # generate a reconstructed summary for later analysis
    @staticmethod
    def gather_partial_results(log_dir='./log1'):
        import re
        from glob import glob
        logs = glob(os.path.join(log_dir, '*', 'done_*'))
        results = []
        for fn in logs:
            if '_num' in fn:
                continue
            dists = {}
            with open(fn) as f:
                for l in reversed(f.readlines()):
                    try:
                        dname = re.search('for (.+), {', l).group(1)
                        if dname in dists:
                            continue
                        node_num = int(re.search('#nodes=(\d+)', l).group(1))
                        dist = float(re.search('dist\((.+)\)', l).group(1))
                        dists[dname] = [node_num, dist]
                    except:
                        pass
            model_name = os.path.basename(os.path.dirname(fn))
            para_str = os.path.basename(fn)
            from utils import parse_parameters
            paras = dict(sorted(parse_parameters(para_str).items()))

            results.append([model_name, paras, dists])
            # break

        node_nums = [(dname, node_num) for model_name, paras,
                     dists in results for dname, (node_num, _) in dists.items()]
        max_nums = defaultdict(lambda: 0)
        for dname, node_num in node_nums:
            if node_num > max_nums[dname]:
                max_nums[dname] = node_num

        max_nums = dict(max_nums)
        for model_name, paras, dists in results:
            for dname, (node_num, dist) in dists.items():
                dists[dname] = [
                    ('sub=' if node_num < max_nums[dname] else 'complete=') + str(node_num), dist]
        with open(os.path.join(log_dir, 'abc_reconstructed.json'), 'w') as outf:
            json.dump(results, outf, indent=4)

    @staticmethod
    def __get_top_fits(result_json, k=12):
        best_paras_for_each_model = defaultdict(
            lambda: defaultdict(lambda: (2**30,)))
        top_fits = defaultdict(list)
        for model_name, paras, dists in result_json:
            for key, (state, dist) in dists.items():
                if 'sub' in state:
                    # sub = int(state.split('=')[-1])
                    # if dist < top_fits[key][0][0]:
                    #     print((model_name, paras))
                    continue
                l = best_paras_for_each_model[key][model_name]
                if len(l) == 0 or dist < l[0]:
                    best_paras_for_each_model[key][model_name] = (dist, paras)
                # if "threshold_padme_graph" in model_name and float(paras['r']) == 1.:
                #     l = best_paras_for_each_model[key]["sub||" + model_name]
                #     if len(l) == 0 or dist < l[0]:
                #         best_paras_for_each_model[key]["sub||" +
                #                                        model_name] = (dist, paras)
                if len(top_fits[key]) < k or dist < top_fits[key][k - 1][0]:
                    top_fits[key].append((dist, model_name))
                    top_fits[key].sort(key=lambda val: val[0])
                    if len(top_fits[key]) > k:
                        top_fits[key] = top_fits[key][:k]

        top_fits = dict(sorted(top_fits.items()))
        best_paras_for_each_model = {key: dict(sorted(dists.items()))
                                     for key, dists in sorted(best_paras_for_each_model.items())}

        return top_fits, best_paras_for_each_model

    @staticmethod
    def __make_top_dist_plot(top_fits):
        from matplotlib import pyplot as plt
        from matplotlib.pyplot import rcParams
        rcParams["figure.figsize"] = (12, 8)
        rcParams['font.family'] = "sans-serif"
        rcParams['font.sans-serif'] = "Arial"
        rcParams['font.size'] = "15"
        from utils import get_color, format_name
        labels = set()

        for i, dname in enumerate(top_fits):
            norm = top_fits[dname][-1][0]
            for j, (dist, mname) in enumerate(top_fits[dname]):
                plt.bar(i + 0.05 * j, dist / norm, 0.05,
                        color=get_color(mname), label=format_name(mname) if format_name(mname) not in labels else '')
                labels.add(format_name(mname))
        plt.xticks(range(len(top_fits)), top_fits.keys(), rotation=70)
        # plt.ylabel('Top Euclidean Distances')
        plt.ylabel('Kullback–Leibler Divergence')
        # plt.title('Top fits')
        plt.tight_layout()
        plt.legend()
        plt.show()

    @staticmethod
    def __make_best_dist_plot(best_paras_for_each_model):
        from matplotlib import pyplot as plt
        from matplotlib.pyplot import rcParams
        rcParams["figure.figsize"] = (12, 7)
        rcParams['font.family'] = "sans-serif"
        rcParams['font.sans-serif'] = "Arial"
        rcParams['font.size'] = "14"
        # from brokenaxes import brokenaxes
        # bax = brokenaxes(ylims=((0, 55), (200, 225)))
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                       gridspec_kw={'height_ratios': [1, 3]})
        # fig.subplots_adjust(hspace=0.)
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False,
                        bottom=False, left=False, right=False)
        plt.grid(False)
        # plt.xlabel("common X")
        plt.ylabel("Mean Kullback–Leibler Divergence")
        from utils import get_color, format_name
        for i, dname in enumerate(best_paras_for_each_model):
            norm = max(
                [dist for dist, _ in best_paras_for_each_model[dname].values()])
            for j, (mname, (dist, _)) in enumerate(best_paras_for_each_model[dname].items()):
                ax1.bar(i + 0.15 * j, dist / 1, 0.15,
                        color=get_color(mname), label=(format_name(mname)) if i == 0 else '')
                ax2.bar(i + 0.15 * j, dist / 1, 0.15,
                        color=get_color(mname), label=(format_name(mname)) if i == 0 else '')
                # ax3.bar(i + 0.15 * j, dist / 1, 0.15,
                #         color=get_color(mname), label=(format_name(mname)) if i == 0 else '')
        # plt.xticks(range(len(best_paras_for_each_model)),
        #            best_paras_for_each_model.keys(), rotation=70)
        # ax2.set_ylim(0, 53)
        # ax1.set_ylim(60, 230)
        ax2.set_ylim(0, 0.57)
        ax1.set_ylim(1, 2.5)
        ax1.spines.bottom.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()
        ax2.set_xticks([i for i in range(len(best_paras_for_each_model))])
        ax2.set_xticklabels(best_paras_for_each_model.keys(), rotation=70)
        # ax2.set_ylabel('Kullback–Leibler Divergence')
        # plt.ylabel('Kullback–Leibler Divergence')
        # ax1.legend(loc=2,bbox_to_anchor=(0, 0.5))
        # ax1.legend(loc=2)
        ax2.legend(loc=2, bbox_to_anchor=(0., 1.37))
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        # kwargs = dict(marker=[(1, d), (-1, -d)], markersize=12,
        #               linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
        # ax3.plot([0, 1], [1, 1], transform=ax3.transAxes, **kwargs)
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.05)

        # plt.subplots_adjust(bottom=0.2)
        plt.show()

    @staticmethod
    def __make_posterior_table(top_fits):
        from collections import Counter
        posterior = {k: dict(sorted(Counter([it[1] for it in v]).items(), key=lambda it: it[1], reverse=True))
                     for k, v in top_fits.items()}
        # print(json.dumps(posterior, indent=4))
        mat = []
        indexes = []
        for dname in posterior:
            indexes.append(dname)
            mat.append([])
            for mname in DISPLAY_NAME.keys():
                if mname in posterior[dname]:
                    mat[-1].append(posterior[dname][mname])
                else:
                    mat[-1].append(0)
        import pandas as pd
        df = pd.DataFrame(mat, columns=DISPLAY_NAME.keys())
        df.index = indexes
        print(df)

    @staticmethod
    def parse_results(log_dir='./logA', prefix='N', k=12):
        emp_clique_nums = load_datasets(100000)
        from glob import glob
        from collections import Counter
        print(f'abc_{prefix}*.json')
        result_json = json.load(
            open(glob(os.path.join(log_dir, f'abc_{prefix}*.json'))[0]))
        top_fits, best_paras_for_each_model = ABCModelChoice.__get_top_fits(
            result_json, k)

        print(json.dumps(best_paras_for_each_model, indent=4))
        print(json.dumps(top_fits, indent=4))

        ABCModelChoice.__make_top_dist_plot(top_fits)
        
        for key in best_paras_for_each_model:
            for mname in best_paras_for_each_model[key]:
                dist, paras = best_paras_for_each_model[key][mname]
                best_paras_for_each_model[key][mname] = [
                    dist / len(emp_clique_nums[key][0]), paras]
                
        ABCModelChoice.__make_best_dist_plot(best_paras_for_each_model)
        ABCModelChoice.__make_posterior_table(top_fits)

        # mapping = {k: v.__name__ for k, v in MODELS.items()}
        best_fits = {key: [name + '_' + '_'.join(['_'.join(str(val) for val in it) for it in models[name][1].items(
        )]) for name in models] for key, models in best_paras_for_each_model.items()}
        with open(os.path.join(log_dir, 'best_fits.json'), 'w') as f:
            json.dump(best_fits, f, indent=4)


def cap_node_num(emp_mean_clique_size, max_n):
    idx = np.searchsorted(emp_mean_clique_size[0], max_n)
    return [emp_mean_clique_size[0][:idx], emp_mean_clique_size[1][:idx]]


def runABC(max_n: dict({'default': 100000}), log_dir='./logA'):
    emp_clique_nums = load_datasets(max_n)
    abc = ABCModelChoice(
        emp_clique_nums, k=21, N=2500, log_dir=log_dir, h_timeout=1)

    abc.select()

