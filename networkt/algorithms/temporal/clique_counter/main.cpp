#include <cmath>
// #include <valarray>
#include <algorithm>
#include <bits/stdc++.h>
#include <vector>
#include <deque>
#include <queue>
#include <array>
#include <map>
#include <set>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <sys/time.h>
#include <chrono>
#include <thread>
// #include <boost/heap/fibonacci_heap.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;
// using namespace boost;

struct ArrayHasher
{
    std::size_t operator()(const array<uint64_t, 2> &a) const
    {
        std::size_t h = 0;

        for (auto e : a)
        {
            h ^= std::hash<int>{}(e) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};
bool comp(int a, int b)
{
    return a > b;
}

template <class Container>
void rsort_unique(Container &nbrs)
{
    sort(nbrs.begin(), nbrs.end());
    reverse(nbrs.begin(), nbrs.end());
    nbrs.erase(unique(nbrs.begin(), nbrs.end()), nbrs.end());
}
// template <typename T>
// void rsort_unique(list<T> &nbrs)
// {
//     nbrs.sort();
//     reverse(nbrs.begin(), nbrs.end());

//     nbrs.erase(unique(nbrs.begin(), nbrs.end()), nbrs.end());
// }

inline uint64_t find_insertion_position(uint64_t time, const vector<uint64_t> &timestamps)
{
    uint64_t low = lower_bound(timestamps.begin(), timestamps.end(), time) - timestamps.begin();
    if (timestamps[low] == time)
        low++;
    return low;
}

template <bool temporal>
class TemporalBase
{
};

template <>
class TemporalBase<true>
{
public:
    uint64_t time = numeric_limits<uint64_t>::max();
};

template <bool maximal>
class MaximalBase
{
};
template <>
class MaximalBase<true>
{
public:
    vector<uint64_t> forbidden;
    vector<uint64_t> pivot_candidates;
    uint64_t pivot;
};

//by combing inheritance and template
//we don't need to pay for the extra time&space

template <bool temporal, bool maximal>
class PartialClique : public TemporalBase<temporal>, public MaximalBase<maximal>
{
public:
    vector<uint64_t> base;
    vector<uint64_t> candidates;
    // vector<uint64_t> &later_neighbors();
    // operator vector<uint64_t> *() const { return this->values_; };

    template <bool temporal_ = temporal>
    typename enable_if<temporal_, void>::type
    update_time(unordered_map<array<uint64_t, 2>, vector<uint64_t>, ArrayHasher> &edgetime)
    {
        vector<uint64_t> new_edge_timestamps;
        for (uint i = 0; i < base.size() - 1; i++)
        {
            //check the newly-added edges' timestamps
            uint64_t node0 = min(base.back(), base[i]), node1 = max(base.back(), base[i]);
            new_edge_timestamps.push_back(edgetime[{node0, node1}].front()); //the earliest timestamp is taken
        }
        uint64_t new_time = *max_element(new_edge_timestamps.begin(), new_edge_timestamps.end());
        this->time = max(this->time, new_time);
    };

    void print()
    {
        cout << "base: ";
        for (auto b : base)
            cout << b << ", ";
        if constexpr (maximal)
        {
            cout << "snbrs: ";
            for (auto b : this->forbidden)
                cout << b << ", ";
            cout << "pnbrs: ";
            for (auto b : this->pivot_candidates)
                cout << b << ", ";
            cout << "pivot: ";
            cout << this->pivot << " ";
        }
        cout << "nbrs: ";
        for (auto b : candidates)
            cout << b << ", ";

        if constexpr (temporal)
            cout << "time: " << this->time;
        cout << endl;
    };
};

template <bool temporal, bool maximal>
class Compare
{
public:
    bool operator()(const PartialClique<temporal, maximal> &a, const PartialClique<temporal, maximal> &b) const
    {
        if constexpr (temporal)
        {
            return a.time > b.time;
        }
        else
        {
            return false;
        }
    };
};

template <bool temporal, bool maximal>
class CliqueCounter
{
protected:
    typedef unordered_map<array<uint64_t, 2>, vector<uint64_t>, ArrayHasher> array_map;
    unsigned int max_clique_size;                   //the number of cliques larger than this is not counted
    uint64_t count_limit;                           //the temporal cliques formed after the time when there exists #k-clique > 'count_limit' would not be counted
    bool stop_flag;                                 //this'll be set to true once 'count_limit' is reached
    uint64_t stop_time;                             //clique formed after this timestamp will be discarded
    vector<PartialClique<temporal, maximal>> stack; //LIFO container for partial cliques
    vector<PartialClique<temporal, maximal>> adjs;  //adjacency list
    array_map edgetime;                             //mapping from edge to timestamp
    vector<uint64_t> timestamps;                    //timestamps that partition a temporal graph into several stages
    int num_working_threads = 0;                    //number of active threads
    size_t tlen;                                    //#timestamps, which'll be set to 1 when performing static-counting
    vector<vector<uint64_t>> clique_count;          //the final result

public:
    CliqueCounter(const vector<array<uint64_t, 2>> &edges,
                  vector<uint64_t> &edgetimestamps,
                  vector<uint64_t> timestamps,
                  unsigned int max_clique_size,
                  uint64_t count_limit)
        : timestamps(timestamps),
          max_clique_size(max_clique_size),
          count_limit(count_limit)
    {
        if constexpr (temporal)
        {
            stop_time = timestamps[0];
        }
        stop_flag = false;
        tlen = temporal ? timestamps.size() : 1;
        //find the largest clique if max_clique_size is set to <= 0
        max_clique_size = max_clique_size < 0 ? 0 : max_clique_size;
        clique_count = vector<vector<uint64_t>>(tlen, vector<uint64_t>(max_clique_size));
        //to-do: degeneracy order
        prepare(edges, edgetimestamps);
    };

    void prepare(const vector<array<uint64_t, 2>> &edges, vector<uint64_t> &edgetimestamps);

    //find and set the pivot for a partial clique
    void find_pivot(PartialClique<temporal, maximal> &new_pc)
    {
        //only do it for those have many candidates
        //since the time taken for computing pivot might be longer than the time it'd possibly save
        if (new_pc.candidates.size() > 7)
        {
            //to-do: refactor using chained iterator; temporal version of find_pivot
            vector<uint64_t> lens;
            for (auto p : new_pc.candidates)
            {
                vector<uint64_t> common;
                set_intersection(adjs[p].candidates.begin(), adjs[p].candidates.end(),
                                 new_pc.candidates.begin(), new_pc.candidates.end(),
                                 back_inserter(common), comp);
                set_intersection(adjs[p].forbidden.begin(), adjs[p].forbidden.end(),
                                 new_pc.candidates.begin(), new_pc.candidates.end(),
                                 back_inserter(common), comp);
                lens.push_back(common.size());
            }

            vector<uint64_t> lens2;
            for (auto p : new_pc.forbidden)
            {
                vector<uint64_t> common;
                set_intersection(adjs[p].candidates.begin(), adjs[p].candidates.end(),
                                 new_pc.candidates.begin(), new_pc.candidates.end(),
                                 back_inserter(common), comp);
                set_intersection(adjs[p].forbidden.begin(), adjs[p].forbidden.end(),
                                 new_pc.candidates.begin(), new_pc.candidates.end(),
                                 back_inserter(common), comp);
                lens2.push_back(common.size());
            }

            auto max = max_element(lens.begin(), lens.end()), max2 = max_element(lens2.begin(), lens2.end());

            if (max > max2)
            {
                size_t max_index = distance(lens.begin(), max);
                new_pc.pivot = new_pc.candidates[max_index];
            }
            else
            {
                size_t max_index = distance(lens2.begin(), max2);
                new_pc.pivot = new_pc.forbidden[max_index];
            }
        }
        else if (new_pc.candidates.size() > 0)
        {
            new_pc.pivot = new_pc.candidates.back();
        }
    };

    //do depth-first clique expansion for sliced_stack assigned to each thread
    unique_ptr<vector<vector<uint64_t>>> dfs(vector<PartialClique<temporal, maximal>> &sub_stack)
    {
        unique_ptr<vector<vector<uint64_t>>> clique_count_(
            new vector<vector<uint64_t>>(1, vector<uint64_t>(max_clique_size)));

        //count the number of expansions
        size_t k0 = 0;

        while (!sub_stack.empty() && !stop_flag)
        {
            auto &pc = sub_stack.back();
            // #pragma omp critical
            //             pc.print();
            if (pc.candidates.empty() || pc.base.size() == max_clique_size)
            {
                if constexpr (maximal)
                    if (!pc.forbidden.empty() || !pc.candidates.empty())
                    {
                        sub_stack.pop_back();
                        continue;
                    }

                size_t pos = 0;
                if constexpr (temporal)
                {
                    pos = find_insertion_position(pc.time, timestamps);
                    if (pos + 1 > clique_count_->size() && pos + 1 <= timestamps.size())
                        clique_count_->resize(pos + 1);
                }
                // for (auto i = low; i < clique_count_->size(); i++)
                // {
                auto &target = clique_count_->at(pos);
                if (pc.base.size() > target.size())
                    target.resize(pc.base.size());
                target[pc.base.size() - 1] += 1;
                // }

                sub_stack.pop_back();
                continue;
            }

            PartialClique<temporal, maximal> new_pc;
            auto u = pc.candidates.back();
            pc.candidates.pop_back();

            if constexpr (maximal)
            {
                //skip if the candidate is a neighbor of the pivot node
                if (binary_search(adjs[pc.pivot].candidates.begin(), adjs[pc.pivot].candidates.end(), u, comp))
                {
                    pc.pivot_candidates.push_back(u);
                    continue;
                }
                if (binary_search(adjs[pc.pivot].forbidden.begin(), adjs[pc.pivot].forbidden.end(), u, comp))
                {
                    pc.pivot_candidates.push_back(u);
                    continue;
                }
            }
            //expand 'base'
            new_pc.base = pc.base;
            new_pc.base.push_back(u);
            if constexpr (temporal)
            {
                new_pc.time = pc.time;
                // new_pc.time = 0;
                new_pc.update_time(edgetime);
            }

            //N_later(u) intersect P
            set_intersection(adjs[u].candidates.begin(), adjs[u].candidates.end(),
                             pc.candidates.begin(), pc.candidates.end(),
                             back_inserter(new_pc.candidates), comp);
            if constexpr (maximal)
            {
                //N_earlier(u) intersect P
                set_intersection(adjs[u].forbidden.begin(), adjs[u].forbidden.end(),
                                 pc.pivot_candidates.rbegin(), pc.pivot_candidates.rend(),
                                 back_inserter(new_pc.candidates), comp);
                //N_later(u) intersect X
                set_intersection(adjs[u].candidates.begin(), adjs[u].candidates.end(),
                                 pc.forbidden.begin(), pc.forbidden.end(),
                                 back_inserter(new_pc.forbidden), comp);
                //N_earlier(u) intersect X
                set_intersection(adjs[u].forbidden.begin(), adjs[u].forbidden.end(),
                                 pc.forbidden.begin(), pc.forbidden.end(),
                                 back_inserter(new_pc.forbidden), comp);

                //insert to an ordered vector
                auto pos = lower_bound(pc.forbidden.begin(), pc.forbidden.end(), u, comp);
                pc.forbidden.insert(pos, u);
                find_pivot(new_pc);
            }
            sub_stack.push_back(new_pc);
            k0++;
            if (k0 > 500)
            {
                // cout << num_working_threads << endl;
                if (num_working_threads < omp_get_max_threads() && stack.size() == 0)
                {
                    break;
                }
                else
                    k0 = 0;
            }
        }
        return clique_count_;
    };

    vector<vector<uint64_t>> run()
    {
#pragma omp parallel shared(num_working_threads)
        while (true)
        {
            vector<PartialClique<temporal, maximal>> sub_stack;
#pragma omp critical
            if (!stack.empty())
            {
                sub_stack.push_back(stack.back());
                stack.pop_back();
                num_working_threads++;
            }

            //currently setting count limit is only compatible with static graphs
            if constexpr (!temporal)
            {
#pragma omp critical
                if (count_limit != 0 && !clique_count[0].empty() && !stop_flag)
                {
                    auto max_num = *max_element(clique_count[0].begin(),
                                                clique_count[0].end());

                    if (max_num > count_limit)
                    {
                        cout << max_num << " > " << count_limit << endl;
                        cout << "exiting..." << endl;
                        stop_flag = true;
                    }
                }
            }

            if (stop_flag)
                break;

            if (num_working_threads == 0)
            {
                break;
            }

            //ask again when received no work
            if (sub_stack.size() == 0)
                continue;

            auto temp = dfs(sub_stack);
#pragma omp critical
            num_working_threads--;

//merge results
#pragma omp critical
            //loop over all given times
            for (uint i = 0; i < temp->size(); i++)
            {
                //skip if the result at this timestamp is empty
                if (temp->at(i).size() == 0)
                    continue;
                //resize the container if the to-be-merged result is longer (i.e. has larger cliques)
                clique_count[i].resize(max(clique_count[i].size(), temp->at(i).size()));
                //add upp the numbers for each clique size
                for (size_t j = 0; j < temp->at(i).size(); j++)
                    clique_count[i].at(j) += temp->at(i).at(j);
            }

            //put the remaining work back to the main stack
            if (sub_stack.size() > 0)
            {
#pragma omp critical
                move(sub_stack.begin(), sub_stack.end(), back_inserter(stack));
            }
        }
        if (stop_flag)
        {
            for (size_t i = 0; i < clique_count[0].size(); i++)
                clique_count[0][i] = 0;
        }

        return clique_count;
    };

    //     vector<vector<uint64_t>> run_single_stack_ver()
    //     {
    //         // boost::heap::fibonacci_heap<PartialClique<temporal, maximal>,
    //         //                             boost::heap::compare<Compare<temporal, maximal>>>
    //         //     pq;
    //         priority_queue<PartialClique<temporal, maximal>,
    //                        vector<PartialClique<temporal, maximal>>,
    //                        Compare<temporal, maximal>>
    //             pq;

    //         for (auto node : stack)
    //         {
    //             pq.push(node);
    //         }

    // #pragma omp parallel shared(num_working_threads)
    //         while (!stop_flag)
    //         {
    //             PartialClique<temporal, maximal> pc;
    // #pragma omp critical
    //             if (!pq.empty())
    //             {
    //                 pc = pq.top();
    //                 pq.pop();
    //                 num_working_threads++;
    //             }

    //             if (pc.base.size() == 0 && num_working_threads != 0)
    //                 continue;
    //             // cout << num_working_threads << endl;
    //             if (num_working_threads == 0)
    //                 break;
    // // #pragma omp critical
    // //             {
    // //                 pc.print();
    // //                 cout << pc.candidates.empty() << endl;
    // //             }

    //             if (pc.candidates.empty() || pc.base.size() == max_clique_size)
    //             {
    //                 if constexpr (maximal)
    //                     if (!pc.forbidden.empty() || !pc.candidates.empty())
    //                     {
    //                         // pq.pop();
    // #pragma omp critical
    //                         num_working_threads--;
    //                         continue;
    //                     }

    //                 size_t pos = 0;
    //                 if constexpr (temporal)
    //                 {
    //                     pos = find_insertion_position(pc.time, timestamps);
    //                     // #pragma omp critical
    //                     //                     if (pos + 1 > clique_count.size() && pos + 1 <= timestamps.size())
    //                     //                         clique_count.resize(pos + 1);
    //                 }

    // #pragma omp critical
    //                 {
    //                     // if constexpr (temporal)
    //                     //     cout << "time: " << pc.time << endl;
    //                     // cout << "pos: " << pos << endl;
    //                     auto &target = clique_count.at(pos);
    //                     if (pc.base.size() > target.size())
    //                         target.resize(pc.base.size());
    //                     target[pc.base.size() - 1] += 1;
    //                 }
    // #pragma omp critical
    //                 num_working_threads--;
    //                 continue;
    //             }

    //             PartialClique<temporal, maximal> new_pc;
    //             auto u = pc.candidates.back();
    //             pc.candidates.pop_back();

    //             if constexpr (maximal)
    //             {
    //                 //skip if the candidate is a neighbor of the pivot node
    //                 if (binary_search(adjs[pc.pivot].candidates.begin(), adjs[pc.pivot].candidates.end(), u, comp))
    //                 {
    //                     pc.pivot_candidates.push_back(u);
    //                     continue;
    //                 }
    //                 if (binary_search(adjs[pc.pivot].forbidden.begin(), adjs[pc.pivot].forbidden.end(), u, comp))
    //                 {
    //                     pc.pivot_candidates.push_back(u);
    //                     continue;
    //                 }
    //             }
    //             //expand 'base'
    //             new_pc.base = pc.base;
    //             new_pc.base.push_back(u);

    //             if constexpr (temporal)
    //             {
    //                 new_pc.time = pc.time;
    //                 new_pc.update_time(edgetime);
    //             }

    //             //N_later(u) intersect P
    //             set_intersection(adjs[u].candidates.begin(), adjs[u].candidates.end(),
    //                              pc.candidates.begin(), pc.candidates.end(),
    //                              back_inserter(new_pc.candidates), comp);
    //             if constexpr (maximal)
    //             {
    //                 //N_earlier(u) intersect P
    //                 set_intersection(adjs[u].forbidden.begin(), adjs[u].forbidden.end(),
    //                                  pc.pivot_candidates.rbegin(), pc.pivot_candidates.rend(),
    //                                  back_inserter(new_pc.candidates), comp);
    //                 //N_later(u) intersect X
    //                 set_intersection(adjs[u].candidates.begin(), adjs[u].candidates.end(),
    //                                  pc.forbidden.begin(), pc.forbidden.end(),
    //                                  back_inserter(new_pc.forbidden), comp);
    //                 //N_earlier(u) intersect X
    //                 set_intersection(adjs[u].forbidden.begin(), adjs[u].forbidden.end(),
    //                                  pc.forbidden.begin(), pc.forbidden.end(),
    //                                  back_inserter(new_pc.forbidden), comp);

    //                 //insert to an ordered vector
    //                 auto pos = lower_bound(pc.forbidden.begin(), pc.forbidden.end(), u, comp);
    //                 pc.forbidden.insert(pos, u);
    //                 find_pivot(new_pc);
    //             }

    // #pragma omp critical
    //             pq.push(pc);
    // #pragma omp critical
    //             {
    //                 pq.push(new_pc);
    //                 num_working_threads--;
    //             }
    //         }
    //         return clique_count;
    //     };
};

template <bool temporal, bool maximal>
void CliqueCounter<temporal, maximal>::prepare(const vector<array<uint64_t, 2>> &edges,
                                               vector<uint64_t> &edgetimestamps)
{
    if constexpr (temporal)
    {
        for (uint i = 0; i < edges.size(); i++)
        {
            //directionality is ignored
            //put all timestamps for an edge in a vector
            array<uint64_t, 2> edge;
            edge[0] = *min_element(edges[i].begin(), edges[i].end());
            edge[1] = *max_element(edges[i].begin(), edges[i].end());
            edgetime[edge].push_back(edgetimestamps[i]);
        }
        for (auto pair : edgetime)
        {
            //sort the timestamps in increasing order
            //we'll only use the earliest(first) timestamp for now
            sort(pair.second.begin(), pair.second.end());
            // pair.second.erase(unique(pair.second.begin(), pair.second.end()), pair.second.end());
        }
    }

    //build a node set from edges
    set<uint64_t> nodes;
    for (auto edge : edges)
    {
        nodes.insert(edge[0]);
        nodes.insert(edge[1]);
    }

    //build adjacency list
    //'adjs' shares the same data structure as 'stack' but has different meaning for the member variables
    adjs.resize(nodes.size());
    for (auto edge : edges)
    {
        uint64_t node1 = min(edge[0], edge[1]), node2 = max(edge[0], edge[1]);
        if (node1 != node2)
        {
            //'candidates' refers to the neighbors whose indices are greater
            adjs[node1].candidates.push_back(node2);
            if constexpr (maximal)
                //'forbidden' refers to the neighbors whose indices are smaller
                adjs[node2].forbidden.push_back(node1);
        }
        if constexpr (temporal)
        {
            adjs[node1].time = min(adjs[node1].time, edgetime[{node1, node2}].front());
            adjs[node2].time = min(adjs[node2].time, edgetime[{node1, node2}].front());
        }
    }

    for (auto node : nodes)
    {
        adjs[node].base.push_back(node);
        //reverse it in order to place the nodes with smaller indices at the end of the vector
        //those nodes are the candidate nodes that'll be poped-back first
        //see dfs for details
        rsort_unique(adjs[node].candidates);
        if constexpr (maximal)
        {
            rsort_unique(adjs[node].forbidden);
            find_pivot(adjs[node]); //this is needed for 'stack'
        }
        // adjs[node].print();
    }

    //at the beginning, 'stack' and 'adjs' will be exactly the same
    //they'll differ only after we start to expand partial cliques larger than 1
    stack.reserve(adjs.size());
    copy(adjs.begin(), adjs.end(), back_inserter(stack));

    // stack.resize(timestamps.size());
    // for (auto node : adjs)
    // {
    //     size_t pos = 0;
    //     if constexpr (temporal)
    //     {
    //         pos = find_insertion_position(node.time, timestamps);
    //     }
    //     stack.push_back(node);
    // }
};

template <bool temporal, bool maximal>
vector<vector<uint64_t>> count_cliques(vector<array<uint64_t, 2>> &edges,
                                       vector<uint64_t> &edgetimestamps,
                                       vector<uint64_t> &timestamps,
                                       unsigned int clique_size,
                                       uint64_t count_limit)
{
    CliqueCounter<temporal, maximal> clique_counter(edges,
                                                    edgetimestamps,
                                                    timestamps,
                                                    clique_size,
                                                    count_limit);
    //time the code
    //the time for building the adjacency list and preparing stack is not included
    struct timeval start, end;
    gettimeofday(&start, NULL);
    vector<vector<uint64_t>> clique_count = clique_counter.run();
    gettimeofday(&end, NULL);
    double delta = ((end.tv_sec - start.tv_sec) * 1000000u +
                    end.tv_usec - start.tv_usec) /
                   1.e6f;
    cout << "It took " << delta << endl;
    return clique_count;
}

typedef vector<vector<uint64_t>> (*fptr)(vector<array<uint64_t, 2>> &edges,
                                         vector<uint64_t> &edgetimestamps,
                                         vector<uint64_t> &timestamps,
                                         unsigned int clique_size,
                                         uint64_t count_limit);

template <int i0, int i1>
class loop
{
    //all combinations from below options will be instantiated during compile-time
    constexpr static bool temporal_option[2] = {false, true};
    constexpr static bool maximal_option[2] = {false, true};
    constexpr static int i0max = extent<decltype(temporal_option)>::value;
    constexpr static int i1max = extent<decltype(maximal_option)>::value;

public:
    // constexpr fptr build_map()
    // {
    //     unordered_map<array<bool, 2>, fptr, ArrayHasher> funcs;
    //     for (int ii0 = 0; ii0 < i0max; ii0++)
    //     {
    //         for (int ii1 = 0; ii1 < i1max; ii1++)
    //         {
    //             funcs[{temporal_option[ii0], maximal_option[ii1]}] = count_cliques<temporal_option[ii0], maximal_option[ii1]>;
    //         }
    //     }
    // };
    static fptr get_function(bool temporal, bool maximal)
    {
        if constexpr (i0 < i0max && i1 < i1max)
        {
            if (temporal == temporal_option[i0] && maximal == maximal_option[i1])
            {
                return count_cliques<temporal_option[i0], maximal_option[i1]>;
            }
            constexpr int i0p = i1 == i0max - 1 ? i0 + 1 : i0;
            constexpr int i1p = i1 == i0max - 1 ? 0 : i1 + 1;
            //mimicking a 2d for-loop during compile-time
            return loop<i0p, i1p>::get_function(temporal, maximal);
        }
        throw invalid_argument("invalid option combination");
    };
};

vector<vector<uint64_t>> count_cliques_dispatcher(vector<array<uint64_t, 2>> edges,
                                                  vector<uint64_t> &edgetimestamps, vector<uint64_t> &timestamps,
                                                  unsigned int clique_size,
                                                  uint64_t count_limit,
                                                  const bool temporal,
                                                  const bool count_maximal_only)
{
    //a bridge dynamic and static polymorphism
    //it can choose the corresponding template function based on user inputs
    fptr func = loop<0, 0>::get_function(temporal, count_maximal_only);
    return (*func)(edges,
                   edgetimestamps,
                   timestamps,
                   clique_size,
                   count_limit);
}

PYBIND11_MODULE(clique_counter, m)
{
    m.doc() = "(maximal) clique counter for static/temporal graphs";
    m.def("count_cliques", &count_cliques_dispatcher, "count_cliques");
}
