#ifndef MVC_ENV_H
#define MVC_ENV_H

#include <vector>
#include <set>
#include <memory>
#include "graph.h"
#include "disjoint_set.h"

class MvcEnv
{
public:
    MvcEnv(double _norm);

    ~MvcEnv();

    void s0(std::shared_ptr<Graph> _g, std::shared_ptr<Graph> _g_dual, int nstep);

    double step(int a);

    void updateReward(double r, int a);

    void stepWithoutReward(int a);

    std::vector<double> Betweenness(std::vector< std::vector <int> > adj_list);

    int randomAction();

    int betweenAction();

    bool isTerminal();

    double getReward();

    double getMaxConnectedNodesNum();

    double getNumofConnectedComponents();

    double CcNum;

    void printGraph();

    double norm;

    std::shared_ptr<Graph> graph;

    std::shared_ptr<Graph> graph_dual;

    std::vector< std::pair<std::set<int>, std::set<int>> > state_seq;

    std::vector<int> act_seq, action_list;

    std::vector<double> reward_seq, sum_rewards;

    int numCoveredEdges;

    int numCoveredNodes;

    std::set<int> covered_set;

    std::set<int> edge_covered_set;

    std::vector<int> avail_list, avail_list_edge;

    std::vector<int > node_degrees;

    int total_degrees;

    int save_state_cnt;

    int save_every_nstep;

    double current_weight;
};

#endif