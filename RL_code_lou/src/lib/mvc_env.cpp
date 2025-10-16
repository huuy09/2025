#include "mvc_env.h"
#include "graph.h"
#include <cassert>
#include <random>
#include <algorithm>
#include <set>
#include <queue>
#include <stack>


MvcEnv::MvcEnv(double _norm)
{
norm = _norm;
graph = nullptr;
graph_dual = nullptr;
numCoveredEdges = 0;
numCoveredNodes = 0;
CcNum = 1.0;
state_seq.clear();
act_seq.clear();
action_list.clear();
reward_seq.clear();
sum_rewards.clear();
covered_set.clear();
edge_covered_set.clear();
avail_list.clear();
avail_list_edge.clear();
save_state_cnt = 0;
save_every_nstep = 0;
current_weight = 0.0;
}

MvcEnv::~MvcEnv()
{
    norm = 0;
    graph = nullptr;
    graph_dual = nullptr;
    numCoveredEdges = 0;
    numCoveredNodes = 0;
    state_seq.clear();
    act_seq.clear();
    action_list.clear();
    reward_seq.clear();
    sum_rewards.clear();
    covered_set.clear();
    edge_covered_set.clear();
    avail_list.clear();
    avail_list_edge.clear();
    save_state_cnt = 0;
    save_every_nstep = 0;
    current_weight = 0.0;
}

void MvcEnv::s0(std::shared_ptr<Graph> _g, std::shared_ptr<Graph> _g_dual, int nstep)
{
    graph = _g;
    graph_dual = _g_dual;
    covered_set.clear();
    edge_covered_set.clear();
    action_list.clear();
    numCoveredEdges = 0;
    numCoveredNodes = 0;
    CcNum = 1.0;
    state_seq.clear();
    act_seq.clear();
    reward_seq.clear();
    sum_rewards.clear();
    save_state_cnt = 0;
    save_every_nstep = nstep;
    current_weight = 0.0;
}

double MvcEnv::step(int a)

{
    assert(graph);
    // remove node
    if (a < graph->num_nodes){
        assert(covered_set.count(a) == 0);
        // save state information
        if (save_state_cnt == 0){
            std::pair<std::set<int>, std::set<int>> p = std::make_pair(covered_set, edge_covered_set);
            state_seq.push_back(p);
            action_list.push_back(a);
            act_seq.push_back(a);
        }
        // remove node
        covered_set.insert(a);

        // find this node's edge and remove them
        numCoveredNodes += 1;
        for (auto neigh : graph->adj_list[a]){
            int v1 = std::min(a, neigh), v2 = std::max(a, neigh);
            std::pair<int, int> e = std::make_pair(v1, v2);
            int idx = graph->edge_map[e];
            if (edge_covered_set.count(idx) == 0){
                edge_covered_set.insert(idx);
                numCoveredEdges += 1;
            }
        }
    }
    // remove edge
    else{
        int idx = a - graph->num_nodes;
        assert(edge_covered_set.count(idx) == 0);
        // save state information
        if (save_state_cnt == 0){
            std::pair<std::set<int>, std::set<int>> p = std::make_pair(covered_set, edge_covered_set);
            state_seq.push_back(p);
            act_seq.push_back(a);
            action_list.push_back(a);
        }
        // remove edge
        edge_covered_set.insert(idx);

        numCoveredEdges += 1;
    }

    // update save state counter
    save_state_cnt = (save_state_cnt + 1) % save_every_nstep;

    return 0.0;
}

void MvcEnv::updateReward(double r, int a){
    // r is the reward of pairwies connectivity compute by python function
    r += getReward() * 0.5;

    // multiply with the removed target's weight over total weight
    if (a < graph->num_nodes){
        r *= graph->nodes_weight[a] / graph_dual->total_nodes_weight;
    }
    else{
        r *= graph_dual->nodes_weight[a - graph->num_nodes] / graph_dual->total_nodes_weight;
    }
    reward_seq.push_back(r);
    sum_rewards.push_back(r);
}


void MvcEnv::stepWithoutReward(int a)

{
    assert(graph);
    if (a < graph->num_nodes){
        assert(covered_set.count(a) == 0);
        covered_set.insert(a);

        numCoveredNodes += 1;
        for (auto neigh : graph->adj_list[a]){
            int v1 = std::min(a, neigh), v2 = std::max(a, neigh);
            std::pair<int, int> e = std::make_pair(v1, v2);
            int idx = graph->edge_map[e];
            if (edge_covered_set.count(idx) == 0){
                edge_covered_set.insert(idx);
                numCoveredEdges += 1;
            }
        }
    }
    else{
        int idx = a - graph->num_nodes;
        assert(edge_covered_set.count(idx) == 0);
        edge_covered_set.insert(idx);

        numCoveredEdges += 1;
    }
}


// random
int MvcEnv::randomAction()
{
    assert(graph);
    avail_list.clear();
    avail_list_edge.clear();

    // filter non-isolated node
    for (int i = 0; i < graph->num_nodes; ++i){
        if (covered_set.count(i) == 0)
        {
            bool useful = false;
            for (auto neigh : graph->adj_list[i])
                if (covered_set.count(neigh) == 0)
                {
                    useful = true;
                    break;
                }
            if (useful)
                avail_list.push_back(i);
        }
    }
    // filter non-covered edges
    for (int i = 0; i < graph->num_edges; i++){
        if (edge_covered_set.count(i) == 0){
            avail_list_edge.push_back((i + graph->num_nodes));
        }
    }
    // choose an object
    assert(avail_list.size());
    assert(avail_list_edge.size());
    int idx_node = rand() % avail_list.size();
    int idx_edge = rand() % avail_list_edge.size();
    double r = (double) rand() / (RAND_MAX);
    // |V| <<< |E|
    // r is a random number between 0~1, to prevent DQN always choose edge in a random action
    if (r >= 0.5){
        return avail_list[idx_node];
    }
    else{
        return avail_list_edge[idx_edge];
    }
    // return avail_list[idx];
}

////degree
//int MvcEnv::randomAction()
//{
//    assert(graph);
//    avail_list.clear();
//
//    int maxID = -1;
//    int maxDegree = 0;
//    for (int i = 0; i < graph->num_nodes; ++i)
//    {
//        int degree = 0;
//        if (covered_set.count(i) == 0)
//        {
//            for (auto neigh : graph->adj_list[i])
//                if (covered_set.count(neigh) == 0)
//                {
//                    degree++;
//                }
//        }
//        if(degree>maxDegree){
//            maxDegree = degree;
//            maxID = i;
//        }
//    }
//    return maxID;
//}


 //betweenness
//int MvcEnv::randomAction()
//{
//    assert(graph);
//
//    std::map<int,int> id2node;
//    std::map<int,int> node2id;
//
//    std::map <int,std::vector<int>> adj_dic_origin;
//    std::vector<std::vector<int>> adj_list_reID;
//
//
//    for (int i = 0; i < graph->num_nodes; ++i)
//    {
//        if (covered_set.count(i) == 0)
//        {
//            for (auto neigh : graph->adj_list[i])
//            {
//                if (covered_set.count(neigh) == 0)
//                {
//                   if(adj_dic_origin.find(i) != adj_dic_origin.end())
//                   {
//                       adj_dic_origin[i].push_back(neigh);
//                   }
//                   else{
//                       std::vector<int> neigh_list;
//                       neigh_list.push_back(neigh);
//                       adj_dic_origin.insert(std::make_pair(i,neigh_list));
//                   }
//                }
//            }
//        }
//
//    }
//
//
//     std::map<int, std::vector<int>>::iterator iter;
//     iter = adj_dic_origin.begin();
//
//     int numrealnodes = 0;
//     while(iter != adj_dic_origin.end())
//     {
//        id2node[numrealnodes] = iter->first;
//        node2id[iter->first] = numrealnodes;
//        numrealnodes += 1;
//        iter++;
//     }
//
//     adj_list_reID.resize(adj_dic_origin.size());
//
//     iter = adj_dic_origin.begin();
//     while(iter != adj_dic_origin.end())
//     {
//        for(int i=0;i<iter->second.size();++i){
//            adj_list_reID[node2id[iter->first]].push_back(node2id[iter->second[i]]);
//        }
//        iter++;
//     }
//
//
//    std::vector<double> BC = Betweenness(adj_list_reID);
//    std::vector<double>::iterator biggest_BC = std::max_element(std::begin(BC), std::end(BC));
//    int maxID = std::distance(std::begin(BC), biggest_BC);
//    int idx = id2node[maxID];
////    printGraph();
////    printf("\n maxBetID:%d, value:%.6f\n",idx,BC[maxID]);
//    return idx;
//}

int MvcEnv::betweenAction()
{
    assert(graph);

    std::map<int,int> id2node;
    std::map<int,int> node2id;

    std::map <int,std::vector<int>> adj_dic_origin;
    std::vector<std::vector<int>> adj_list_reID;


    for (int i = 0; i < graph->num_nodes; ++i)
    {
        if (covered_set.count(i) == 0)
        {
            for (auto neigh : graph->adj_list[i])
            {
                if (covered_set.count(neigh) == 0)
                {
                   if(adj_dic_origin.find(i) != adj_dic_origin.end())
                   {
                       adj_dic_origin[i].push_back(neigh);
                   }
                   else{
                       std::vector<int> neigh_list;
                       neigh_list.push_back(neigh);
                       adj_dic_origin.insert(std::make_pair(i,neigh_list));
                   }
                }
            }
        }

    }


     std::map<int, std::vector<int>>::iterator iter;
     iter = adj_dic_origin.begin();

     int numrealnodes = 0;
     while(iter != adj_dic_origin.end())
     {
        id2node[numrealnodes] = iter->first;
        node2id[iter->first] = numrealnodes;
        numrealnodes += 1;
        iter++;
     }

     adj_list_reID.resize(adj_dic_origin.size());

     iter = adj_dic_origin.begin();
     while(iter != adj_dic_origin.end())
     {
        for(int i=0;i<(int)iter->second.size();++i){
            adj_list_reID[node2id[iter->first]].push_back(node2id[iter->second[i]]);
        }
        iter++;
     }


    std::vector<double> BC = Betweenness(adj_list_reID);
    std::vector<double>::iterator biggest_BC = std::max_element(std::begin(BC), std::end(BC));
    int maxID = std::distance(std::begin(BC), biggest_BC);
    int idx = id2node[maxID];
//    printGraph();
//    printf("\n maxBetID:%d, value:%.6f\n",idx,BC[maxID]);
    return idx;
}

bool MvcEnv::isTerminal()
{
    assert(graph);
    return graph->num_edges * 0.8 <= numCoveredEdges;
}


double MvcEnv::getReward()
{
    // LCC nodes weight / all nodes weight
    return -(double)(getMaxConnectedNodesNum()/graph->total_nodes_weight); //*(graph->nodes_weight[a]/(graph->total_nodes_weight + graph_dual->total_nodes_weight));
}

void MvcEnv::printGraph()
{
    printf("edge_list:\n");
    printf("[");
    for (int i = 0; i < (int)graph->edge_list.size();i++)
    {
    printf("[%d,%d],",graph->edge_list[i].first,graph->edge_list[i].second);
    }
    printf("]\n");


    printf("covered_set:\n");

    std::set<int>::iterator it;
    printf("[");
    for (it=covered_set.begin();it!=covered_set.end();it++)
    {
        printf("%d,",*it);
    }
    printf("]\n");

}

double MvcEnv::getNumofConnectedComponents()
{
    assert(graph);
    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes);

    for (int i = 0; i < graph->num_nodes; i++)
    {
        if (covered_set.count(i) == 0)
        {
            for (auto neigh : graph->adj_list[i])
            {
                if (covered_set.count(neigh) == 0)
                {
                    disjoint_Set.merge(i, neigh);
                }
            }
        }
    }
    std::set<int> lccIDs;
    for(int i =0;i< graph->num_nodes; i++){
        lccIDs.insert(disjoint_Set.unionSet[i]);
    }
    return (double)lccIDs.size();
}

double MvcEnv::getMaxConnectedNodesNum()
{
    assert(graph);
    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes);
    // create disjoint set
    for (int i = 0; i < graph->num_nodes; i++)
    {
        if (covered_set.count(i) == 0)
        {
            for (auto neigh : graph->adj_list[i])
            {
                if (covered_set.count(neigh) == 0)
                {
                    disjoint_Set.merge(i, neigh);
                }
            }
        }
    }

    int targetCount = disjoint_Set.maxRankCount;
    int root = -1;

    // find root
    for (int i = 0; i < graph->num_nodes; i++){
        if (disjoint_Set.rankCount[i] == targetCount){
            root = i;
            break;
        }
    }

    // find nodes in LCC
    double res = 0.0;
    for (int i = 0; i < graph->num_nodes; i++){
        if (disjoint_Set.unionSet[i] == root){
            res += graph->nodes_weight[i];
        }
    }
    return res;
}


std::vector<double> MvcEnv::Betweenness(std::vector< std::vector <int> > adj_list) {

	int i, j, u, v;
	int Long_max = 4294967295;
	int nvertices = adj_list.size();	// The number of vertices in the network
	std::vector<double> CB;
    double norm=(double)(nvertices-1)*(double)(nvertices-2);

	CB.resize(nvertices);

	std::vector<int> d;								// A vector storing shortest distance estimates
	std::vector<int> sigma;							// sigma is the number of shortest paths
	std::vector<double> delta;							// A vector storing dependency of the source vertex on all other vertices
	std::vector< std::vector <int> > PredList;			// A list of predecessors of all vertices

	std::queue <int> Q;								// A priority queue soring vertices
	std::stack <int> S;								// A stack containing vertices in the order found by Dijkstra's Algorithm

	// Set the start time of Brandes' Algorithm

	// Compute Betweenness Centrality for every vertex i
	for (i=0; i < nvertices; i++) {
		/* Initialize */
		PredList.assign(nvertices, std::vector <int> (0, 0));
		d.assign(nvertices, Long_max);
		d[i] = 0;
		sigma.assign(nvertices, 0);
		sigma[i] = 1;
		delta.assign(nvertices, 0);
		Q.push(i);

		// Use Breadth First Search algorithm
		while (!Q.empty()) {
			// Get the next element in the queue
			u = Q.front();
			Q.pop();
			// Push u onto the stack S. Needed later for betweenness computation
			S.push(u);
			// Iterate over all the neighbors of u
			for (j=0; j < (int) adj_list[u].size(); j++) {
				// Get the neighbor v of vertex u
				// v = (ui64) network->vertex[u].edge[j].target;
				v = (int) adj_list[u][j];

				/* Relax and Count */
				if (d[v] == Long_max) {
					 d[v] = d[u] + 1;
					 Q.push(v);
				}
				if (d[v] == d[u] + 1) {
					sigma[v] += sigma[u];
					PredList[v].push_back(u);
				}
			} // End For

		} // End While

		/* Accumulation */
		while (!S.empty()) {
			u = S.top();
			S.pop();
			for (j=0; j < (int)PredList[u].size(); j++) {
				delta[PredList[u][j]] += ((double) sigma[PredList[u][j]]/sigma[u]) * (1+delta[u]);
			}
			if (u != i)
				CB[u] += delta[u];
		}

		// Clear data for the next run
		PredList.clear();
		d.clear();
		sigma.clear();
		delta.clear();
	} // End For

	// End time after Brandes' algorithm and the time difference

    for(int i =0; i<nvertices;++i){
        if (norm == 0)
        {
            CB[i] = 0;
        }
        else
        {
            CB[i]=CB[i]/norm;
        }
    }

	return CB;

} // End of BrandesAlgorithm_Unweighted