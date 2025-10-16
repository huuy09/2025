
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.pair cimport pair
from libcpp.memory cimport shared_ptr
from libcpp cimport bool
from graph cimport Graph

cdef extern from "./src/lib/mvc_env.h":
    cdef cppclass MvcEnv:
        MvcEnv(double _norm)
        void s0(shared_ptr[Graph] _g, shared_ptr[Graph] _g_dual, int nstep)except+
        double step(int a)except+
        void stepWithoutReward(int a)except+
        int randomAction()except+
        int betweenAction()except+
        bool isTerminal()except+
        # double getReward(double oldCcNum)except+
        double getReward()except+
        void updateReward(double r, int a)except+
        double getMaxConnectedNodesNum()except+
        double norm
        double CcNum
        shared_ptr[Graph] graph
        shared_ptr[Graph] graph_dual
        vector[pair[set[int], set[int]]]  state_seq
        vector[int] act_seq
        vector[int] action_list
        vector[double] reward_seq
        vector[double] sum_rewards
        int numCoveredEdges
        int numCoveredNodes
        set[int] covered_set
        set[int] edge_covered_set
        vector[int] avail_list
        vector[int] avail_list_edge
        int save_state_cnt
        int save_every_nstep
        double current_weight
