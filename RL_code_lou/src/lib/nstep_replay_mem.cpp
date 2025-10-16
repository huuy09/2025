#include "nstep_replay_mem.h"
#include "i_env.h"
#include <cassert>
#include <algorithm>
#include <time.h>
#include <math.h>

#define max(x, y) (x > y ? x : y)
#define min(x, y) (x > y ? y : x)


 ReplaySample::ReplaySample(int batch_size){
    g_list.resize(batch_size);
    g_dual_list.resize(batch_size);
    list_st_node.resize(batch_size);
    list_st_edge.resize(batch_size);
    list_s_primes.resize(batch_size);
    list_s_primes_edge.resize(batch_size);
    list_at.resize(batch_size);
    list_rt.resize(batch_size);
    list_term.resize(batch_size);
 }
 NStepReplayMem::NStepReplayMem(int _memory_size)
{
    memory_size = _memory_size;
    graphs.resize(memory_size);
    graphs_dual.resize(memory_size);
    actions.resize(memory_size);
    rewards.resize(memory_size);
    states_node.resize(memory_size);
    states_edge.resize(memory_size);
    s_primes.resize(memory_size);
    s_primes_edge.resize(memory_size);
    terminals.resize(memory_size);

    current = 0;
    count = 0;
    distribution = new std::uniform_int_distribution<int>(0, memory_size - 1);
}

void NStepReplayMem::Add(std::shared_ptr<Graph> g, 
                        std::shared_ptr<Graph> g_dual,
                        // std::vector<int> s_t,
                        std::vector<int> s_t_node,
                        std::vector<int> s_t_edge,
                        int a_t, 
                        double r_t,
                        std::vector<int> s_prime,
                        std::vector<int> s_prime_edge,
                        bool terminal)
{
    graphs[current] = g;
    graphs_dual[current] = g_dual;
    actions[current] = a_t;
    rewards[current] = r_t;
    states_node[current] = s_t_node;
    states_edge[current] = s_t_edge;
    s_primes[current] = s_prime;
    s_primes_edge[current] = s_prime_edge;
    terminals[current] = terminal;

    count = max(count, current + 1);
    current = (current + 1) % memory_size; 
}

void NStepReplayMem::Add(std::shared_ptr<MvcEnv> env,int n_step)
{
    assert(env->isTerminal());
    int num_steps = env->state_seq.size();
    assert(num_steps);

    env->sum_rewards[num_steps - 1] = env->reward_seq[num_steps - 1];
    for (int i = num_steps - 1; i >= 0; --i)
        if (i < num_steps - 1)
            env->sum_rewards[i] = env->sum_rewards[i + 1] + env->reward_seq[i];

    for (int i = 0; i < num_steps; ++i)
    {
        bool term_t = false;
        double cur_r;
        std::vector<int> s_prime;
        std::vector<int> s_prime_edge;
        if (i + n_step >= num_steps)
        {
            cur_r = env->sum_rewards[i];
            // s_prime = (env->action_list);
            std::vector<int> v_node(env->covered_set.begin(), env->covered_set.end());
            std::vector<int> v_edge(env->edge_covered_set.begin(), env->edge_covered_set.end());
            s_prime = (v_node);
            s_prime_edge = (v_edge);
            term_t = true;
        } else {
            cur_r = env->sum_rewards[i] - env->sum_rewards[i + n_step];
            std::vector<int> v_node(env->state_seq[i + n_step].first.begin(), env->state_seq[i + n_step].first.end());
            std::vector<int> v_edge(env->state_seq[i + n_step].second.begin(), env->state_seq[i + n_step].second.end());
            s_prime = (v_node);
            s_prime_edge = (v_edge);
        }
        std::vector<int> state_node(env->state_seq[i].first.begin(), env->state_seq[i].first.end());
        std::vector<int> state_edge(env->state_seq[i].second.begin(), env->state_seq[i].second.end());
        Add(env->graph, env->graph_dual, state_node, state_edge, env->act_seq[i], cur_r, s_prime, s_prime_edge, term_t);
    }
}

std::shared_ptr<ReplaySample> NStepReplayMem::Sampling(int batch_size)
{
//    std::shared_ptr<ReplaySample> result {new ReplaySample(batch_size)};
    std::shared_ptr<ReplaySample> result =std::shared_ptr<ReplaySample>(new ReplaySample(batch_size));
    assert(count >= batch_size);

    result->g_list.resize(batch_size);
    result->g_dual_list.resize(batch_size);
    result->list_st_node.resize(batch_size);
    result->list_at.resize(batch_size);
    result->list_rt.resize(batch_size);
    result->list_s_primes.resize(batch_size);
    result->list_term.resize(batch_size);
    auto& dist = *distribution;
    for (int i = 0; i < batch_size; ++i)
    {
        int idx = dist(generator) % count;
        result->g_list[i] = graphs[idx];
        result->g_dual_list[i] = graphs_dual[idx];
        result->list_st_node[i] = (states_node[idx]);
        result->list_st_edge[i] = (states_edge[idx]);
        result->list_at[i] = actions[idx];
        result->list_rt[i] = rewards[idx];
        result->list_s_primes[i] = (s_primes[idx]);
        result->list_s_primes_edge[i] = (s_primes_edge[idx]);
        result->list_term[i] = terminals[idx];
    }
    return result;
}
