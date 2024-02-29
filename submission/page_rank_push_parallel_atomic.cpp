#include "core/graph.h"
#include "core/utils.h"
#include <iomanip>
#include <iostream>
#include <stdlib.h>

#include <vector>
#include <thread>
#include <atomic>

using namespace std;

#ifdef USE_INT
#define INIT_PAGE_RANK 100000
#define EPSILON 1000
#define PAGE_RANK(x) (15000 + (5 * x) / 6)
#define CHANGE_IN_PAGE_RANK(x, y) std::abs(x - y)
typedef int64_t PageRankType;
#else
#define INIT_PAGE_RANK 1.0
#define EPSILON 0.01
#define DAMPING 0.85
#define PAGE_RANK(x) (1 - DAMPING + DAMPING * x)
#define CHANGE_IN_PAGE_RANK(x, y) std::fabs(x - y)
typedef double PageRankType;
#endif

void pageRankHelperAtomic1(Graph &g, int max_iters, uintV start, uintV end, std::atomic<PageRankType> *pr_curr, std::atomic<PageRankType> *pr_next, double &vertices_processed, double &edges_processed, CustomBarrier &barrier1, CustomBarrier &barrier2, double &total_time_taken, double &time_taken_barrier1, double &time_taken_barrier2, double &getNextVertex_time) 
{
  uintV n = g.n_;

  timer t_barrier1;
  timer t_barrier2;
  timer t1;
  t1.start();

  for (int iter = 0; iter < max_iters; iter++) {
    // for each vertex 'u', process all its outNeighbors 'v'
    for (uintV u = start; u < end; u++) {
      uintE out_degree = g.vertices_[u].getOutDegree();
      edges_processed += out_degree;  // Update edges_processed
      for (uintE i = 0; i < out_degree; i++) {
        uintV v = g.vertices_[u].getOutNeighbor(i);
        PageRankType update = pr_curr[u] / (PageRankType)out_degree;
        
        // Atomically update pr_next[v]
        PageRankType old_value = pr_next[v].load();
        PageRankType new_value;
        do {
          new_value = old_value + update;} 
        while (!pr_next[v].compare_exchange_weak(old_value, new_value));  
      }
    }

    // Wait for all threads to complete computation
    t_barrier1.start();
    barrier1.wait();
    time_taken_barrier1 += t_barrier1.stop();

    // Update PageRanks using the computed values
    for (uintV v = start; v < end; v++) {

      PageRankType new_pagerank = PAGE_RANK(pr_next[v].load());
        // reset for the next iteration
        pr_curr[v].store(new_pagerank);
        pr_next[v].store(0.0);
    }

    // Wait for all threads to complete updating PageRanks
    t_barrier2.start();
    barrier2.wait();
    time_taken_barrier2 += t_barrier2.stop();
  }
  vertices_processed = end - start;  // Update vertices_processed 
  total_time_taken = t1.stop();
  getNextVertex_time = 0.0;
}

void pageRankHelperAtomic2(Graph &g, int max_iters, int n_threads, uintV start, uintV end, std::atomic<PageRankType> *pr_curr, std::atomic<PageRankType> *pr_next, double &vertices_processed, double &edges_processed, CustomBarrier &barrier1, CustomBarrier &barrier2, double &total_time_taken, double &time_taken_barrier1, double &time_taken_barrier2, double &getNextVertex_time) 
{
  uintV n = g.n_;

  timer t_barrier1;
  timer t_barrier2;
  timer t1;
  t1.start();

  vector<double> threadEdgeCounts(n_threads, 0);

  for (int iter = 0; iter < max_iters; iter++) {
    // Compute page ranks
    for (int t = 0; t < n_threads; t++) {
      for (uintV v = start + t; v < end; v += n_threads) {
        uintE out_degree = g.vertices_[v].getOutDegree();
        for (uintE j = 0; j < out_degree; j++) {
          uintV u = g.vertices_[v].getOutNeighbor(j);
          PageRankType update = pr_curr[u] / (PageRankType)out_degree;
          
          // Atomically update pr_next[v]
          PageRankType old_value = pr_next[v].load(std::memory_order_relaxed);
          PageRankType new_value;
          do {
            new_value = old_value + update;} 
          while (!pr_next[v].compare_exchange_weak(old_value, new_value)); 
          threadEdgeCounts[t]+= 1; // Increment edge count for this thread
        }
      }
    }

    // Barrier 1
    t_barrier1.start();
    barrier1.wait();
    time_taken_barrier1 += t_barrier1.stop();

    // Update page ranks
    for (int t = 0; t < n_threads; t++) {
      for (uintV v = start + t; v < end; v += n_threads) {
        PageRankType new_pagerank = pr_next[v].load(std::memory_order_relaxed);
        pr_curr[v] = new_pagerank;
        pr_next[v] = 0;
      }
    }
    // Barrier 2
    t_barrier2.start();
    barrier2.wait();
    time_taken_barrier2 += t_barrier2.stop();
  }
  total_time_taken = t1.stop();
  getNextVertex_time = 0.0;

  // Calculate vertices_processed and edges_processed
  double total_edges_processed = 0;
  for (int i = 0; i < n_threads; i++) {
    total_edges_processed += threadEdgeCounts[i];
  }
  edges_processed = total_edges_processed;
  vertices_processed = (end - start + n_threads - 1) / n_threads;
}

uintV getNextVertexToBeProcessed(int granularity, uintV n, std::atomic<uintV> &next_vertex) {
  uintV current = next_vertex.fetch_add(granularity);
  if (current >= n) {
    return n; // Return n to indicate that there are no more vertices to process
  }
  return current;
}

void pageRankHelperAtomic3(Graph &g, int max_iters, std::atomic<uintV> &next_vertex, double &vertices_processed, double &edges_processed, std::atomic<PageRankType> *pr_curr, std::atomic<PageRankType> *pr_next, CustomBarrier &barrier1, CustomBarrier &barrier2, double &total_time_taken, double &time_taken_barrier1, double &time_taken_barrier2, double &getNextVertex_time) 
{
  uintV n = g.n_;

  timer t_barrier1;
  timer t_barrier2;
  timer t_nextVertex;
  timer t_total;
  t_total.start();

  double total_getNextVertex_time = 0.0;

  for (int iter = 0; iter < max_iters; iter++) 
  {
    uintE local_edges_processed = 0;
    uintE local_vertices_processed = 0;

    while (true) {
      t_nextVertex.start();
      uintV u = getNextVertexToBeProcessed(1, n, next_vertex);
      getNextVertex_time += t_nextVertex.stop();
      if (u == -1) break;
      local_edges_processed += g.vertices_[u].getOutDegree();

      for (uintE i = 0; i < g.vertices_[u].getOutDegree(); i++) {
        uintV v = g.vertices_[u].getOutNeighbor(i);
        pr_next[v].store(pr_next[v].load() + pr_curr[u] / (PageRankType)g.vertices_[u].getOutDegree());
      }
    }
    t_barrier1.start();
    barrier1.wait();
    time_taken_barrier1 += t_barrier1.stop();

    while (true) {
      t_nextVertex.start();
      uintV v = getNextVertexToBeProcessed(1, n, next_vertex);
      getNextVertex_time += t_nextVertex.stop();
      
      if (v == -1) break;
      local_vertices_processed++;

      PageRankType new_pagerank = PAGE_RANK(pr_next[v].load());
      pr_curr[v] = new_pagerank;
      pr_next[v].store(0.0);
    }
    //vertices_processed.fetch_add(local_vertices_processed);
    t_barrier2.start();
    barrier2.wait();
    time_taken_barrier2 += t_barrier2.stop();

    edges_processed = local_edges_processed;
    vertices_processed = local_vertices_processed;
  }
  total_time_taken = t_total.stop();
}

void pageRankHelperAtomic4(Graph &g, int max_iters, int n_threads, int granularity, std::atomic<uintV> &next_vertex, double &vertices_processed, double &edges_processed, std::atomic<PageRankType> *pr_curr, std::atomic<PageRankType> *pr_next, CustomBarrier &barrier1, CustomBarrier &barrier2, double &total_time_taken, double &time_taken_barrier1, double &time_taken_barrier2, double &getNextVertex_time) 
{
  uintV n = g.n_;

  timer t_barrier1;
  timer t_barrier2;
  timer t_nextVertex;
  timer t_total;
  t_total.start();

  double total_getNextVertex_time = 0.0;

  for (int iter = 0; iter < max_iters; iter++) 
  {
    uintE local_edges_processed = 0;
    uintE local_vertices_processed = 0;

    while (true) {
      t_nextVertex.start();
      uintV u_start = getNextVertexToBeProcessed(granularity, n, next_vertex);
      getNextVertex_time += t_nextVertex.stop();

      if (u_start >= n) break;
      uintV u_end = std::min(u_start + granularity, n);
      for (uintV u = u_start; u < u_end; u++) {
        local_edges_processed += g.vertices_[u].getOutDegree();
        for (uintE i = 0; i < g.vertices_[u].getOutDegree(); i++) {
          uintV v = g.vertices_[u].getOutNeighbor(i);
          pr_next[v].store(pr_next[v].load() + pr_curr[u] / (PageRankType)g.vertices_[u].getOutDegree());
        }
      }
  
    }
    t_barrier1.start();
    barrier1.wait();
    time_taken_barrier1 += t_barrier1.stop();

    while (true) {
      t_nextVertex.start();
      uintV v_start = getNextVertexToBeProcessed(granularity, n, next_vertex);
      getNextVertex_time += t_nextVertex.stop();

      if (v_start >= n) break;
      uintV v_end = std::min(v_start + granularity, n);
      for (uintV v = v_start; v < v_end; v++) {
        local_vertices_processed++;
        PageRankType new_pagerank = PAGE_RANK(pr_next[v].load());
        pr_curr[v] = new_pagerank;
        pr_next[v].store(0.0);
      }
    }
    t_barrier2.start();
    barrier2.wait();
    time_taken_barrier2 += t_barrier2.stop();

    edges_processed = local_edges_processed;
    vertices_processed = local_vertices_processed;
  }
  total_time_taken = t_total.stop();
}

void pageRankParallelAtomic(Graph &g, int max_iters, int n_threads, int strategy, int granularity) 
{
  uintV n = g.n_;

  std::atomic<PageRankType> *pr_curr = new std::atomic<PageRankType>[n];
  std::atomic<PageRankType> *pr_next = new std::atomic<PageRankType>[n];

  for (uintV i = 0; i < n; i++) 
  {
    pr_curr[i] = INIT_PAGE_RANK;
    pr_next[i] = 0.0;
  }

  // Check for valid strategy and granularity
  if (strategy < 1 || strategy > 4) {
    std::cerr << "Invalid strategy: " << strategy << ". Strategy must be between 1 and 4." << std::endl;
    delete[] pr_curr;
    delete[] pr_next;
    return;
  }
  if (granularity <= 0 && strategy == 4) {
    std::cerr << "Invalid granularity: " << granularity << ". Granularity must be a positive integer for strategy 4." << std::endl;
    delete[] pr_curr;
    delete[] pr_next;
    return;
  }

  // Push based pagerank
  timer t1;
  double time_taken = 0.0;
  // Create threads and distribute the work across T threads
  // -------------------------------------------------------------------
  t1.start();

  std::atomic<int> next_vertex(0);
  double vertices_processed[n_threads]; 
  double edges_processed[n_threads]; 
  double getNextVertex_times[n_threads]; 

  CustomBarrier barrier1(n_threads);
  CustomBarrier barrier2(n_threads);

  vector<thread> threads;
  double thread_times_barrier1[n_threads];
  double thread_times_barrier2[n_threads];

  double thread_times[n_threads];
  
  int verticiesPerThread = (n + n_threads -1 ) / n_threads;
  
  if(strategy == 1) 
  {
   for (int i = 0; i < n_threads; ++i) 
    {
      uintV start = i * verticiesPerThread;
      uintV end = std::min((i + 1) * verticiesPerThread, (int)n);
      threads.emplace_back(pageRankHelperAtomic1, std::ref(g), max_iters, start, end, pr_curr, pr_next, std::ref(vertices_processed[i]), std::ref(edges_processed[i]), std::ref(barrier1), std::ref(barrier2), std::ref(thread_times[i]), std::ref(thread_times_barrier1[i]), std::ref(thread_times_barrier2[i]), std::ref(getNextVertex_times[i]));
    }
  } 
  else if(strategy == 2) 
  {
   for (int i = 0; i < n_threads; ++i) 
    {
      uintV start = i * verticiesPerThread;
      uintV end = std::min((i + 1) * verticiesPerThread, (int)n);
      threads.emplace_back(pageRankHelperAtomic2, std::ref(g), max_iters, n_threads, start, end, pr_curr, pr_next, std::ref(vertices_processed[i]), std::ref(edges_processed[i]), std::ref(barrier1), std::ref(barrier2), std::ref(thread_times[i]), std::ref(thread_times_barrier1[i]), std::ref(thread_times_barrier2[i]), std::ref(getNextVertex_times[i]));
    }
  } 
  else if(strategy == 3) 
  {
    for (int i = 0; i < n_threads; ++i) 
    {
    threads.emplace_back(pageRankHelperAtomic3, std::ref(g), max_iters, std::ref(next_vertex), std::ref(vertices_processed[i]), std::ref(edges_processed[i]), pr_curr, pr_next, std::ref(barrier1), std::ref(barrier2), std::ref(thread_times[i]), std::ref(thread_times_barrier1[i]), std::ref(thread_times_barrier2[i]), std::ref(getNextVertex_times[i]));
    }
  } 
  else if(strategy == 4) 
  {
    for (int i = 0; i < n_threads; ++i) 
    {
      threads.emplace_back(pageRankHelperAtomic4, std::ref(g), max_iters, n_threads, granularity, std::ref(next_vertex), std::ref(vertices_processed[i]), std::ref(edges_processed[i]), pr_curr, pr_next, std::ref(barrier1), std::ref(barrier2), std::ref(thread_times[i]), std::ref(thread_times_barrier1[i]), std::ref(thread_times_barrier2[i]), std::ref(getNextVertex_times[i]));
    }
  }
  
  // Join threads
  for (auto &t : threads) 
  {
    t.join();
  }

  time_taken = t1.stop();
  // -------------------------------------------------------------------
  
  std::cout << "thread_id, num_vertices, num_edges, barrier1_time, barrier2_time, getNextVertex_time, total_time" << endl;
  for (int i = 0; i < n_threads; i++) 
  {
    std::cout << i << ", "
      << vertices_processed[i] << ", "
      << edges_processed[i] << ", "
      << thread_times_barrier1[i] << ", "
      << thread_times_barrier2[i] << ", "
      << getNextVertex_times[i] << ", "
      << thread_times[i] << endl;
  }
  // Print the above statistics for each thread
  // Example output for 2 threads:
  // thread_id, time_taken
  // 0, 0.12
  // 1, 0.12

  PageRankType sum_of_page_ranks = 0;
  for (uintV u = 0; u < n; u++) {
    sum_of_page_ranks += pr_curr[u];
  }

  std::cout << "Sum of page ranks : " << sum_of_page_ranks << "\n";
  std::cout << "Time taken (in seconds) : " << time_taken << "\n";

  delete[] pr_curr;
  delete[] pr_next;
}

int main(int argc, char *argv[]) {
  cxxopts::Options options(
      "page_rank_push",
      "Calculate page_rank using serial and parallel execution");
  options.add_options(
      "",
      {
          {"nThreads", "Number of Threads",
           cxxopts::value<uint>()->default_value(DEFAULT_NUMBER_OF_THREADS)},
          {"nIterations", "Maximum number of iterations",
           cxxopts::value<uint>()->default_value(DEFAULT_MAX_ITER)},
          {"inputFile", "Input graph file path",
           cxxopts::value<std::string>()->default_value(
               "/scratch/input_graphs/roadNet-CA")},
          {"strategy", "Strategy for task decomposition (1-4)",
           cxxopts::value<int>()->default_value("1")},
          {"granularity", "Granularity for coarse-grained dynamic mapping",
           cxxopts::value<int>()->default_value("1000")}
      });

  auto cl_options = options.parse(argc, argv);
  uint n_threads = cl_options["nThreads"].as<uint>();
  uint max_iterations = cl_options["nIterations"].as<uint>();
  std::string input_file_path = cl_options["inputFile"].as<std::string>();
  int strategy = cl_options["strategy"].as<int>();
  int granularity = cl_options["granularity"].as<int>();

#ifdef USE_INT
  std::cout << "Using INT" << std::endl;
#else
  std::cout << "Using DOUBLE" << std::endl;
#endif
  std::cout << std::fixed;
  std::cout << "Number of Threads : " << n_threads << std::endl;
  std::cout << "Number of Iterations: " << max_iterations << std::endl;

  Graph g;
  std::cout << "Reading graph\n";
  g.readGraphFromBinary<int>(input_file_path);
  std::cout << "Created graph\n";
  pageRankParallelAtomic(g, max_iterations, n_threads, strategy, granularity);

  return 0;
}
