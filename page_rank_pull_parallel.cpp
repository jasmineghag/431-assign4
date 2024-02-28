#include "core/graph.h"
#include "core/utils.h"
#include <iomanip>
#include <iostream>
#include <stdlib.h>

#include <vector>
#include <thread>
#include <mutex>

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

void pageRankHelper1(Graph &g, PageRankType *pr_curr, PageRankType *pr_next, uintV start, uintV end, int max_iters, double &vertices_processed, double &edges_processed, CustomBarrier &barrier1, CustomBarrier &barrier2, double &total_time_taken, double &time_taken_barrier1, double &time_taken_barrier2, double &getNextVertex_time) 
{
  uintV n = g.n_;

  timer t_barrier1;
  timer t_barrier2;
  timer t1;
  t1.start();

  for (int iter = 0; iter < max_iters; iter++) 
  {
    // for each vertex 'v', process all its inNeighbors 'u'
    for (uintV v = start; v < end; v++) 
    {
      t_barrier1.start();
      uintE in_degree = g.vertices_[v].getInDegree();
      edges_processed += in_degree;  // Update edges_processed
      for (uintE i = 0; i < in_degree; i++) 
      {
        uintV u = g.vertices_[v].getInNeighbor(i);
        uintE u_out_degree = g.vertices_[u].getOutDegree();
        if (u_out_degree > 0) 
        {
          pr_next[v] += (pr_curr[u] / (PageRankType) u_out_degree);
        }
      }
      time_taken_barrier1 += t_barrier1.stop();
    }
    // Wait for all threads to complete computation
    barrier1.wait();

    // Update PageRanks using the computed values
    for (uintV v = start; v < end; v++) 
    {
      t_barrier2.start();
      {
        pr_next[v] = PAGE_RANK(pr_next[v]);

        // reset pr_curr for the next iteration
        pr_curr[v] = pr_next[v];
        pr_next[v] = 0.0;
      }
      time_taken_barrier2 += t_barrier2.stop();
    }
    // Wait for all threads to complete updating PageRanks
    barrier2.wait();
  }

  vertices_processed = end - start;   // Update vertices_processed
  total_time_taken = t1.stop();
  getNextVertex_time = 0.0;
}

void pageRankHelper2(Graph &g, PageRankType *pr_curr, PageRankType *pr_next, uintV start, uintV end, int max_iters, double &vertices_processed, double &edges_processed, CustomBarrier &barrier1, CustomBarrier &barrier2, double &total_time_taken, double &time_taken_barrier1, double &time_taken_barrier2, double &getNextVertex_time) 
{
  uintV n = g.n_;

  timer t_barrier1;
  timer t_barrier2;
  timer t1;
  t1.start();

  for (int iter = 0; iter < max_iters; iter++) 
  {
    // for each vertex 'v', process all its inNeighbors 'u'
    for (uintV v = start; v < end; v++) 
    {
      t_barrier1.start();
      uintE in_degree = g.vertices_[v].getInDegree();
      edges_processed += in_degree;  // Update edges_processed

      for (uintE i = 0; i < in_degree; i++) 
      {
        uintV u = g.vertices_[v].getInNeighbor(i);
        uintE u_out_degree = g.vertices_[u].getOutDegree();
        if (u_out_degree > 0) 
        {
          pr_next[v] += (pr_curr[u] / (PageRankType) u_out_degree);
        }
      }
      time_taken_barrier1 += t_barrier1.stop();
    }
    // Wait for all threads to complete computation
    barrier1.wait();

    // Update PageRanks using the computed values
    for (uintV v = start; v < end; v++) 
    {
      t_barrier2.start();
      {
        pr_next[v] = PAGE_RANK(pr_next[v]);

        // reset pr_curr for the next iteration
        pr_curr[v] = pr_next[v];
        pr_next[v] = 0.0;
      }
      time_taken_barrier2 += t_barrier2.stop();
    }
    // Wait for all threads to complete updating PageRanks
    barrier2.wait();
  }

  vertices_processed = end - start;   // Update vertices_processed 
  total_time_taken = t1.stop();
  getNextVertex_time = 0.0;
}

std::mutex vertex_mutex;
uintV next_vertex_global = 0;

uintV getNextVertexToBeProcessed(int granularity, uintV n) {
  std::lock_guard<std::mutex> lock(vertex_mutex);
  uintV next_vertex = next_vertex_global;
  if (next_vertex >= n) {
    return -1;
  }
  next_vertex_global = std::min(next_vertex + granularity, n);
  return next_vertex;
}

uintE outDegree(Graph &g, uintV u) {
  return g.vertices_[u].getOutDegree();
}

void pageRankHelper3(Graph &g, PageRankType *pr_curr, PageRankType *pr_next, int max_iters, double &vertices_processed, double &edges_processed, CustomBarrier &barrier1, CustomBarrier &barrier2, double &total_time_taken, double &time_taken_barrier1, double &time_taken_barrier2, double &getNextVertex_time) 
{
  uintV n = g.n_;
  timer t_barrier1;
  timer t_barrier2;
  timer t_getNextVertex;
  timer t_total;
  t_total.start();

  for (int iter = 0; iter < max_iters; iter++) 
  {
    while (true) {
      t_getNextVertex.start();
      uintV v = getNextVertexToBeProcessed(1, n);
      getNextVertex_time += t_getNextVertex.stop();

      if (v == -1) break;
      vertices_processed++;

      t_barrier1.start();
      uintE in_degree = g.vertices_[v].getInDegree();
      edges_processed += in_degree;
      for (uintE i = 0; i < in_degree; i++) {
        uintV u = g.vertices_[v].getInNeighbor(i);
        uintE u_out_degree = g.vertices_[u].getOutDegree();
        if (u_out_degree > 0) {
          pr_next[v] += (pr_curr[u] / (PageRankType) u_out_degree);
        }
      }
      time_taken_barrier1 += t_barrier1.stop();
    }
    barrier1.wait();

    while (true) {
      t_getNextVertex.start();
      uintV v = getNextVertexToBeProcessed(1, n);
      getNextVertex_time += t_getNextVertex.stop();

      if (v == -1) break;

      t_barrier2.start();
      pr_next[v] = PAGE_RANK(pr_next[v]);
      pr_curr[v] = pr_next[v];
      pr_next[v] = 0.0;
      time_taken_barrier2 += t_barrier2.stop();
    }
    barrier2.wait();
  }
  total_time_taken = t_total.stop();
}

void pageRankHelper4(Graph &g, PageRankType *pr_curr, PageRankType *pr_next, int max_iters, int granularity, double &vertices_processed, double &edges_processed, CustomBarrier &barrier1, CustomBarrier &barrier2, double &total_time_taken, double &time_taken_barrier1, double &time_taken_barrier2, double &getNextVertex_time) 
{
  uintV n = g.n_;

  timer t_barrier1;
  timer t_barrier2;
  timer t_getNextVertex;

  timer t_total;
  t_total.start();

  for (int iter = 0; iter < max_iters; iter++) 
  {
    while (true) {
      t_getNextVertex.start();
      uintV v_start = getNextVertexToBeProcessed(granularity, n);
      getNextVertex_time += t_getNextVertex.stop();

      if (v_start == -1) break;

      for (uintV v = v_start; v < std::min(v_start + granularity, n); v++) {
        vertices_processed++;

        t_barrier1.start();
        uintE in_degree = g.vertices_[v].getInDegree();
        edges_processed += in_degree;
        for (uintE i = 0; i < in_degree; i++) {
          uintV u = g.vertices_[v].getInNeighbor(i);
          uintE u_out_degree = g.vertices_[u].getOutDegree();
          if (u_out_degree > 0) {
            pr_next[v] += (pr_curr[u] / (PageRankType) u_out_degree);
          }
        }
        time_taken_barrier1 += t_barrier1.stop();
      }
    }
    barrier1.wait();

    while (true) {
      t_getNextVertex.start();
      uintV v_start = getNextVertexToBeProcessed(granularity, n);
      getNextVertex_time += t_getNextVertex.stop();

      if (v_start == -1) break;

      for (uintV v = v_start; v < std::min(v_start + granularity, n); v++) {
        t_barrier2.start();
        pr_next[v] = PAGE_RANK(pr_next[v]);
        pr_curr[v] = pr_next[v];
        pr_next[v] = 0.0;
        time_taken_barrier2 += t_barrier2.stop();
      }
    }
    barrier2.wait();
  }
  total_time_taken = t_total.stop();
}

void pageRankParallel(Graph &g, int max_iters, int n_threads, int strategy, int granularity) 
{
  uintV n = g.n_;

  PageRankType *pr_curr = new PageRankType[n];
  PageRankType *pr_next = new PageRankType[n];

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

  // Pull based pagerank
  timer t1;
  double time_taken = 0.0;
  // Create threads and distribute the work across T threads
  // -------------------------------------------------------------------
  t1.start();
  
  // code snippet moved to helper fn 
  // Create threads and distribute the work across T threads
  thread threads[n_threads];
  double thread_times[n_threads];

  double thread_times_barrier1[n_threads];
  double thread_times_barrier2[n_threads];
  double vertices_processed[n_threads]; //TODO
  double edges_processed[n_threads]; //TODO
  double getNextVertex_times[n_threads]; //TODO

  CustomBarrier barrier(n_threads);
  CustomBarrier barrier1(n_threads);
  CustomBarrier barrier2(n_threads);

  int vertex_per_thread = n / n_threads;
  int edge_per_thread = n / n_threads;
  

  if(strategy == 1) 
  {
    for (int i = 0; i < n_threads; ++i) 
    {
      uintV start = (vertex_per_thread) * i;
      uintV end = (i == n_threads - 1) ? n : (vertex_per_thread) * (i + 1);
      threads[i] = thread(pageRankHelper1, ref(g), pr_curr, pr_next, start, end, max_iters, std::ref(vertices_processed[i]), std::ref(edges_processed[i]),std::ref(barrier1),std::ref(barrier2), std::ref(thread_times[i]), std::ref(thread_times_barrier1[i]), std::ref(thread_times_barrier2[i]), std::ref(getNextVertex_times[i]));
    }
  } 
  else if(strategy == 2) 
  {
    // Edge-based decomposition
    uintV start = 0;
    for (int i = 0; i < n_threads; ++i) 
    {
      uintV edges_covered = 0;
      while (start < n && edges_covered < (i + 1) * g.m_ / n_threads) 
      {
        edges_covered += g.vertices_[start].getInDegree();
        start++;
      }
      uintV end = start;
      threads[i] = thread(pageRankHelper2, ref(g), pr_curr, pr_next, start, end, max_iters, std::ref(vertices_processed[i]), std::ref(edges_processed[i]), std::ref(barrier1), std::ref(barrier2), std::ref(thread_times[i]), std::ref(thread_times_barrier1[i]), std::ref(thread_times_barrier2[i]), std::ref(getNextVertex_times[i]));
      start = end;
    }
  } 
  else if(strategy == 3) 
  {
    //vertex-based decomposition with dynamic mapping
    for (int i = 0; i < n_threads; ++i) 
    {
      threads[i] = thread(pageRankHelper3, ref(g), pr_curr, pr_next, max_iters, std::ref(vertices_processed[i]), std::ref(edges_processed[i]), std::ref(barrier1), std::ref(barrier2), std::ref(thread_times[i]), std::ref(thread_times_barrier1[i]), std::ref(thread_times_barrier2[i]), std::ref(getNextVertex_times[i]));
    }
  } 
  else if(strategy == 4) 
  {
    // vertex-based decomposition with coarse-grained dynamic mapping
    
    for (int i = 0; i < n_threads; ++i) 
    {
    threads[i] = thread(pageRankHelper4, ref(g), pr_curr, pr_next, max_iters, granularity, std::ref(vertices_processed[i]), std::ref(edges_processed[i]), std::ref(barrier1), std::ref(barrier2), std::ref(thread_times[i]), std::ref(thread_times_barrier1[i]), std::ref(thread_times_barrier2[i]), std::ref(getNextVertex_times[i]));
    }
  }

  // Join threads
  for (auto &t : threads) 
  {
    t.join();
  }

  time_taken = t1.stop();
  // -------------------------------------------------------------------
  std::cout << "Using DOUBLE\n";
  std::cout << "Number of Threads : " << n_threads << endl;
  std::cout << "Strategy : " << strategy << endl;
  std::cout << "Granularity : " << granularity << endl;
  std::cout << "Iterations : " << max_iters << endl;

  std::cout << "Reading graph" << endl;
  std::cout << "Created graph" << endl;

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
  for (uintV u = 0; u < n; u++) 
  {
    sum_of_page_ranks += pr_curr[u];
  }
  std::cout << "Sum of page ranks : " << sum_of_page_ranks << "\n";
  std::cout << "Time taken (in seconds) : " << time_taken << "\n";
  delete[] pr_curr;
  delete[] pr_next;
}

int main(int argc, char *argv[]) 
{
  cxxopts::Options options(
      "page_rank_pull",
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
  std::cout << "Using INT\n";
#else
  std::cout << "Using DOUBLE\n";
#endif
  std::cout << std::fixed;
  std::cout << "Number of Threads : " << n_threads << std::endl;
  std::cout << "Number of Iterations: " << max_iterations << std::endl;

  Graph g;
  std::cout << "Reading graph\n";
  g.readGraphFromBinary<int>(input_file_path);
  std::cout << "Created graph\n";
  pageRankParallel(g, max_iterations, n_threads, strategy, granularity);

  return 0;
}
