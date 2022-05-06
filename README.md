---
title: Brandes algorithm implemented on GPU
author: Witalis Domitrz
date: today
---

## Comment on hiperparameters

The hiperparameters in file `sizes.hpp` were chosen to work well with graphs of size of the one from `loc-gowalla_edges.txt` file (or up to 4.5 times larger). If one wants to test the programme on greater graphs, decreasing `BLOCKS` constant in `sizes.hpp` is recommended - though it might affect the performance. I have chosen this parameter, as it allows us to use the programme on graphs on which the programme terminates in under 5 minutes (as specified in the task description).

This variable controls the *coarse-grained parallelism*, which directly impacts the memory requirement for the programme.

## Source files

I provide source code in `src` directory. The files in that directory are:

* `sizes.hpp` -- defines of size of the blocks and grid (used to run the kernel) and the *mdeg* hyperparameter (used in implementations with vertex virtualization optimization).
* `errors.hpp` -- definition of a function (and a macro) used to handle cuda errors.
* `disable_cuda.hpp` -- a file with defines used in early stage of development to move the cuda code to a host code (useful for testing without an access to Nvidia GPU).
* `compact_graph_representation.hpp` -- an **important** file in which I create a
    * `Compact_graph_representation` -- a class that provides a *CRS representation* of a graph;
    * `Compacted_graph_representation` -- a class that compresses a graph (by iteratively removing nodes of degree 0) and provides a *CRS representation* of such a compressed graph. It also can create betweens centrality for all vertices in the original graph, when provided betweens centrality for the compressed graph.
    * `Virtualized_graph_representation<Compact_graph_representation>` and `Virtualized_graph_representation<Compacted_graph_representation>` -- classes which transform a given graph to a graph with virtualized vertices -- to *Virtual-CSR representation* -- where each vertex has a degree of at most `mdeg`.
    * `Virtualized_graph_representation_with_stride<Compact_graph_representation>` and `Virtualized_graph_representation_with_stride<Compacted_graph_representation>` -- classes similar to the ones above, but the representation used here is *Stride-CSR representation*.
* Files used to run the programme and provide a declaration of the most important computing function:
    * `brandes-seq*.cpp` -- files with sequential implementations of the Brandes algorithm
        * `brandes-seq.cpp` -- first implementation, using `stl` implementation of `queue`, `stack` and `vector`.
        * `brandes-seq-vector.cpp` -- implementation in which we get rid of the `queue` and `stack` by replacing them both with `vector`s.
        * `brandes-seq-array.cpp` -- last sequential implementation, without using `stl` at all. It uses C style `arrays`.
    * `brandes.cpp` -- a file used to execute the sequential implementations of the Brandes algorithms.
    * `brandes-old.hpp` -- a file which provides the declaration of the function used for both the sequential implementations and the most basic versions of the cuda implementations.
    * `brandes-old.cu` -- a file used to execute the basic cuda implementations -- `brandes-par-{vert,edge,vert-queue}`.
    * `brandes.{cu,hpp}` -- files used to execute, and provide an interface for, the basic cuda implementations using just compacted graph representation -- `brandes-par-{vert,edge,vert-queue}-comp`.
    * `brandes.{cu,hpp}` -- files used to execute, and provide an interface for, the cuda implementations using just compacted graph representation -- `brandes-par-{vert,edge,vert-queue}-comp`.
    * `brandes-virt-nocomp.{cu,hpp}` -- files used to execute, and provide an interface for, the cuda implementation using only *Virtual-CSR representation* -- `brandes-par-vert-virt`.
    * `brandes-virt.{cu,hpp}` -- files used to execute, and provide an interface for, the cuda implementation using BOTH compacted graph representation and *Virtual-CSR representation* -- `brandes-par-vert-comp-virt`.
    * `brandes-virt-stride-nocomp.{cu,hpp}` -- files used to execute, and provide an interface for, the cuda implementation using only *Stride-CSR representation* -- `brandes-par-vert-virt-stride`.
    * `brandes-virt-stride.{cu,hpp}` -- files used to execute, and provide an interface for, the cuda implementation using BOTH compacted graph representation and *Stride-CSR representation* -- `brandes-par-vert-comp-virt-stride*`. **These are the files used in the fastest version.**
* `brandes-par*.cu` -- filest with implementations of the kernels.
    * `brandes-par-vert.cu` -- kernel implementing vertex parallel version of the algorithm.
    * `brandes-par-edge.cu` -- kernel implementing edge parallel version of the algorithm.
    * `brandes-par-vert-queue.cu` -- kernel implementing vertex parallel version of the algorithm with queues (see *Unsuccessful optimizations* for description).
    * `brandes-par-vert-comp.cu` -- kernel implementing vertex parallel version of the algorithm on the compressed graph.
    * `brandes-par-edge-comp.cu` -- kernel implementing edge parallel version of the algorithm on the compressed graph.
    * `brandes-par-vert-queue-comp.cu` -- kernel implementing vertex parallel version of the algorithm with queues on the compressed graph.
    * `brandes-par-vert-virt.cu` -- kernel implementing vertex parallel version of the algorithm with *Virtual-CSR representation*.
    * `brandes-par-vert-virt-stride.cu` -- kernel implementing vertex parallel version of the algorithm with *Stride-CSR representation*.
    * `brandes-par-vert-comp-virt.cu` -- kernel implementing vertex parallel version of the algorithm with *Virtual-CSR representation* of the compressed graph.
    * `brandes-par-vert-comp-virt-stride.cu` -- kernel implementing vertex parallel version of the algorithm with *Stride-CSR representation* of the compressed graph. **This is the the fastest version.**
    * `brandes-par-vert-comp-virt-stride-const.cu` -- kernel implementing vertex parallel version of the algorithm with *Stride-CSR representation* of the compressed graph which uses `__constant__` memory (see *Unsuccessful optimizations* for description).

In the repository, there also is `scripts` directory, with some scripts that I used during implementation and testing.

## Algorithm

In my implementation I use both *fine* and *coarse-grained parallelism*.
The cuda kernel is executed exactly once, and in the kernel, the *fine-grained parallelism* appears inside each block, and the *coarse-grained parallelism* appears between the blocks.

### Fine-grained parallelism
All threads in a block are used to compute `BFS` for the same vertex *s* at the same time. It is done, as described in the provided paper, by synchronizing the threads so to make them work on the same *layer* of the graph (given by the distance from the node *s*). When all the threads finish processing one layer, they go to the next *layer*. The synchronization is done using the cuda methods for synchronizing threads inside a block.

### Coarse-grained parallelism
All blocks, in parallel consider different starting vertices for the `BFS` algorithms, which is possible, because the significant size of the memory on the provided GPUs in comparison to the memory used for one `BFS` instance. These executions do not require synchronization (we only need atomic operations to collect the results -- or it could be done later in another kernel execution). This approach allows us to utilize both the memory as well as computing capabilities of the provided devices.

### Possible different approach and its drawbacks
A different possible approach would be to execute one kernel for each `s` and each distance `l` from `s` -- so to implement the algorithm *exactly* as described in the paper.

#### Benefits of this approach
This approach would have a benefit of significantly lower memory usage. It also makes the synchronization between the blocks and threads implicit (which might be considered nice), by utilizing the synchronization coming from kernel execution.

This approach can also be extended by coarse-grained parallelism, but I did not test this solution.

#### Downsides of this approach
While this approach uses less memory, we have enough resources to utilize the coarse-grained parallelism, I decided to do so.

We should also note that the thread synchronization inside one block is faster than the blocks synchronization coming from the kernel execution, which poses a significant challenge for this implementation.

### Summary

The algorithm I implemented uses *coarse-grained parallelism* as well as *fine-grained parallelism*. For the graphs of the size that we consider I found this approach to be a good balance between speed and resource usage, so I have decided to stick with it.

## Optimizations

I implemented all the *algorithmic* implementations mentioned in the provided paper with the fastest approach being usage of *Stride-CSR representation* of a compressed graph. Before I proceed to describe other optimizations, let's think for a while where the speedup comes from.

### Comment on *Virtual* and *Stride-CSR representations*, and their relation to *edge-parallelism*

One should not that while the *Virtual/Stride-CSR representations* are parallel with respect to the *virtual* vertices, they are in fact an *edge* parallel implementation, where we ensure that one thread process a few edges with the same endvertex and the moves to the edges with different endvertex. This allows us to utilize the *strided* memory access, to optimize the memory access (but that is an optimization that I already have in my *edge-parallel* version), but more importantly, it reduces the number of atomic operations needed to update values of the *delta* array (as we might first accumulate all the values, and then add them to *delta* at once). So the *Strided-CSR representation* is in fact *edge-parallelism* with cashed updates of *delta*.

### Usage of shared memory
I utilize the `__shared__` memory to keep variables that should be shared inside a block -- namely all the `sigma`, `d` and `delta` arrays, and, more importantly, `cont` and `l` variables. This allows me to update them once and keep the synchronized value across all the threads (which is most important in case of `cont` variable). It also frees up some registers and makes the algorithm utilize this ultra-fast memory.

### Usage of `__syncthreads()`
As I use the `fine-grained parallelism` inside each block, we have no need to wait until whole kernel terminates and start them again. We can easily use `__syncthreads()` to synchronize and gain a significant speedup. It is different then what the provided paper proposes, as they suggest to run a kernel multiple times. In my implementation we avoid the overhead that comes with it.

### Strided memory access
I should also note that the access to the *first endvertex* of each edge is strided and threads in one wrap will try to reach *vmap* array in the same memory area.

One might see artifacts of other tested version of memory access in `brandes-par-vert-comp-virt-stride.cu` file, with commented-out `// const uint32_t big_step` line (and 3 following commented lines).

### Usage of steams
While it didn't give a significant speedup (as it is not the bottleneck of the slow part of the algorithm), I use streams to copy the memory to the GPU in parallel (which gave me a small speedup).

### Bulk allocation of memory
When I allocate a significant amount of memory (for *coarse-grained parallelism*), I do in one allocation, in a block. A slower version of it would be to allocate an array of pointers, and for each such pointer allocate an array (so basically create a 2d array by allocating each 2d array separately) -- that would be slower. We also access the block of memory in a way that makes the threads of the same warp to use the memory which is at the same address (one might consider different indexing -- instead of `A[x][y] = A_bulk[Y_SIZE * x + y]`, `A[x][y] = A_bulk[x + X_SIZE * y]` -- that would also work, but it would be slower).

## A reasonable things to do, but that is not what the problem is about

### Running on GPU only when it is reasonable
We should point out that running the kernel is pointless for small graphs, so, if we were to create a reasonable programme to handle both small graphs and graphs of significant size, we should add a size threshold for which the graphs would be processed on CPU only. That would mean that while running the programme on small graph, we would use CPU implementation and do not test the GPU one. I believe it is not the point of this problem at all.

### Further graph reductions
One might consider move methods of reducing size of the considered graphs, for example we should notice that if some vertex has a degree 2, and their both neighbors are connected, then no shortest path will go though this vertex. We could remove all such vertices. This wouldn't improve the performance significantly, but we can combine this method to compactify the graph with removing the vertices of degree 1, and as a result, get a meaningful reduction of the graph.

## Unsuccessful optimizations

### Queues
One of the first optimizations coming to mind when migrating from the sequential version of an algorithm to a parallel one, is to use some king of queue implementation to store information which vertices should be accessed in the next *layer of the graph*. One can do it by keeping two arrays -- for the current considered layer and the next considered layer. Unfortunately adding a vertex to such a queue exactly once requires usage of atomic operations and as a result slowing the code down. This is by no means a desired optimization (at least in the way I implemented it).

### Constant memory
This optimization was Unsuccessful, because we can keep the small number of two constant, globally accessible variables in registers and access them at will. If our graph had more constant parameters, it would be reasonable approach. In my implementation it did not speed up the code, and it made it even slightly slower.

### Reasonable balance of CPU and GPU computing
For that optimization see the next section, as I just did not implement the slower variant at all, so I describe it in different section.

## Possible, currently pointless, modifications
Here I discuss some modifications of the code that we would not benefit from, but if having faster GPUs and slower CPUs with fast memory transfer between them, or at least having a grapl of reasonable size already in the memory of GPU, might be a reasonable approach to think of.

### Computing prefix sums on GPU
In the implementation of `Compact_graph_representation` class, we obviously need to compute prefix sums. This takes linear time, and could be implemented faster on GPU (as we have already seen during the classes). However, in our case, it would not be an optimization at all because of various reasons

* we still need to process the graph in linear time on CPU,
* this is not at all the slowest operation that we do,
* it is linear in terms of the number of vertices (which is most likely smaller than the number of edges),
* a single CPU core is significantly more powerful than a singe GPU core,
* we are required to measure the time of the kernel, so computing as much as possible on CPU is free.

### Removing vertices of degree 1 on GPU
While in the provided paper it is proposed to remove the vertices of degree 1 on GPU, it is not a reasonable thing to do because of various reasons (some of which intersect with the previous non-optimization)

* while it would not be a significant problem for the considered graphs, to reduce the size of the graph on GPU, we need to transfer the whole original graph to GPU, and if we do it on CPU, we can just transfer the reduced graph. For sparse, tree-like graphs it might be a significant benefit.
* we still need to process the graph in linear time on CPU,
* this is not at all the slowest operation that we do,
* it is linear in terms of maximum of the number of edges and the number of vertices, so it is still fast,
* a single CPU core is significantly more powerful than a singe GPU core,
* we are required to measure the time of the kernel, so computing as much as possible on CPU is free.

## Other comments

### Accuracy of the virtualized version
As in the virtualized version we accumulate the values first to `sum` variable, and then add them to `delta`, in comparison to non-virtualized versions, when we always add to delta, the results might differ due to limited accuracy of the floating point variables.
