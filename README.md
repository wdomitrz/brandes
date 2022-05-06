# Brandes algorithm implemented on GPU

## Source files

I provide source code in `src` directory. The files in that directory are:

* `sizes.hpp` -- defines of size of the blocks and grid (used to run the kernel) and the mdeg hyperparameter (used in implementations with vertex virtualization optimization).
* `errors.hpp` -- definition of a function (and a macro) used to handle cuda erros.
* `disable_cuda.hpp` -- a file with defines used in early stage of development to move the cuda code to a host code (useful for testing without an access to Nvidia GPU).
* `compact_graph_representation.hpp` -- an **important** file in which I create a
    * `Compact_graph_representation` -- a class that provides a *CRS representation* of a graph;
    * `Compacted_graph_representation` -- a class that compresses a graph (by iterativelly removing nodes of degree 0) and provides a *CRS representation* of such a compressed graph. It also can create betweeness centrality for all vertices in the original graph, when provided betweeness centrality for the compressed graph.
    * `Virtualized_graph_representation<Compact_graph_representation>` and `Virtualized_graph_representation<Compacted_graph_representation>` -- classes which transform a given graph to a graph with virtualized vertices -- to *Virtual-CSR representation* -- where each vertex has a degree of at most `mdeg`.
    * `Virtualized_graph_representation_with_stride<Compact_graph_representation>` and `Virtualized_graph_representation_with_stride<Compacted_graph_representation>` -- classes similar to the ones above, but the representation used here is *Stride-CSR representation*.
* Files used to run the programme and provide a daclaration of the most important computing function:
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
    * `brandes-par-vert.cu` -- kernel implementing vertex pararell version of the algorithm.
    * `brandes-par-edge.cu` -- kernel implementing edge pararell version of the algorithm.
    * `brandes-par-vert-queue.cu` -- kernel implementing vertex pararell version of the algorithm with queues (see *Unsuccessful optimizations* for description).
    * `brandes-par-vert-comp.cu` -- kernel implementing vertex pararell version of the algorithm on the compressed graph.
    * `brandes-par-edge-comp.cu` -- kernel implementing edge pararell version of the algorithm on the compressed graph.
    * `brandes-par-vert-queue-comp.cu` -- kernel implementing vertex pararell version of the algorithm with queues on the compressed graph.
    * `brandes-par-vert-virt.cu` -- kernel implementing vertex pararell version of the algorithm with *Virtual-CSR representation*.
    * `brandes-par-vert-virt-stride.cu` -- kernel implementing vertex pararell version of the algorithm with *Stride-CSR representation*.
    * `brandes-par-vert-comp-virt.cu` -- kernel implementing vertex pararell version of the algorithm with *Virtual-CSR representation* of the compressed graph.
    * `brandes-par-vert-comp-virt-stride.cu` -- kernel implementing vertex pararell version of the algorithm with *Stride-CSR representation* of the compressed graph. **This is the the fastest version.**
    * `brandes-par-vert-comp-virt-stride-const.cu` -- kernel implementing vertex pararell version of the algorithm with *Stride-CSR representation* of the compressed graph which uses `__constant__` memory (see *Unsuccessful optimizations* for description).

In the repository, there also is `scripts` directory, with some scripts that I used during implementation and testing.

## Algorithm

In my implementation I use both *fine* and *coarse-grained parallelism*.
The cuda kernel is executed exactly once, and in the kernel, the *fine-grained parallelism* appears inside each block, and the *coarse-graind parallelism* appreas between the blocks.

### Fine-grained parallelism
All threads in a block are used to compute `BFS` for the same vertex *s* at the same time. It is done, as described in the provided paper, by synchronizing the threads so to make them work on the same *layer* of the graph (given by the distance from the node *s*). When all the threads finish processing one layer, they go to the next *layer*. The synchronization is done using the cuda methods for synchronizing threads inside a block.

### Coarse-grained parallelism
All blocks, in pararell consider different starting vertices for the `BFS` algorithms, which is possible, because the significant size of the memory on the provided GPUs in comparison to the memory used for one `BFS` instance. These executions do not require synchronization (we only need atomic operations to collect the results -- or it could be done later in another kernel execution). This approach allows us to utilize both the memory as well as computing capabilities of the provided devices.

### Possible different approach and its drawbacks
A different possible approach would be to execute one kernel for each `s` and each distance `l` from `s` -- so to implement the algorithm *exaclty* as described in the paper.

#### Benefits of this approach
This approach would have a benefit of significantly lower memory usage. It also makes the synchronization between the blocks and threads implicit (which might be considered nice), by utilizing the synchronization comming from kernel execution.

This approach can also be extended by coarse-grained parallelism, but I did not test this solution.

#### Downsides of this approach
While this approach uses less memory, we have enough resources to utilize the coarse-grained parallelism, I decided to do so.

We should also note that the thread synchronization inside one block is faster than the blocks synchronization comming from the kernel execution, which poses a significant challenge for this implementation.

### Summary

The algorithm I implemented uses *coarse-grained parallelism* as well as *fine-grained parallelism*. For the graphs of the size that we consider I found this approach to be a good balance between speed and resource usage, so I have decided to stick with it.

## Optimizations

I implemented all the *algorithmic* implementations mentioned in the provided paper with the fastest appraoach being usage of *Stride-CSR representation* of a compressed graph. Before I proceed to describe other optimizations, let's think for a while where the speedup compes from.

### Comment on *Virtual* and *Stride-CSR representations*, and their relation to *edge-parallelism*

One should not that while the *Virtual/Stride-CSR representations* are parallel with respect to the *virtual* vertices, they are in fact an *edge* parallel implementation, where we ensure that one thread process a few edges with the same endvertex and the moves to the eges with different endvertex. This allows us to utilize the *strided* memory access, to optimize the memory acces (but that is an optimization that I already have in my *edge-parallel* version), but more importantly, it reduces the number of atomic operations needed to update values of the *delta* array (as we might first accumulate all the values, and then add them to *delta* at once). So the *Strided-CSR representation* is in fact *edge-parallelism* with cashed updates of *delta*.

### Usage of shared memory
I utilize the `__shared__` memory to keep variables that should be shared inside a block -- namely all the `sigma`, `d` and `delta` arrays, and, more importantly, `cont` and `l` variables. This allows me to update them once and keep the synchronized value across all the threads (which is most important in case of `cont` vaiable). It also frees up some registers and makes the algorithm utilze this ultra-fast memory.

### Usage of `__syncthreads()`
As I use the `fine-grained parallelism` inside each block, we have no need to wait until whole kernel terminates and start them again. We can easilly use `__syncthreads()` to synchronize and gain a significant speedup. It is different then what the provided paper proposes, as they suggest to run a kernel multiple times. In my inplementation we avoid the overhead that comes with it.

### Strided memory access
I should also note that the access to the *first endvertex* of each edge is strided and threads in one wrap will try to reac *vmap* array in the same memory area.

### Usage of steams
While it didn't give a significant speedup (as it is not the bootleneck of the slow part of the algorithm), I use streams to copy the memory to the GPU in parallel (which gave me a small speedup).

## Unsuccessful optimizations

## Possible, currently pointless, modifications

## Comparison and discussion
