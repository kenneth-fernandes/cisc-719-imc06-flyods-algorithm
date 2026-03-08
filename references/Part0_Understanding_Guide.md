# Part 0 - Understanding Guide (Floyd's Algorithm + PCAM)

## What is Floyd-Warshall? (The Big Picture)

Imagine you have a map of cities connected by roads. Each road has a distance.
You want to find the **shortest path between EVERY pair of cities** - not just one pair, ALL of them.

That is the **All-Pairs Shortest Path (APSP)** problem, and Floyd-Warshall solves it.

### How does it work? (Simple Example)

Say you have 4 cities: A, B, C, D.

```
A ---3---> B
A ---7---> D
B ---2---> C
C ---1---> D
C ---5---> A
D ---2---> A
```

We store distances in a table (matrix) called D:

```
     A    B    C    D
A  [ 0    3    INF  7  ]
B  [ 8    0    2    INF]
C  [ 5    INF  0    1  ]
D  [ 2    INF  INF  0  ]
```

INF means "no direct road exists."

**The key idea:** Can we find a shorter path by going THROUGH another city?

The formula is:

```
D[i][j] = min( D[i][j],  D[i][k] + D[k][j] )
```

In plain English: "Is it shorter to go directly from i to j, or to go from i to k and then k to j?"

**Step-by-step (k = A, meaning "try going through city A"):**

- Can B reach C faster through A?  B->A + A->C = 8 + INF = INF. No improvement.
- Can B reach D faster through A?  B->A + A->D = 8 + 7 = 15. Current B->D = INF. YES! Update to 15.
- Can D reach B faster through A?  D->A + A->B = 2 + 3 = 5. Current D->B = INF. YES! Update to 5.

We repeat this for k = B, k = C, k = D. After all 4 rounds, every cell has the shortest path.

**That is Floyd-Warshall.** Three nested loops: k (which city to go through), i (start city), j (end city).

---

## What is PCAM? (Quinn Chapter 6)

PCAM is a **4-step method for designing parallel programs**. Think of it as a recipe:
"How do I take a big computation and split it across many workers (threads/processors)?"

### P - Partition (Break the work into tiny pieces)

**What it means:** Take your big problem and chop it into the smallest possible independent tasks.

**Real-life analogy:** You have 1000 envelopes to address. The simplest partition: each envelope is one task.
You don't think about WHO does them yet - you just identify what all the individual tasks are.

**Floyd-Warshall partition:**
Look at the matrix D. It has n x n cells. For each value of k, every cell (i,j) needs to be updated.
So the finest partition is: **each cell (i,j) update is one task.**

That means for a 4x4 matrix, you have 16 tasks per k-step.

**Where you see this in R1 (the CUDA C++ code):**
```c
// R1: floyd_step_kernel (line 65-82)
int j = blockIdx.x * blockDim.x + threadIdx.x;
int i = blockIdx.y * blockDim.y + threadIdx.y;
```
Each thread gets its own (i,j) coordinate. One thread = one cell = one task.
That IS the partition. Each thread computes one update: `D[i][j] = min(D[i][j], D[i][k] + D[k][j])`.

**Where you see this in R2 (the Python notebook):**
```python
# R2: floyd_step_kernel (cell 5)
i, j = cuda.grid(2)
if i < n and j < n:
    # one thread updates one (i,j) cell
```
Exact same idea. Each Numba CUDA thread handles one (i,j) cell.

The NumPy version in R2 does it differently - it updates ALL cells at once using broadcasting:
```python
# R2: floyd_warshall_numpy (cell 3)
D = np.minimum(D, Dik + Dkj)
```
Here, NumPy internally partitions the work across all cells, but you don't see individual threads.

---

### C - Communication (What data do tasks need to share?)

**What it means:** After you partition, tasks are NOT fully independent. Some tasks need data from other tasks.
Communication = identifying what data needs to flow between tasks.

**Real-life analogy:** You have 10 people building a house. The person installing windows NEEDS to wait for
the person building walls. That dependency is "communication" - wall-builder must TELL window-installer
"walls are done." They share information.

**Floyd-Warshall communication:**
Look at the formula again: `D[i][j] = min(D[i][j], D[i][k] + D[k][j])`

To update cell (i,j), the thread needs:
1. `D[i][k]` - a value from the SAME ROW as (i,j), but column k
2. `D[k][j]` - a value from ROW k, same column as (i,j)

So every thread in row i needs the value `D[i][k]` (one value from column k).
And every thread in column j needs the value `D[k][j]` (one value from row k).

**This means:** At each k-step, ROW k and COLUMN k must be "broadcast" to ALL threads.
This is the communication pattern - row k and column k are shared/read by everyone.

**Where you see this in R1:**
```c
// R1: lines 75-76
float Dik = D[ik];   // reading from column k (same row i)
float Dkj = D[kj];   // reading from row k (same column j)
```
Every thread reads from global GPU memory. If 1000 threads all read `D[k][5]`, that is 1000
reads of the same value. This is wasteful - this is where shared memory could help (but R1 doesn't use it).

**Where you see this in R2 (Numba kernel):**
```python
# R2: cell 5
Dik = D[i, k]    # same idea - read column k
Dkj = D[k, j]    # same idea - read row k
```

**Where you see this in R2 (NumPy version):**
```python
# R2: cell 3
Dik = D[:, k][:, None]     # ENTIRE column k, reshaped for broadcasting
Dkj = D[k, :][None, :]     # ENTIRE row k, reshaped for broadcasting
```
NumPy grabs the whole row and column at once, then broadcasts. The communication is the same
(everyone needs row k and column k), but NumPy handles it in one shot.

**Key insight about communication in Floyd-Warshall:**
The communication is IMPLICIT in R1 and R2. There are no explicit "send" or "receive" calls.
Instead, threads communicate by reading/writing to the SAME global memory array D.
The GPU memory hierarchy IS the communication channel.

---

### A - Agglomeration (Group tasks together for efficiency)

**What it means:** Having one task per cell sounds great in theory, but in practice it creates too much
overhead. Launching millions of tiny tasks is expensive. So we GROUP tasks together into bigger chunks.

**Real-life analogy:** Instead of assigning one envelope per person, you give each person a STACK of 50
envelopes. Less coordination overhead, and each person can work more efficiently (they don't have to
keep asking for the next envelope).

**Floyd-Warshall agglomeration:**
Instead of one thread per cell, you could:
- Give each thread a TILE (block) of cells, say 16x16 = 256 cells
- Load shared data (row k, column k for that tile) into fast shared memory ONCE
- Process all 256 cells using that cached data

This is called **tiling** or **blocking**. It reduces how many times you read from slow global memory.

**Where you see this in R1:**
```c
// R1: line 90
dim3 block(16, 16);
```
R1 groups threads into 16x16 blocks. BUT... it does NOT use shared memory within those blocks.
Each thread still reads D[i][k] and D[k][j] independently from global memory.
So the block is just an organizational unit for the GPU, NOT a true agglomeration for data reuse.

**What R1 is NOT doing (and could do better):**
A smarter approach would be:
1. Load the 16 values of row k that this block needs into shared memory (fast, on-chip memory)
2. Load the 16 values of column k that this block needs into shared memory
3. Now all 256 threads in the block read from shared memory instead of global memory
4. This turns 256 global memory reads into just 32 (16 for row + 16 for column)

R1 and R2 skip this optimization entirely. This is a major missed opportunity.

**Where you see this in R2:**
```python
# R2: cell 5
threads = (16, 16)
blocks = ((n + threads[0] - 1) // threads[0],
          (n + threads[1] - 1) // threads[1])
```
Same as R1 - threads are grouped into 16x16 blocks, but no shared memory tiling is done.

The NumPy version in R2 does a form of agglomeration automatically - NumPy operates on entire
arrays at once, which internally uses optimized memory access patterns.

---

### M - Mapping (Assign work to actual hardware)

**What it means:** Now that you have your grouped tasks, you decide WHICH processor/thread runs WHICH chunk.

**Real-life analogy:** You have 10 workers and 100 stacks of envelopes. Mapping = "Worker 1 gets stacks
1-10, Worker 2 gets stacks 11-20, ..." or maybe "Workers take the next available stack" (dynamic mapping).

**Floyd-Warshall mapping:**
We need to decide how to map the n x n matrix cells onto GPU threads.

**Where you see this in R1:**
```c
// R1: lines 66-67, 90-92
dim3 block(16, 16);                              // each block has 16x16 = 256 threads
dim3 grid((n + block.x - 1) / block.x,           // enough blocks to cover n columns
          (n + block.y - 1) / block.y);           // enough blocks to cover n rows
```

For n=512:
- Grid = (32, 32) = 1024 blocks
- Each block = (16, 16) = 256 threads
- Total = 1024 x 256 = 262,144 threads
- Matrix has 512 x 512 = 262,144 cells
- So exactly ONE thread per cell. Perfect 1-to-1 mapping.

Thread (tx, ty) in block (bx, by) handles cell:
- j = bx * 16 + tx  (column)
- i = by * 16 + ty  (row)

**Where you see this in R2:**
```python
# R2: cell 5
i, j = cuda.grid(2)   # Numba helper that computes the same thing as R1
```
`cuda.grid(2)` is shorthand for `(blockIdx * blockDim + threadIdx)` in both dimensions.
Same mapping as R1.

**CPU mapping (sequential):**
In both R1 and R2, the CPU version has no explicit mapping - it just uses one core and runs
three nested loops (k, i, j) sequentially. All work is mapped to a single processor.

```c
// R1: lines 49-60 (CPU version)
for (int k = 0; k < n; ++k)        // outer loop: which city to go through
    for (int i = 0; i < n; ++i)     // middle loop: start city
        for (int j = 0; j < n; ++j) // inner loop: end city
```

With OpenMP, you could add `#pragma omp parallel for` to map rows (i) across CPU cores.
That would be a different mapping: each core gets a chunk of rows.

---

## How the Outer k-Loop Works (Important Detail)

One thing that confuses many people: the k-loop is NOT parallelized.

```c
// R1: lines 94-96
for (int k = 0; k < n; ++k) {
    floyd_step_kernel<<<grid, block>>>(d_D, n, k);   // launch kernel for THIS k
}
```

For each k, the GPU launches a kernel that updates ALL (i,j) cells in parallel.
But k=1 must finish BEFORE k=2 starts, because the updates from k=1 feed into k=2.

This is a SEQUENTIAL dependency on k. You cannot parallelize this loop.
Both R1 and R2 handle this the same way - a host-side for-loop that launches one kernel per k.

---

## Timing and Correctness (How R1 and R2 Verify Results)

**R1 (C++ CUDA):**
```c
// R1: lines 140-161
// 1. Run CPU version, measure time
// 2. Run GPU version, measure time
// 3. Compare results cell-by-cell (max absolute difference)
// 4. Compute speedup = CPU time / GPU time
```

**R2 (Python notebook):**
```python
# R2: cell 7 - toy test on a known 4-node graph
# R2: cell 9 - timing on a random graph, same comparison
```

Both check: "Did the GPU produce the same answer as the CPU?" If max difference is near zero,
the GPU kernel is correct. Then they compare wall-clock times to see the speedup.

---

## Summary: PCAM in R1 and R2 at a Glance

| PCAM Step      | What R1/R2 Do                                                | What They Could Do Better                        |
|----------------|--------------------------------------------------------------|--------------------------------------------------|
| **Partition**  | One thread per (i,j) cell per k-step                        | Already fine-grained, good partition              |
| **Communication** | Each thread reads D[i,k] and D[k,j] from global memory  | Could cache row k and col k in shared memory      |
| **Agglomeration** | 16x16 thread blocks, but no shared-memory reuse          | Should tile and load shared data per block         |
| **Mapping**    | 2D grid of 16x16 blocks covers the full n x n matrix        | Could experiment with 32x32 or different layouts   |

---

## Sample Questions for Part 0 (You Need At Least 3)

1. **Grounded in R1/R2:** "In both R1 (line 90) and R2 (cell 5), the block size is set to 16x16.
   Why 16x16 and not 32x32 or 8x8? How does block size affect GPU occupancy (how many threads
   are active), shared memory usage, and overall performance?"

2. **Grounded in R1/R2:** "In R1, every thread reads D[i][k] and D[k][j] from global memory
   (lines 75-76). Since all threads in the same row share the same D[i][k] value, would loading
   column k into shared memory reduce redundant global memory traffic? How much speedup might
   this give?"

3. **Grounded in R1/R2:** "R1 includes the GPU memory transfer time (cudaMemcpy) in its total
   GPU timing (lines 88, 99). For small n, this transfer overhead might dominate the actual
   computation time. At what problem size does the kernel computation time start to outweigh
   the transfer cost?"

4. **PCAM general:** "In the Agglomeration step, how do you decide the optimal tile size?
   A bigger tile means more data reuse but needs more shared memory per block, which limits
   how many blocks can run simultaneously. How do you find the sweet spot?"

5. **PCAM general:** "Quinn describes Communication as explicit message passing (like MPI).
   But in R1 and R2, communication happens implicitly through shared global memory. Is implicit
   communication through memory always sufficient, or are there cases where explicit communication
   (like MPI sends/receives) would be necessary for Floyd-Warshall?"

---

## What Your Final Part 0 PDF Should Look Like

**Page 1:**
- Heading: "PCAM Methodology (Quinn Ch. 6)"
- 4-6 bullet points per PCAM step, in your own words
- Mention at least one example from Quinn Chapter 6

**Page 1-2:**
- Heading: "PCAM Patterns in R1 and R2"
- For each PCAM step, write 2-3 bullets explaining what R1/R2 do
- Label each bullet with [Partition], [Communication], [Agglomeration], or [Mapping]

**Page 2:**
- Heading: "Questions"
- At least 3 numbered questions
- At least one must reference R1 or R2 specifically (cite line numbers or cell numbers)
