#include <cstdint>

void brandes(const uint32_t n, const uint32_t virt_n,
             const uint32_t starting_positions[],
             const uint32_t compact_graph[], const uint32_t vmap[],
             const uint32_t vptrs[], const uint32_t jmp[], double CB[]);
