**GPU Kernel Performance Benchmarks**

Optimized GPU compute kernels for AMD MI300X  
Focus: FP8 GEMM, MoE Inference, MLA Decode (KV cache + MHA)

---

**FP8 GEMM**

| Metric     | Description                                        |
|------------|----------------------------------------------------|
| Latency    | 120 µs                                             |
| Details    | FP8 blockwise matmul with MFMA                     |
|            | Double-buffered shared memory pipeline             |
|            | Vectorized tile access for peak throughput         |

---

**MoE Inference**

| Metric     | Description                                        |
|------------|----------------------------------------------------|
| Latency    | 8.75 ms                                            |
| Details    | Fused routing, matmul, activation                  |
|            | Expert-parallel batching                           |
|            | Shared workspace reuse                             |

---

**MLA Decode**

| Metric     | Description                                        |
|------------|----------------------------------------------------|
| Latency    | 2.68 ms                                            |
| Details    | KV cache + Multi-Head Attention                    |
|            | Fused FP8 projection                               |
|            | Vectorized access pattern, shared memory           |

---

**Platform**

| Field      | Value                                              |
|------------|----------------------------------------------------|
| Target GPU | AMD MI300X                                        |
| Framework  | HIP / ROCm                                        |
| Precision  | FP8 (E4M3)                                        |
