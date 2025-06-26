GPU Kernel Performance Benchmarks

Optimized GPU compute kernels for AMD MI300X.
Focus: FP8 GEMM, MoE Inference, MLA Decode (KV cache + MHA).

---

[ FP8 GEMM ]
Latency: 120 µs
- FP8 blockwise matmul with MFMA
- Double-buffered shared memory pipeline
- Vectorized tile access for peak throughput

---

[ MoE Inference ]
Latency: 8.75 ms
- Fused routing, matmul, activation
- Expert-parallel batching
- Shared workspace reuse

---

[ MLA Decode ]
Latency: 2.68 ms
- KV cache + Multi-Head Attention
- Fused FP8 projection
- Vectorized access pattern, shared memory

---

[ Platform ]
Target GPU: AMD MI300X
Framework: HIP / ROCm
Precision: FP8 (E4M3 / E5M2)
