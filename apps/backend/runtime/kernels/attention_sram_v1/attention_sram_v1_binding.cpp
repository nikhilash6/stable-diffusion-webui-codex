// Generic SRAM/shared-memory attention C++/CUDA extension bindings.
// Build with: python setup.py build_ext --inplace (requires CUDA toolchain).

#include <torch/extension.h>

namespace {
constexpr int kAttentionSramV1AbiVersion = 1;
}

torch::Tensor attention_sram_v1_attn_fwd_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    bool is_causal);

torch::Tensor attention_sram_v1_rope_blhd_inplace_cuda(
    torch::Tensor x_blhd,
    const torch::Tensor& rope_cos,
    const torch::Tensor& rope_sin);

TORCH_LIBRARY(attention_sram_v1, m) {
  m.def("attn_fwd(Tensor q, Tensor k, Tensor v, bool is_causal=False) -> Tensor");
  m.def("rope_blhd_(Tensor x_blhd, Tensor rope_cos, Tensor rope_sin) -> Tensor");
}

TORCH_LIBRARY_IMPL(attention_sram_v1, CPU, m) {
  m.impl(
      "attn_fwd",
      [](const torch::Tensor& /*q*/,
         const torch::Tensor& /*k*/,
         const torch::Tensor& /*v*/,
         bool /*is_causal*/) -> torch::Tensor {
        TORCH_CHECK(
            false,
            "attention_sram_v1.attn_fwd: CPU implementation not available. Build CUDA kernels and run on CUDA tensors.");
      });

  m.impl(
      "rope_blhd_",
      [](torch::Tensor /*x_blhd*/, const torch::Tensor& /*rope_cos*/, const torch::Tensor& /*rope_sin*/) -> torch::Tensor {
        TORCH_CHECK(
            false,
            "attention_sram_v1.rope_blhd_: CPU implementation not available. Build CUDA kernels and run on CUDA tensors.");
      });
}

TORCH_LIBRARY_IMPL(attention_sram_v1, CUDA, m) {
  m.impl("attn_fwd", attention_sram_v1_attn_fwd_cuda);
  m.impl("rope_blhd_", attention_sram_v1_rope_blhd_inplace_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("ATTENTION_SRAM_V1_ABI") = kAttentionSramV1AbiVersion;
}
