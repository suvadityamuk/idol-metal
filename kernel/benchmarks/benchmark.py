import math

import torch
import torch.nn.functional as F

from kernels.benchmark import Benchmark

H, D = 16, 64


class IdolMetalBenchmark(Benchmark):
    """Shapes of the Idol denoiser's conditional pass: 12,294 tokens (8,192 latent + 4,101 image + 1 camera),
    1,024 channels, 16 heads of 64, MLP ratio 4."""

    seed = 0
    rows, channels = 12294, 1024

    def setup_resid_ln_mod(self):
        R, C = self.rows, self.channels
        self.x = torch.randn(R, C, device=self.device).half()
        self.h = torch.randn(R, C, device=self.device).half()
        self.gate = (0.1 * torch.randn(C, device=self.device)).half()
        self.s1p = (1 + 0.1 * torch.randn(C, device=self.device)).half()
        self.shift = (0.1 * torch.randn(C, device=self.device)).half()
        self.rw = 1 + 0.1 * torch.randn(C, device=self.device)
        self.rb = 0.1 * torch.randn(C, device=self.device)

    def benchmark_resid_ln_mod(self):
        _, self.out, _ = self.kernel.resid_ln_mod(self.x, self.s1p, self.shift, h=self.h, gate=self.gate,
                                                  repo_weight=self.rw, repo_bias=self.rb)

    def verify_resid_ln_mod(self) -> torch.Tensor:
        x = self.x + self.h * self.gate
        y = F.layer_norm(x.float(), (self.channels,), eps=1e-6).half() * self.s1p
        return y + self.shift

    def setup_repo_rope_qk(self):
        R = self.rows
        self.qkv = torch.randn(R, 3, H, D, device=self.device).half()
        self.delta = 0.5 * torch.randn(R, H * 3, device=self.device)
        self.freqs = [torch.linspace(1.0, 16.0, n, device=self.device) for n in (10, 10, 12)]
        self.ft = torch.cat([f.tanh() for f in self.freqs])
        self.fd = torch.cat([f - f.tanh() for f in self.freqs])
        self.axis = torch.tensor([0] * 10 + [1] * 10 + [2] * 12, dtype=torch.int32, device=self.device)
        self.qg = torch.ones(H, D, device=self.device)

    def benchmark_repo_rope_qk(self):
        self.out, _ = self.kernel.repo_rope_qk(self.qkv, self.delta, self.ft, self.fd, self.axis, self.qg, self.qg,
                                               1.0)

    def verify_repo_rope_qk(self) -> torch.Tensor:
        R = self.rows
        d = self.delta.view(R, H, 3)
        ang = torch.cat([(d[..., a].unsqueeze(-1) * f.tanh() + d[..., a].unsqueeze(-1) * (f - f.tanh())) * math.pi
                         for a, f in enumerate(self.freqs)], -1)
        fr = torch.polar(torch.ones_like(ang), ang)
        x = torch.view_as_complex(self.qkv[:, 0].float().reshape(R, H, D // 2, 2)) * fr
        x = torch.view_as_real(x).reshape(R, H, D).half()
        return (F.normalize(x.float(), dim=-1) * self.qg).half()

    def setup_linear_gelu_fc1(self):
        R, C = self.rows, self.channels
        self.x = torch.randn(R, C, device=self.device).half()
        self.w = (torch.randn(4 * C, C, device=self.device) / math.sqrt(C)).half()
        self.b = (0.1 * torch.randn(4 * C, device=self.device)).half()

    def benchmark_linear_gelu_fc1(self):
        self.out = self.kernel.linear_gelu(self.x, self.w, self.b)

    def verify_linear_gelu_fc1(self) -> torch.Tensor:
        return F.gelu(F.linear(self.x, self.w, self.b), approximate="tanh")

    def setup_rmsnorm_heads(self):
        self.src = torch.randn(self.rows, 3 * self.channels, device=self.device).half()
        self.gamma = torch.ones(H, D, device=self.device)

    def benchmark_rmsnorm_heads(self):
        self.out = self.kernel.rmsnorm_heads(self.src, self.gamma, 1.0)

    def verify_rmsnorm_heads(self) -> torch.Tensor:
        x = self.src[:, : H * D].reshape(self.rows, H, D)
        return (F.normalize(x.float(), dim=-1) * self.gamma).half()
