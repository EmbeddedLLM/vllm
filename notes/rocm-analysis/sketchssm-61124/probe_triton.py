import torch, triton, traceback
from vllm.triton_utils import tl, triton as tt

@triton.jit
def dotk(x_ptr, r_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
         PREC: tl.constexpr, CM: tl.constexpr):
    om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
    x = tl.load(x_ptr + om[:, None]*K + ok[None, :], cache_modifier=CM)
    r = tl.load(r_ptr + ok[:, None]*N + on[None, :])
    y = tl.dot(x, r, input_precision=PREC)
    tl.store(y_ptr + om[:, None]*N + on[None, :], y)

M, N, K = 32, 128, 128
x = torch.randn(M, K, device="cuda", dtype=torch.float32)
r = torch.randn(K, N, device="cuda", dtype=torch.float32)
y = torch.empty(M, N, device="cuda", dtype=torch.float32)
ref = x.double() @ r.double()

for prec in ["ieee", "tf32x3", "bf16x3", "tf32"]:
    for cm in ["", ".cg"]:
        try:
            y.zero_()
            dotk[(1,)](x, r, y, M, N, K, PREC=prec, CM=cm)
            torch.cuda.synchronize()
            err = (y.double()-ref).abs().max().item()/ref.abs().max().item()
            print(f"prec={prec:7s} cm={cm!r:6s} OK   relerr={err:.3e}")
        except Exception as e:
            msg = str(e).splitlines()
            print(f"prec={prec:7s} cm={cm!r:6s} FAIL {type(e).__name__}: {msg[0][:160]}")
