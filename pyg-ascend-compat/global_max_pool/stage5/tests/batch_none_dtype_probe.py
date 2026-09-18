import sys, warnings, torch, torch_npu
sys.path.insert(0, "/root/zyg/global_max_pool/stage6")
import pyg_ascend_compat
from torch_geometric.nn import global_max_pool
torch.npu.set_device(0)
ok = True

def check(tag, xv, dt, oned=False):
    global ok
    xc = torch.tensor(xv, dtype=dt)
    if oned:
        xc = xc.reshape(-1)
    xc = xc.requires_grad_(True)
    o_c = global_max_pool(xc, None); o_c.sum().backward()
    xn = xc.detach().clone().to("npu:0").requires_grad_(True)
    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter("always")
        o_n = global_max_pool(xn, None); o_n.sum().backward(); torch.npu.synchronize()
    fb = [w for w in c if "fall back" in str(w.message)]
    fn = torch.equal(torch.isnan(o_c.detach()), torch.isnan(o_n.detach().cpu()))
    f_ok = torch.equal(o_c.detach(), o_n.detach().cpu())
    g_ok = torch.equal(xc.grad, xn.grad.cpu())
    good = f_ok and g_ok and fn and not fb
    ok &= good
    print(f"{tag:28s} dtype={dt} fwd_eq={f_ok} bwd_eq={g_ok} nan_pattern={fn} fallback={len(fb)} -> {chr(80) if good else chr(70)}")

for dt in (torch.float16, torch.bfloat16):
    check(f"{dt}_unique", [[1.,10.],[3.,20.],[2.,30.]], dt)
    check(f"{dt}_tie", [[3.,1.],[3.,2.],[1.,3.]], dt)
    check(f"{dt}_1d", [1.,3.,2.], dt, oned=True)
print("BATCH_NONE_DTYPE:", "PASS" if ok else "FAIL")
