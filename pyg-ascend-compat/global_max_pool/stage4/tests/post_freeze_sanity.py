import torch, torch_npu, warnings, pyg_ascend_compat
pyg_ascend_compat.enable()
from torch_geometric.nn import global_max_pool
torch.npu.set_device(0)
cpu_gmp = getattr(global_max_pool, "__wrapped__", global_max_pool)
ok = True

def check(tag, x, batch, size):
    global ok
    xr = x.clone().requires_grad_(True)
    out_c = cpu_gmp(xr, batch, size); out_c.sum().backward()
    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter("always")
        xn = x.clone().to("npu:0").requires_grad_(True)
        out = global_max_pool(xn, batch.to("npu:0"), size)
        out.sum().backward(); torch.npu.synchronize()
    fb = [w for w in c if "fall back" in str(w.message)]
    oc, on = out_c.detach(), out.detach().cpu()
    fwd_nan_ok = torch.equal(torch.isnan(oc), torch.isnan(on))
    fin_f = ~torch.isnan(oc)
    dg = (oc[fin_f] - on[fin_f]).abs().max().item() if bool(fin_f.any()) else 0.0
    g_c, g_n = xr.grad, xn.grad.cpu()
    nan_ok = torch.equal(torch.isnan(g_c), torch.isnan(g_n)) and fwd_nan_ok
    fin = ~torch.isnan(g_c)
    dgrad = (g_c[fin] - g_n[fin]).abs().max().item() if bool(fin.any()) else 0.0
    good = bool(dg == 0.0 and dgrad <= 1e-6 and nan_ok and not fb)
    ok &= good
    verdict = "PASS" if good else "FAIL"
    print(f"{tag:26s} fwd_max_diff={dg} grad_max_abs_diff={dgrad} nan_pattern_ok={nan_ok} "
          f"fallback={len(fb)} -> {verdict}")

check("tie N=12 F=9 S=3", torch.randint(-4, 5, (12, 9)).float(), torch.arange(12) % 3, 3)
x = (torch.arange(40, dtype=torch.float32).unsqueeze(1) % 7.0).expand(40, 48825).contiguous()
x[39, 48824] = 11.0; x[39, 48823] = 11.0; x[0, 0] = 9.0
check("largeTail N=40 F=48825", x, torch.arange(40) % 8, 8)
check("nan N=3 F=1 S=1", torch.tensor([[float("nan")], [1.0], [2.0]]), torch.tensor([0, 0, 0]), 1)
check("pm-zero N=2 F=1 S=1", torch.tensor([[0.0], [-0.0]]), torch.tensor([0, 0]), 1)
print("STAGE4_POST_FREEZE_SANITY:", "PASS" if ok else "FAIL")
