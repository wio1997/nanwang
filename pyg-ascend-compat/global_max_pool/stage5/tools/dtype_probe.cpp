// Stage 5A — does the delivered ScatterMaxV1 OPP accept FP16 / BF16 tensors?
//
// Calls aclnnScatterMaxV1GetWorkspaceSize (host tiling + dtype validation) for fp32 (control),
// fp16 and bf16 src/res tensors.  No kernel is launched, so tiny buffers are enough.
#include <acl/acl.h>
#include "aclnn_scatter_max_v1.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr size_t kTiny = 512;

aclTensor* Make(const std::vector<int64_t>& dims, aclDataType dt, void* p) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return aclCreateTensor(dims.data(), dims.size(), dt, strides.data(), 0, ACL_FORMAT_ND,
                           dims.data(), dims.size(), p);
}

int TryDtype(aclDataType dt, const char* name, const std::vector<int64_t>& srcDims,
             const std::vector<int64_t>& idxDims, const std::vector<int64_t>& outDims) {
    void* buf[4] = {nullptr, nullptr, nullptr, nullptr};
    for (auto& b : buf) {
        if (aclrtMalloc(&b, kTiny, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return 5;
    }
    aclTensor* aSrc = Make(srcDims, dt, buf[0]);
    aclTensor* aIdx = Make(idxDims, ACL_INT32, buf[1]);
    aclTensor* aOut = Make(outDims, dt, buf[2]);
    aclTensor* aArg = Make(outDims, ACL_INT32, buf[3]);
    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    aclnnStatus st = aclnnScatterMaxV1GetWorkspaceSize(aSrc, aIdx, aOut, aArg, &ws, &ex);
    std::printf("DTYPE_PROBE %-12s status=%d workspace=%llu\n", name, (int)st,
                (unsigned long long)ws);
    aclDestroyTensor(aSrc);
    aclDestroyTensor(aIdx);
    aclDestroyTensor(aOut);
    aclDestroyTensor(aArg);
    for (auto& b : buf) aclrtFree(b);
    return st == 0 ? 0 : 1;
}
}  // namespace

int main() {
    if (aclInit(nullptr) != ACL_SUCCESS) return 2;
    if (aclrtSetDevice(0) != ACL_SUCCESS) return 3;
    const std::vector<int64_t> srcDims{64, 32}, idxDims{64}, outDims{4, 32};
    int rc = 0;
    rc |= TryDtype(ACL_FLOAT, "fp32(control)", srcDims, idxDims, outDims);
    rc |= TryDtype(ACL_FLOAT16, "fp16", srcDims, idxDims, outDims);
    rc |= TryDtype(ACL_BF16, "bf16", srcDims, idxDims, outDims);
    aclrtResetDevice(0);
    aclFinalize();
    return rc;
}
