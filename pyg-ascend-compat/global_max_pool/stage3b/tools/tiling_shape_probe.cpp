// Stage 3B shape-only tiling probe.
//
// Calls only aclnnScatterMaxV1GetWorkspaceSize (= host tiling) for an arbitrary (N, F, S) while
// allocating a tiny device buffer. The kernel is NEVER launched, so shapes that would need tens of
// GB (e.g. N=163800, F=44801) can be used to read the *actual* tiling decision safely.
//
// Combined with the instrumented probe-only OPP (tools/instrument_tiling_probe.py) the tiling
// function prints the real tiling key on stderr.

#include <acl/acl.h>
#include "aclnn_scatter_max_v1.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr size_t kTinyBytes = 512;  // deliberately NOT sized for the requested shape

aclTensor* MakeTensor(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides,
                      aclDataType dtype, void* ptr) {
    return aclCreateTensor(dims.data(), dims.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                           dims.data(), dims.size(), ptr);
}

int64_t ArgI(int argc, char** argv, const char* name, int64_t def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return std::strtoll(argv[i + 1], nullptr, 10);
    }
    return def;
}
}  // namespace

int main(int argc, char** argv) {
    const int64_t n = ArgI(argc, argv, "--n", 0);
    const int64_t f = ArgI(argc, argv, "--f", 0);
    const int64_t s = ArgI(argc, argv, "--s", 1);
    if (n <= 0 || f <= 0 || s <= 0) {
        std::printf("usage: %s --n N --f F --s S\n", argv[0]);
        return 2;
    }
    if (aclInit(nullptr) != ACL_SUCCESS) return 3;
    if (aclrtSetDevice(0) != ACL_SUCCESS) return 4;

    void* buf[4] = {nullptr, nullptr, nullptr, nullptr};
    for (auto& b : buf) {
        if (aclrtMalloc(&b, kTinyBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return 5;
    }
    std::vector<int64_t> srcDims{n, f}, srcStride{f, 1};
    std::vector<int64_t> idxDims{n}, idxStride{1};
    std::vector<int64_t> outDims{s, f}, outStride{f, 1};
    aclTensor* aSrc = MakeTensor(srcDims, srcStride, ACL_FLOAT, buf[0]);
    aclTensor* aIdx = MakeTensor(idxDims, idxStride, ACL_INT32, buf[1]);
    aclTensor* aOut = MakeTensor(outDims, outStride, ACL_FLOAT, buf[2]);
    aclTensor* aArg = MakeTensor(outDims, outStride, ACL_INT32, buf[3]);
    if (!aSrc || !aIdx || !aOut || !aArg) return 6;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // host-only call: tiling / validation. No device kernel is launched, so the tiny buffers are
    // never dereferenced.
    aclnnStatus st = aclnnScatterMaxV1GetWorkspaceSize(aSrc, aIdx, aOut, aArg, &workspaceSize,
                                                      &executor);
    std::printf("SHAPE_PROBE N=%lld F=%lld S=%lld status=%d workspace=%llu\n",
                (long long)n, (long long)f, (long long)s, (int)st,
                (unsigned long long)workspaceSize);
    std::fflush(stdout);

    aclDestroyTensor(aSrc);
    aclDestroyTensor(aIdx);
    aclDestroyTensor(aOut);
    aclDestroyTensor(aArg);
    for (auto& b : buf) aclrtFree(b);
    aclrtResetDevice(0);
    aclFinalize();
    return st == 0 ? 0 : 7;
}
