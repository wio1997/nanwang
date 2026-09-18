// Stage 1B — minimal ACLNN runner for DrivingSDK ScatterMaxV1
//   data  : FP32 (aclDataType ACL_FLOAT)
//   index : INT32 (aclDataType ACL_INT32)
//   device: aclrtSetDevice(0)  (single NPU only)
//
// The runner calls the generated custom-op API (aclnnScatterMaxV1) directly.
// No torch / torch_npu / PyG / mx_driving involvement.
//
// Build: see build_runner.sh
// Usage: scattermaxv1_runner --case <name> --n <N> --f <F> --size <S>
//                            --index 0,1,2 --pattern <baseline|distinct>
//                            [--dump <file>] [--tol <float>]

#include <acl/acl.h>
#include "aclnn_scatter_max_v1.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
// aclnnStatus is int32_t and the success code of the generated ACLNN API is 0
// (this CANN version does not export an ACLNN_SUCCESS enumerator in acl_meta.h)
constexpr aclnnStatus kAclnnSuccess = 0;

struct Case {
    std::string name = "case";
    int64_t N = 8;
    int64_t F = 8;
    int64_t S = 0;
    std::vector<int32_t> index;
    std::string pattern = "baseline";
    std::string dump;
    float tol = 0.0f;
};

void Fail(const std::string& what) {
    std::printf("RUNNER_ERROR %s\n", what.c_str());
    std::exit(2);
}

#define ACL_CHECK(expr, what)                                                    \
    do {                                                                         \
        aclError _e = (expr);                                                    \
        if (_e != ACL_SUCCESS) {                                                 \
            std::printf("ACL_ERROR %s -> %d (%s:%d)\n", what, (int)_e, __FILE__, \
                        __LINE__);                                               \
            std::exit(4);                                                        \
        }                                                                        \
    } while (0)

// deterministic, no RNG dependency: mixes positive and negative values and
// makes every (row, feature) pair distinct
std::vector<float> MakeSrc(const Case& c) {
    std::vector<float> src(static_cast<size_t>(c.N * c.F));
    for (int64_t n = 0; n < c.N; ++n) {
        for (int64_t f = 0; f < c.F; ++f) {
            float v;
            if (c.pattern == "distinct") {
                v = 1.0f + static_cast<float>(n) + 0.125f * static_cast<float>(f);
                if (n % 2 == 1) {
                    v = -v;  // half the rows negative, all values distinct
                }
            } else if (c.pattern == "negative") {
                // every value strictly negative and distinct: the max of a
                // negative-only group must stay negative (never become 0)
                v = -(1.0f + static_cast<float>(n) + 0.125f * static_cast<float>(f));
            } else {
                // baseline: positives + negatives + repeated magnitudes
                int64_t m = (n * 7 + f * 13) % 37 - 18;      // [-18, 18]
                float frac = 0.25f * static_cast<float>((n * 3 + f * 5) % 7);
                v = static_cast<float>(m) + frac;
            }
            src[static_cast<size_t>(n * c.F + f)] = v;
        }
    }
    return src;
}

std::vector<float> Golden(const Case& c, const std::vector<float>& src) {
    std::vector<float> out(static_cast<size_t>(c.S * c.F), kNegInf);
    for (int64_t n = 0; n < c.N; ++n) {
        const int32_t g = c.index[static_cast<size_t>(n)];
        if (g < 0 || g >= c.S) {
            Fail("index out of requested size in golden input");
        }
        for (int64_t f = 0; f < c.F; ++f) {
            float& slot = out[static_cast<size_t>(g * c.F + f)];
            slot = std::max(slot, src[static_cast<size_t>(n * c.F + f)]);
        }
    }
    return out;
}

Case ParseArgs(int argc, char** argv) {
    Case c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) Fail("missing value for " + a);
            return argv[++i];
        };
        if (a == "--case") {
            c.name = next();
        } else if (a == "--n") {
            c.N = std::stoll(next());
        } else if (a == "--f") {
            c.F = std::stoll(next());
        } else if (a == "--size") {
            c.S = std::stoll(next());
        } else if (a == "--tol") {
            c.tol = std::stof(next());
        } else if (a == "--pattern") {
            c.pattern = next();
        } else if (a == "--dump") {
            c.dump = next();
        } else if (a == "--index") {
            std::string list = next();
            size_t pos = 0;
            while (pos <= list.size()) {
                size_t comma = list.find(',', pos);
                std::string tok = list.substr(pos, comma - pos);
                if (!tok.empty()) c.index.push_back(static_cast<int32_t>(std::stol(tok)));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        } else {
            Fail("unknown arg " + a);
        }
    }
    if (c.index.empty()) Fail("--index is required");
    if (static_cast<int64_t>(c.index.size()) != c.N) Fail("--index length must equal --n");
    if (c.S <= 0) Fail("--size must be > 0");
    return c;
}

aclTensor* MakeTensor(const std::vector<int64_t>& shape, const std::vector<int64_t>& stride,
                      aclDataType dtype, void* devPtr) {
    return aclCreateTensor(shape.data(), shape.size(), dtype, stride.data(), 0, ACL_FORMAT_ND,
                           shape.data(), shape.size(), devPtr);
}

void DumpToFile(const Case& c, const std::vector<float>& src, const std::vector<float>& expected,
                const std::vector<float>& actual, float maxAbsDiff, int64_t mismatches,
                int64_t untouchedViolations, uint64_t workspaceBytes) {
    if (c.dump.empty()) return;
    FILE* fp = std::fopen(c.dump.c_str(), "w");
    if (fp == nullptr) {
        std::printf("RUNNER_WARN cannot open dump %s\n", c.dump.c_str());
        return;
    }
    std::fprintf(fp, "case=%s\nN=%lld\nF=%lld\nsize=%lld\nworkspace_bytes=%llu\n",
                 c.name.c_str(), (long long)c.N, (long long)c.F, (long long)c.S,
                 (unsigned long long)workspaceBytes);
    std::fprintf(fp, "index=");
    for (size_t i = 0; i < c.index.size(); ++i) {
        std::fprintf(fp, "%s%d", i ? "," : "", c.index[i]);
    }
    std::fprintf(fp, "\n\nsrc[%lld][%lld]:\n", (long long)c.N, (long long)c.F);
    for (int64_t n = 0; n < c.N; ++n) {
        std::fprintf(fp, "  n=%2lld ", (long long)n);
        for (int64_t f = 0; f < c.F; ++f) {
            std::fprintf(fp, "%9.4f", src[static_cast<size_t>(n * c.F + f)]);
        }
        std::fprintf(fp, "\n");
    }
    std::fprintf(fp, "\nexpected[%lld][%lld]  (CPU golden, raw op semantics)\n", (long long)c.S,
                 (long long)c.F);
    std::fprintf(fp, "actual[%lld][%lld]\n", (long long)c.S, (long long)c.F);
    for (int64_t g = 0; g < c.S; ++g) {
        std::fprintf(fp, "  g=%2lld exp ", (long long)g);
        for (int64_t f = 0; f < c.F; ++f) {
            std::fprintf(fp, "%9.4f", expected[static_cast<size_t>(g * c.F + f)]);
        }
        std::fprintf(fp, "\n       act ");
        for (int64_t f = 0; f < c.F; ++f) {
            std::fprintf(fp, "%9.4f", actual[static_cast<size_t>(g * c.F + f)]);
        }
        std::fprintf(fp, "\n");
    }
    std::fprintf(fp, "\nmax_abs_diff=%g\nmismatch_count=%lld\nuntouched_row_violations=%lld\n",
                 maxAbsDiff, (long long)mismatches, (long long)untouchedViolations);
    std::fclose(fp);
}

int RunCase(const Case& c, const std::vector<float>& src) {
    const int64_t N = c.N, F = c.F, S = c.S;
    const size_t srcBytes = static_cast<size_t>(N * F) * sizeof(float);
    const size_t idxBytes = static_cast<size_t>(N) * sizeof(int32_t);
    const size_t resBytes = static_cast<size_t>(S * F) * sizeof(float);
    const size_t argBytes = static_cast<size_t>(S * F) * sizeof(int32_t);

    // ---- host side ----
    std::vector<float> expected = Golden(c, src);
    std::vector<float> resHost(static_cast<size_t>(S * F), kNegInf);  // caller-provided init
    std::vector<int32_t> argHost(static_cast<size_t>(S * F), -1);     // argmax required by API
    std::vector<float> resOut(static_cast<size_t>(S * F), 0.0f);

    // ---- device init ----
    ACL_CHECK(aclInit(nullptr), "aclInit");
    ACL_CHECK(aclrtSetDevice(0), "aclrtSetDevice(0)");
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream), "aclrtCreateStream");

    void* srcDev = nullptr;
    void* idxDev = nullptr;
    void* resDev = nullptr;
    void* argDev = nullptr;
    void* wsDev = nullptr;
    ACL_CHECK(aclrtMalloc(&srcDev, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc src");
    ACL_CHECK(aclrtMalloc(&idxDev, idxBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc index");
    ACL_CHECK(aclrtMalloc(&resDev, resBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc res");
    ACL_CHECK(aclrtMalloc(&argDev, argBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc argmax");

    ACL_CHECK(aclrtMemcpy(srcDev, srcBytes, src.data(), srcBytes, ACL_MEMCPY_HOST_TO_DEVICE),
              "H2D src");
    ACL_CHECK(aclrtMemcpy(idxDev, idxBytes, c.index.data(), idxBytes, ACL_MEMCPY_HOST_TO_DEVICE),
              "H2D index");
    ACL_CHECK(aclrtMemcpy(resDev, resBytes, resHost.data(), resBytes, ACL_MEMCPY_HOST_TO_DEVICE),
              "H2D res(-inf init)");
    ACL_CHECK(aclrtMemcpy(argDev, argBytes, argHost.data(), argBytes, ACL_MEMCPY_HOST_TO_DEVICE),
              "H2D argmax(-1 init)");

    // ---- aclTensor ----
    std::vector<int64_t> srcShape{N, F}, srcStride{F, 1};
    std::vector<int64_t> idxShape{N}, idxStride{1};
    std::vector<int64_t> resShape{S, F}, resStride{F, 1};
    aclTensor* srcT = MakeTensor(srcShape, srcStride, ACL_FLOAT, srcDev);
    aclTensor* idxT = MakeTensor(idxShape, idxStride, ACL_INT32, idxDev);
    aclTensor* resT = MakeTensor(resShape, resStride, ACL_FLOAT, resDev);
    aclTensor* argT = MakeTensor(resShape, resStride, ACL_INT32, argDev);
    if (!srcT || !idxT || !resT || !argT) Fail("aclCreateTensor failed");

    // ---- aclnn ----
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus st = aclnnScatterMaxV1GetWorkspaceSize(srcT, idxT, resT, argT, &workspaceSize,
                                                       &executor);
    if (st != kAclnnSuccess) {
        std::printf("ACLNN_ERROR GetWorkspaceSize status=%d workspace=%llu\n", (int)st,
                    (unsigned long long)workspaceSize);
        return 5;
    }
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc(&wsDev, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc ws");
    }
    st = aclnnScatterMaxV1(wsDev, workspaceSize, executor, stream);
    if (st != kAclnnSuccess) {
        std::printf("ACLNN_ERROR launch status=%d\n", (int)st);
        return 6;
    }
    ACL_CHECK(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    ACL_CHECK(aclrtMemcpy(resOut.data(), resBytes, resDev, resBytes, ACL_MEMCPY_DEVICE_TO_HOST),
              "D2H res");

    // ---- compare (raw op semantics: untouched rows stay -inf) ----
    float maxAbsDiff = 0.0f;
    int64_t mismatches = 0;
    for (size_t i = 0; i < resOut.size(); ++i) {
        const float e = expected[i];
        const float a = resOut[i];
        float d;
        if (std::isinf(e) && std::isinf(a) && (e > 0) == (a > 0)) {
            d = 0.0f;
        } else {
            d = std::fabs(e - a);
        }
        maxAbsDiff = std::max(maxAbsDiff, d);
        if (d > c.tol) ++mismatches;
    }
    std::vector<char> rowTouched(static_cast<size_t>(S), 0);
    for (int32_t g : c.index) {
        if (g >= 0 && g < S) rowTouched[static_cast<size_t>(g)] = 1;
    }
    int64_t untouchedViolations = 0;
    int64_t emptyRows = 0;
    for (int64_t g = 0; g < S; ++g) {
        if (!rowTouched[static_cast<size_t>(g)]) {
            ++emptyRows;
            for (int64_t f = 0; f < F; ++f) {
                if (resOut[static_cast<size_t>(g * F + f)] != kNegInf) ++untouchedViolations;
            }
        }
    }
    int64_t idxMax = *std::max_element(c.index.begin(), c.index.end());

    DumpToFile(c, src, expected, resOut, maxAbsDiff, mismatches, untouchedViolations, workspaceSize);

    std::printf("CASE %s\n", c.name.c_str());
    std::printf("N=%lld F=%lld requested_size=%lld idx_max_plus_1=%lld shape_out=[%lld,%lld]\n",
                (long long)N, (long long)F, (long long)S, (long long)(idxMax + 1), (long long)S,
                (long long)F);
    std::printf("workspace_bytes=%llu\n", (unsigned long long)workspaceSize);
    std::printf("empty_rows=%lld untouched_row_violations=%lld\n", (long long)emptyRows,
                (long long)untouchedViolations);
    std::printf("max_abs_diff=%g mismatch_count=%lld tol=%g\n", maxAbsDiff, (long long)mismatches,
                c.tol);
    std::printf("expected_row0=");
    for (int64_t f = 0; f < F; ++f) std::printf("%s%g", f ? "," : "", expected[f]);
    std::printf("\nactual_row0=");
    for (int64_t f = 0; f < F; ++f) std::printf("%s%g", f ? "," : "", resOut[f]);
    std::printf("\n");
    const bool pass = (mismatches == 0) && (untouchedViolations == 0);
    std::printf("RESULT %s\n", pass ? "PASS" : "FAIL");

    // ---- cleanup ----
    aclDestroyTensor(srcT);
    aclDestroyTensor(idxT);
    aclDestroyTensor(resT);
    aclDestroyTensor(argT);
    if (wsDev) aclrtFree(wsDev);
    aclrtFree(srcDev);
    aclrtFree(idxDev);
    aclrtFree(resDev);
    aclrtFree(argDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    Case c = ParseArgs(argc, argv);
    std::vector<float> src = MakeSrc(c);
    return RunCase(c, src);
}
