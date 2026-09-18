// Stage 2 — minimal PyTorch <-> ACLNN bridge for DrivingSDK ScatterMaxV1.
//
// Python/ATen Tensor  ->  aclTensor  ->  aclnnScatterMaxV1  ->  out (in place)
//
// Scope: FP32 src [N,F], INT32 index [N], FP32 out [S,F] (caller initialised, typically -inf),
//        INT32 argmax scratch [S,F] (mandatory in the generated ACLNN schema, contents unused by
//        the forward kernel).
//
// The op is launched on the *current torch_npu stream*, so it stays ordered with the rest of the
// adapter (occupancy scatter_, masked_fill_, ...).

#include <torch/extension.h>

#include <acl/acl.h>
#include "aclnn_scatter_max_v1.h"

#include <c10/core/DeviceGuard.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include <vector>

namespace {

constexpr aclnnStatus kAclnnSuccess = 0;

void CheckNpuTensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1,
                name, " must be an Ascend NPU tensor, got device ", t.device());
}

aclTensor* MakeAclTensor(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides,
                         aclDataType dtype, void* devicePtr) {
    return aclCreateTensor(dims.data(), dims.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                           dims.data(), dims.size(), devicePtr);
}

// out is updated in place and returned for convenience.
torch::Tensor ScatterMaxV1Forward(torch::Tensor src, torch::Tensor index, torch::Tensor out,
                                  torch::Tensor argmax) {
    CheckNpuTensor(src, "src");
    CheckNpuTensor(index, "index");
    CheckNpuTensor(out, "out");
    CheckNpuTensor(argmax, "argmax");
    TORCH_CHECK(src.device() == index.device() && src.device() == out.device() &&
                    src.device() == argmax.device(),
                "all tensors must be on the same NPU device");
    TORCH_CHECK(src.scalar_type() == at::kFloat, "src must be float32");
    TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
    TORCH_CHECK(index.scalar_type() == at::kInt, "index must be int32");
    TORCH_CHECK(argmax.scalar_type() == at::kInt, "argmax scratch must be int32");
    TORCH_CHECK(src.dim() == 2, "src must be 2D [N,F]");
    TORCH_CHECK(index.dim() == 1, "index must be 1D [N]");
    TORCH_CHECK(out.dim() == 2, "out must be 2D [S,F]");
    TORCH_CHECK(argmax.dim() == 2, "argmax must be 2D [S,F]");
    TORCH_CHECK(src.is_contiguous() && index.is_contiguous() && out.is_contiguous() &&
                    argmax.is_contiguous(),
                "all tensors must be contiguous");

    const int64_t N = src.size(0);
    const int64_t F = src.size(1);
    const int64_t S = out.size(0);
    TORCH_CHECK(index.size(0) == N, "index length must equal src.size(0)");
    TORCH_CHECK(out.size(1) == F, "out feature dim must equal src feature dim");
    TORCH_CHECK(argmax.size(0) == S && argmax.size(1) == F, "argmax scratch shape must be [S,F]");

    c10::DeviceGuard guard(src.device());
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    const std::vector<int64_t> srcDims{N, F};
    const std::vector<int64_t> srcStrides{F, 1};
    const std::vector<int64_t> idxDims{N};
    const std::vector<int64_t> idxStrides{1};

    aclTensor* aSrc = MakeAclTensor(srcDims, srcStrides, ACL_FLOAT, src.data_ptr());
    aclTensor* aIdx = MakeAclTensor(idxDims, idxStrides, ACL_INT32, index.data_ptr());
    aclTensor* aOut = MakeAclTensor(srcDims, srcStrides, ACL_FLOAT, out.data_ptr());
    aclTensor* aArg = MakeAclTensor(srcDims, srcStrides, ACL_INT32, argmax.data_ptr());
    TORCH_CHECK(aSrc && aIdx && aOut && aArg, "aclCreateTensor failed");

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus st =
        aclnnScatterMaxV1GetWorkspaceSize(aSrc, aIdx, aOut, aArg, &workspaceSize, &executor);
    TORCH_CHECK(st == kAclnnSuccess, "aclnnScatterMaxV1GetWorkspaceSize failed, status=", st);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        TORCH_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) ==
                        ACL_SUCCESS,
                    "aclrtMalloc(workspace) failed");
    }
    st = aclnnScatterMaxV1(workspace, workspaceSize, executor, stream);
    if (workspace != nullptr) {
        // the workspace is consumed asynchronously by the kernel on `stream`
        aclrtSynchronizeStream(stream);
        aclrtFree(workspace);
    }
    aclDestroyTensor(aSrc);
    aclDestroyTensor(aIdx);
    aclDestroyTensor(aOut);
    aclDestroyTensor(aArg);
    TORCH_CHECK(st == kAclnnSuccess, "aclnnScatterMaxV1 launch failed, status=", st);
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scatter_max_v1_forward", &ScatterMaxV1Forward,
          "DrivingSDK ScatterMaxV1 forward (FP32 data + INT32 index, in-place on out)",
          py::arg("src"), py::arg("index"), py::arg("out"), py::arg("argmax"));
}
