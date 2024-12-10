
#include <ort_custom_ops/custom_ops.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <mutex>
#include <vector>

#include <ort_custom_ops/opticalflow/correlation.h>
#include <ort_custom_ops/opticalflow/warp.h>

using namespace nmp;

const char* kCUDAProvider = "CUDAExecutionProvider";
const char* kCPUProvider = "CPUExecutionProvider";

OrtStatus* ORT_API_CALL addCustomOps(const OrtApi* ortApi, OrtCustomOpDomain* domain)
{
    std::vector<std::string> executors =  Ort::GetAvailableProviders();
    

    auto contains_executor = [&](std::string key) {
        return std::find(executors.begin(), executors.end(), key) != executors.end();
    };

    auto custom_ops = std::vector<OrtCustomOp*>{};

    /* Instantiate CUDA custom ops */

    if (contains_executor(kCUDAProvider)) {
        custom_ops.push_back(new FlowCorrelationCustomOp(kCUDAProvider));
        custom_ops.push_back(new FlowWarpCustomOp(kCUDAProvider));
    }

    /* Instantiate CPU custom ops */

    custom_ops.push_back(new FlowCorrelationCustomOp(kCPUProvider));
    custom_ops.push_back(new FlowWarpCustomOp(kCPUProvider));

    // Add ops to domain
    for (auto op : custom_ops) {
        if (const auto status = ortApi->CustomOpDomain_Add(domain, op)) {
            std::cerr << "Error occured while adding custom op " << op->GetName(op) << " for provider "
                      << op->GetExecutionProviderType(op) << " : " << ortApi->GetErrorMessage(status) << std::endl;

            return status;
        }
    }

    return nullptr;  // ok
}

static const char* c_OpDomain = "custom";

struct OrtCustomOpDomainDeleter {
    explicit OrtCustomOpDomainDeleter(const OrtApi* ort_api) { ort_api_ = ort_api; }
    void operator()(OrtCustomOpDomain* domain) const { ort_api_->ReleaseCustomOpDomain(domain); }

    const OrtApi* ort_api_;
};

using OrtCustomOpDomainUniquePtr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>;
static std::vector<OrtCustomOpDomainUniquePtr> ort_custom_op_domain_container;
static std::mutex ort_custom_op_domain_mutex;

static void AddOrtCustomOpDomainToContainer(OrtCustomOpDomain* domain, const OrtApi* ort_api)
{
    std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
    auto ptr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>(domain, OrtCustomOpDomainDeleter(ort_api));
    ort_custom_op_domain_container.push_back(std::move(ptr));
}

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api)
{
    OrtCustomOpDomain* domain = nullptr;

    const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);  
    if (ortApi == nullptr) {
        std::cerr << "ortApi is null" << std::endl;
        std::cerr << "api version of calling API is " << api->GetVersionString() << std::endl;
        std::cerr << "while version of library compiletime onnxruntime API is " << ORT_API_VERSION << std::endl;
        return 0;
    }

    if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
        return status;
    }

    AddOrtCustomOpDomainToContainer(domain, ortApi);

    // add actual ops to domain
    if (auto status = addCustomOps(ortApi, domain)) {
        return status;
    }

    return ortApi->AddCustomOpDomain(options, domain);
}
