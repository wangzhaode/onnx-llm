//
//  ortwrapper.hpp
//
//  Created by zhaode on 2024/10/09.
//  ZhaodeWang
//

#ifndef ORTWRAPPER_hpp
#define ORTWRAPPER_hpp

#include <onnxruntime_cxx_api.h>
#include <memory>

namespace Ort {

class RuntimeManager {
public:
    RuntimeManager() {
        env_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx-llm"));
        options_.reset(new Ort::SessionOptions());
        options_->SetIntraOpNumThreads(1);
        options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        allocator_.reset(new Ort::AllocatorWithDefaultOptions());
    }
    ~RuntimeManager() {}
    const Ort::Env& env() const {
        return *env_;
    }
    const Ort::SessionOptions& options() const {
        return *options_;
    }
    const Ort::AllocatorWithDefaultOptions& allocator() const {
        return *allocator_;
    }
private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> options_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;
};

class Module {
public:
    Module(std::shared_ptr<RuntimeManager> runtime, const std::string& path) {
        session_.reset(new Ort::Session(runtime->env(), path.c_str(), runtime->options()));
        input_count_ = session_->GetInputCount();
        output_count_ = session_->GetOutputCount();
        for (int i = 0; i < input_count_; i++) {
            input_strs_.push_back(session_->GetInputNameAllocated(i, runtime->allocator()));
            input_names_.push_back(input_strs_[i].get());
        }
        for (int i = 0; i < output_count_; i++) {
            output_strs_.push_back(session_->GetOutputNameAllocated(i, runtime->allocator()));
            output_names_.push_back(output_strs_[i].get());
        }
    }
    std::vector<Value> onForward(const std::vector<Value>& inputs) {
        auto outputs = session_->Run(Ort::RunOptions{nullptr},
            input_names_.data(), inputs.data(), inputs.size(),
            output_names_.data(), output_names_.size());
        return outputs;
    }
private:
    std::unique_ptr<Ort::Session> session_;
    size_t input_count_, output_count_;
    std::vector<AllocatedStringPtr> input_strs_, output_strs_;
    std::vector<const char*> input_names_, output_names_;
};

template <typename T>
static Value _Input(const std::vector<int>& shape, std::shared_ptr<RuntimeManager> rtmgr) {
    std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    return Value::CreateTensor<T>(rtmgr->allocator(), shape_int64.data(), shape_int64.size());
}

} // namespace Ort

#endif /* ORTWRAPPER_hpp */