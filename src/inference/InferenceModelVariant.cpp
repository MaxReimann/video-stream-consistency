/*  ONNXRuntime Model runner

    Copyright (C) 2022 Jan van Dieken

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
*/

#include "InferenceModelVariant.h"
#include <ort_custom_ops/custom_ops.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_run_options_config_keys.h>

using namespace std::string_literals;

#ifdef _MSC_VER
using OnnxString = std::wstring;
#else
using OnnxString = std::string;
#endif



ONNXTensorElementDataType convertDataType(IOInterface::DataType dataType)
{
    switch (dataType) {
        case IOInterface::DataType::INT8:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        case IOInterface::DataType::BYTE:
        case IOInterface::DataType::UINT8:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case IOInterface::DataType::INT16:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        case IOInterface::DataType::UINT16:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
        case IOInterface::DataType::FLOAT16:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case IOInterface::DataType::INT32:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case IOInterface::DataType::UINT32:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
        case IOInterface::DataType::FLOAT32:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case IOInterface::DataType::INT64:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case IOInterface::DataType::UINT64:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
        case IOInterface::DataType::FLOAT64:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        default:
            throw std::runtime_error("Unimplemented DataType");
    }
}


InferenceModelVariant::InferenceModelVariant(std::filesystem::path modelPath,
    InferenceProvider provider,
    OrtContext& ortContext,
    std::map<std::string, MappedIO> inputs,
    std::map<std::string, MappedIO> outputs) :
    mModelPath{std::move(modelPath)}
    , mProvider{provider}
    , mOrtContext{ortContext}
    , mSession{nullptr}
    , mOutputs{std::move(outputs)}
{
    for (auto& [map, mappedIo] : inputs) {
        mInputs.emplace(map, InternalMappedIO{std::move(mappedIo.first), mappedIo.second, {}});
        // mInputs.emplace(map, InternalMappedIO{std::move(mappedIo.first), std::move(mappedIo.second), {}});
    }
    mHasDynamicOutputShape = std::any_of(mOutputs.cbegin(), mOutputs.cend(), [](const auto& entry) {
      return entry.second.second->hasDynamicShape();
    });
}

InferenceModelVariant::~InferenceModelVariant() = default;

InferenceModelVariant::InferenceModelVariant(InferenceModelVariant&&) noexcept = default;

void InferenceModelVariant::loadSession()
{
    if (sessionIsLoaded()) {
        return;
    }

    mSession = createSession(mModelPath, mProvider, mOrtContext);
}

void InferenceModelVariant::unloadSession()
{
    mSession.reset();
}

void InferenceModelVariant::reloadSession()
{
    mSession = createSession(mModelPath, mProvider, mOrtContext);
}

bool InferenceModelVariant::sessionIsLoaded() const
{
    return mSession != nullptr;
}

IOInterface* InferenceModelVariant::input(const std::string& id)
{
    return mInputs.at(id).io.get();
}

IOInterface* InferenceModelVariant::output(const std::string& id)
{
    return mOutputs.at(id).second.get();
}


void InferenceModelVariant::run()
{
    if (!sessionIsLoaded()) {
        return;
    }

    const auto inputShapeChanged = reloadIfInputShapeChanged();

    //For dynamic outputs: if the input shape did not change, and it is not the first run we can do a static run as
    //we know the output shape from the previous run.
    if ((mIsFirstRun && mHasDynamicOutputShape) || (inputShapeChanged && mHasDynamicOutputShape)) {
        runDynamic();
    } else {
        runStatic();
    }
    mIsFirstRun = false;
}

std::unique_ptr<Ort::Session> InferenceModelVariant::createSession(const std::filesystem::path& modelPath,
    InferenceProvider provider,
    OrtContext& ortContext)
{
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // sessionOptions.EnableProfiling("profile");

    RegisterCustomOps(sessionOptions, OrtGetApiBase());

    // Ort::ArenaCfg arenaCfg{0, 0, 1024, -1};
    if (provider == InferenceProvider::CUDA) {
        OrtCUDAProviderOptions cudaOptions;
        // cudaOptions.cudnn_conv_algo_search =  OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchHeuristic;
        // cudaOptions.default_memory_arena_cfg = arenaCfg;

        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
    }

    auto modelPathString = modelPath.u8string();

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::cout << "InferenceModelVariant session initializing..." << std::endl;
    try {
        auto session = std::make_unique<Ort::Session>(ortContext.environment(),
            OnnxString{modelPathString.cbegin(), modelPathString.cend()}.c_str(),
            sessionOptions);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "InferenceModelVariant session initialized. It took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

        return session;
    } catch (std::exception &e) {
        std::cerr << "Exception occurred during onnx runtime session initialization " << std::endl;
        std::rethrow_exception(std::current_exception());
    }
}

Ort::MemoryInfo InferenceModelVariant::createMemoryInfo(InferenceProvider provider)
{
    switch (provider) {
        case InferenceProvider::CPU:
            return Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        case InferenceProvider::CUDA:
            return Ort::MemoryInfo{"Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault};
        default:
            throw std::runtime_error(
                "Unknown provider"s + std::to_string(static_cast<std::underlying_type_t<InferenceProvider>>(provider)));
    }
}

Ort::RunOptions InferenceModelVariant::createRunOptions(InferenceProvider provider)
{
    switch (provider) {
        case InferenceProvider::CPU:
            return Ort::RunOptions{nullptr};
        case InferenceProvider::CUDA: {
            Ort::RunOptions runOptions;
            // runOptions.AddConfigEntry(kOrtRunOptionsConfigEnableMemoryArenaShrinkage, "gpu:0");
            // runOptions.SetRunLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);
            // runOptions.SetRunLogVerbosityLevel(0);
            return runOptions;
        }
        default:
            throw std::runtime_error(
                "Unknown provider"s + std::to_string(static_cast<std::underlying_type_t<InferenceProvider>>(provider)));
    }
}

void InferenceModelVariant::runDynamic()
{
    auto memoryInfo = createMemoryInfo(mProvider);

    std::vector<Ort::Value> inputTensors;
    inputTensors.reserve(mInputs.size());
    std::vector<std::string> inputNames;
    inputNames.reserve(mInputs.size());
    for (auto& [map, mappedInput] : mInputs) {
        auto* rawInput = mappedInput.io.get();

        inputTensors.push_back(Ort::Value::CreateTensor(memoryInfo,
            rawInput->resourcePointer(),
            rawInput->byteSize(),
            rawInput->shape().data(),
            rawInput->shape().size(),
            convertDataType(rawInput->dataType())));
        inputNames.push_back(mappedInput.id);
    }

    std::vector<std::string> outputNames;
    outputNames.reserve(mOutputs.size());
    for (auto& [map, mappedOutput] : mOutputs) {
        outputNames.push_back(mappedOutput.first);
    }

    std::vector<const char*> rawInputNames(inputNames.size());
    std::transform(inputNames.cbegin(), inputNames.cend(), rawInputNames.begin(), [](const std::string& inputName) {
        return inputName.c_str();
    });
    std::vector<const char*> rawOutputNames(outputNames.size());
    std::transform(outputNames.cbegin(), outputNames.cend(), rawOutputNames.begin(), [](const std::string& outputName) {
        return outputName.c_str();
    });

    auto runOptions = createRunOptions(mProvider);

    std::vector<Ort::Value> outputTensors;
    outputTensors = mSession->Run(runOptions,
        rawInputNames.data(),
        inputTensors.data(),
        inputTensors.size(),
        rawOutputNames.data(),
        rawOutputNames.size());

    for (size_t i = 0; i < outputTensors.size(); ++i) {
        auto& mappedOutput = mOutputs.at(outputNames[i]);
        auto* rawOutput = mappedOutput.second.get();

        auto typeAndShapeInfo = outputTensors[i].GetTensorTypeAndShapeInfo();
        auto shape = typeAndShapeInfo.GetShape();
        rawOutput->reshape(shape);

        auto data = outputTensors[i].GetTensorData<std::byte>();
        rawOutput->setData(data, rawOutput->byteSize());
    }
}

void InferenceModelVariant::runStatic()
{
    Ort::IoBinding io_binding{*mSession.get()}; 
    auto memoryInfo = createMemoryInfo(mProvider);

    std::vector<Ort::Value> inputTensors;
    inputTensors.reserve(mInputs.size());
    std::vector<std::string> inputNames;
    inputNames.reserve(mInputs.size());
    for (auto& [map, mappedInput] : mInputs) {
        auto* rawInput = mappedInput.io.get();

        inputTensors.push_back(Ort::Value::CreateTensor(memoryInfo,
            rawInput->resourcePointer(),
            rawInput->byteSize(),
            rawInput->shape().data(),
            rawInput->shape().size(),
            convertDataType(rawInput->dataType())));
        inputNames.push_back(mappedInput.id);
        io_binding.BindInput(mappedInput.id.c_str(), inputTensors.back());
        
        // // print name and shapes of raw input
        // std::cout << "raw input name: " << mappedInput.id << std::endl;
        // // print every value of rawInput->shape()
        // std::cout << "raw input shape: ";
        // for (auto& i : inputTensors.back().GetTensorTypeAndShapeInfo().GetShape()) {
        //     std::cout << i << " ";
        // }

        // std::cout << std::endl;
    }

    std::vector<Ort::Value> outputTensors;
    outputTensors.reserve(mOutputs.size());
    std::vector<std::string> outputNames;
    outputNames.reserve(mOutputs.size());
    for (auto& [map, mappedOutput] : mOutputs) {
        auto* rawOutput = mappedOutput.second.get();

        outputTensors.push_back(Ort::Value::CreateTensor(memoryInfo,
            rawOutput->resourcePointer(),
            rawOutput->byteSize(),
            rawOutput->shape().data(),
            rawOutput->shape().size(),
            convertDataType(rawOutput->dataType())));
        outputNames.push_back(mappedOutput.first);
        io_binding.BindOutput(outputNames.back().c_str(), outputTensors.back());
    }

    std::vector<const char*> rawInputNames(inputNames.size());
    std::transform(inputNames.cbegin(), inputNames.cend(), rawInputNames.begin(), [](const std::string& inputName) {
        return inputName.c_str();
    });
    std::vector<const char*> rawOutputNames(outputNames.size());
    std::transform(outputNames.cbegin(), outputNames.cend(), rawOutputNames.begin(), [](const std::string& outputName) {
        return outputName.c_str();
    });



    auto runOptions = createRunOptions(mProvider);
    mSession->Run(runOptions,
        rawInputNames.data(),
        inputTensors.data(),
        inputTensors.size(),
        rawOutputNames.data(),
        outputTensors.data(),
        outputTensors.size());
}

bool InferenceModelVariant::reloadIfInputShapeChanged()
{
    //If the shape of an input changed we have to reload the session. This is done to reset internal
    //data of the onnxruntime and is the most performant option. This includes resetting dimension parameters (dim_param).
    //Otherwise, the onnxruntime would expect the shapes we used the last time.
    //See https://github.com/microsoft/onnxruntime/issues/1632
    //and https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/imagenet/main.cc (ResetCache)

    bool needReload = false;
    for (auto& [map, mappedInput] : mInputs) {
        auto* rawIo = mappedInput.io.get();
        if (mappedInput.lastShape) {
            if (rawIo->shape() != *mappedInput.lastShape) {
                needReload = true;
                mappedInput.lastShape = rawIo->shape();
            }
        } else {
            mappedInput.lastShape = rawIo->shape();
        }
    }
    if (needReload) {
        reloadSession();
    }
    return needReload;
}
