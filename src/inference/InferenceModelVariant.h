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
#pragma once

#if __has_include(<filesystem>)
#include <filesystem>
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace std {
namespace filesystem = std::experimental::filesystem;
}
#else
error "Missing the <filesystem> header."
#endif
#include <map>
#include <optional>

#include "OrtContext.h"
#include "IOInterface.h"

namespace Ort {
struct Session;
struct MemoryInfo;
struct RunOptions;
}

enum class InferenceProvider : int8_t { CPU, CUDA };


class InferenceModelVariant {
public:
    using MappedIO = std::pair<std::string, std::shared_ptr<IOInterface>>;

    InferenceModelVariant(std::filesystem::path modelPath,
        InferenceProvider provider,
        OrtContext& ortContext,
        std::map<std::string, MappedIO> inputs,
        std::map<std::string, MappedIO> outputs);

    ~InferenceModelVariant();

    InferenceModelVariant(const InferenceModelVariant&) = delete;
    InferenceModelVariant(InferenceModelVariant&&) noexcept;

    InferenceModelVariant& operator=(const InferenceModelVariant&) = delete;
    InferenceModelVariant& operator=(InferenceModelVariant&&) noexcept = delete;

    void loadSession();
    void unloadSession();
    void reloadSession();
    bool sessionIsLoaded() const;

    IOInterface* input(const std::string& id);
    IOInterface* output(const std::string& id);

    void run();

protected:
    struct InternalMappedIO {
        std::string id;
        std::shared_ptr<IOInterface> io;
        std::optional<std::vector<int64_t>> lastShape;
    };

    static std::unique_ptr<Ort::Session> createSession(const std::filesystem::path& modelPath,
        InferenceProvider provider,
        OrtContext& ortContext);
    static Ort::MemoryInfo createMemoryInfo(InferenceProvider provider);
    static Ort::RunOptions createRunOptions(InferenceProvider provider);

    void runDynamic();
    void runStatic();

    bool reloadIfInputShapeChanged();

    std::filesystem::path mModelPath;
    InferenceProvider mProvider;
    OrtContext& mOrtContext;

    std::unique_ptr<Ort::Session> mSession;

    std::map<std::string, InternalMappedIO> mInputs;
    std::map<std::string, MappedIO> mOutputs;
    bool mHasDynamicOutputShape = false;
    bool mIsFirstRun = true;
};
