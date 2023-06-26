
#pragma once

#include <memory>


namespace Ort {
struct Env;
}


class OrtContext {
public:
    OrtContext();
    explicit OrtContext(Ort::Env& environment);

    ~OrtContext();

    OrtContext(const OrtContext&) = delete;
    OrtContext(OrtContext&&) noexcept;

    OrtContext& operator=(const OrtContext&) = delete;
    OrtContext& operator=(OrtContext&&) noexcept = delete;

    Ort::Env& environment();

protected:
    std::unique_ptr<Ort::Env> mEnvironment;
    Ort::Env& mEnvironmentRef;
};
