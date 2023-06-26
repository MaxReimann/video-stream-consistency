/*  CPU IO Interface for models

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

#include <memory>

#include "IOInterface.h"

// namespace pybind11 {
// class dict;
// class str;
// }


template <typename ValueType>
class CpuIO : public IOInterface {
public:
    CpuIO(std::vector<int64_t> shape, std::vector<std::string> names);
    virtual ~CpuIO();

    DataType dataType() const override;
    MemoryType memoryType() const override;

    void* resourcePointer() override;

    std::vector<std::byte> data() const override;
    void setData(const std::vector<std::byte>& data) override;
    void setData(const std::byte* data, size_t size) override;

    void setDeviceData(const void* data, size_t size) override;

    size_t byteSize() const override;
    size_t size() const override;

    const std::vector<int64_t>& shape() const override;
    void reshape(const std::vector<int64_t>& shape) override;

    bool hasDynamicShape() const;

    const std::vector<std::string>& names() const override;

protected:
    std::vector<int64_t> mShape;
    std::vector<std::string> mNames;
    std::vector<ValueType> mMemory;
};

