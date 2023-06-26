/*  Generic IO Interface for models

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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "MemoryType.h"

class IOInterface {
public:
    enum class DataType : int8_t {
        BYTE,
        INT8,
        UINT8,
        INT16,
        UINT16,
        FLOAT16,
        INT32,
        UINT32,
        FLOAT32,
        INT64,
        UINT64,
        FLOAT64
    };

    virtual DataType dataType() const = 0;
    virtual MemoryType memoryType() const = 0;

    virtual void* resourcePointer() = 0;

    virtual std::vector<std::byte> data() const = 0;
    virtual void setData(const std::vector<std::byte>& data) = 0;
    virtual void setData(const std::byte* data, size_t size) = 0;

    virtual void setDeviceData(const void* data, size_t size) = 0;

    virtual size_t byteSize() const = 0;
    virtual size_t size() const = 0;

    virtual const std::vector<int64_t>& shape() const = 0;
    virtual void reshape(const std::vector<int64_t>& shape) = 0;

    virtual bool hasDynamicShape() const = 0;

    virtual const std::vector<std::string>& names() const = 0;
};
