/*  ONNXRuntime Model IO (CPU)

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
#include "CpuIO.h"

#include <algorithm>
#include <numeric>
#include <iostream>
#include <cstring> 

namespace {
template <typename>
constexpr bool always_false_v = false;
}

template <typename ValueType>
CpuIO<ValueType>::CpuIO(std::vector<int64_t> shape, std::vector<std::string> names)
    : mShape{std::move(shape)}
    , mNames{std::move(names)}
    , mMemory{std::vector<ValueType>(
          std::accumulate(mShape.cbegin(), mShape.cend(), int64_t{1}, std::multiplies<int64_t>()))}
{
}

template <typename ValueType>
CpuIO<ValueType>::~CpuIO() = default;

template <typename ValueType>
IOInterface::DataType CpuIO<ValueType>::dataType() const
{
    if constexpr (std::is_same_v<std::byte, ValueType>) {
        return DataType::BYTE;
    } else if constexpr (std::is_same_v<int8_t, ValueType>) {
        return DataType::INT8;
    } else if constexpr (std::is_same_v<uint8_t, ValueType>) {
        return DataType::UINT8;
    } else if constexpr (std::is_same_v<int16_t, ValueType>) {
        return DataType::INT16;
    } else if constexpr (std::is_same_v<uint16_t, ValueType>) {
        return DataType::UINT16;
    } else if constexpr (std::is_same_v<int32_t, ValueType>) {
        return DataType::INT32;
    } else if constexpr (std::is_same_v<uint32_t, ValueType>) {
        return DataType::UINT32;
    } else if constexpr (std::is_same_v<float, ValueType>) {
        return DataType::FLOAT32;
    } else if constexpr (std::is_same_v<int64_t, ValueType>) {
        return DataType::INT64;
    } else if constexpr (std::is_same_v<uint64_t, ValueType>) {
        return DataType::UINT64;
    } else if constexpr (std::is_same_v<double, ValueType>) {
        return DataType::FLOAT64;
    } else {
        static_assert(always_false_v<ValueType>, "Unimplemented DataType");
    }
}

template <typename ValueType>
MemoryType CpuIO<ValueType>::memoryType() const
{
    return MemoryType::CPU;
}

template <typename ValueType>
void* CpuIO<ValueType>::resourcePointer()
{
    return mMemory.data();
}

template <typename ValueType>
std::vector<std::byte> CpuIO<ValueType>::data() const
{
    std::vector<std::byte> data(byteSize());
    std::memcpy(data.data(), mMemory.data(), data.size());
    return data;
}

template <typename ValueType>
void CpuIO<ValueType>::setData(const std::vector<std::byte>& data)
{
    setData(data.data(), data.size());
}

template <typename ValueType>
void CpuIO<ValueType>::setData(const std::byte* data, size_t size)
{
    if (size != byteSize()) {
        throw std::runtime_error("Size of of data does not match size of memory.");
    }
    std::memcpy(mMemory.data(), data, size);
}

template <typename ValueType>
void CpuIO<ValueType>::setDeviceData(const void* data, size_t size)
{
    setData(reinterpret_cast<const std::byte*>(data), size);
}

template <typename ValueType>
size_t CpuIO<ValueType>::byteSize() const
{
    return mMemory.size() * sizeof(ValueType);
}

template <typename ValueType>
size_t CpuIO<ValueType>::size() const
{
    return mMemory.size();
}

template <typename ValueType>
const std::vector<int64_t>& CpuIO<ValueType>::shape() const
{
    return mShape;
}

template <typename ValueType>
void CpuIO<ValueType>::reshape(const std::vector<int64_t>& shape)
{
    if (shape == mShape) {
        return;
    }

    mMemory.resize(
        static_cast<size_t>(std::accumulate(shape.cbegin(), shape.cend(), int64_t{1}, std::multiplies<int64_t>())));
    mShape = shape;
}

template <typename ValueType>
bool CpuIO<ValueType>::hasDynamicShape() const
{
    return std::any_of(mShape.cbegin(), mShape.cend(), [](int64_t s) { return s == 0; });
}

template <typename ValueType>
const std::vector<std::string>& CpuIO<ValueType>::names() const
{
    return mNames;
}

template class CpuIO<std::byte>;
template class CpuIO<int8_t>;
template class CpuIO<uint8_t>;
template class CpuIO<int16_t>;
template class CpuIO<uint16_t>;
template class CpuIO<int32_t>;
template class CpuIO<uint32_t>;
template class CpuIO<float>;
template class CpuIO<int64_t>;
template class CpuIO<uint64_t>;
template class CpuIO<double>;

