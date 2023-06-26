/*  ONNXRuntime Model IO (CUDA)

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
#include "CudaIO.h"

#include <algorithm>
#include <numeric>
#include <iostream>

#include <cuda.h>



namespace {
template <typename>
constexpr bool always_false_v = false;
}

template <typename ValueType>
CudaIO<ValueType>::CudaIO(std::vector<int64_t> shape, std::vector<std::string> names)
    : mShape{std::move(shape)}
    , mNames{std::move(names)}
{
    const auto byteSize =
        std::accumulate(mShape.cbegin(), mShape.cend(), int64_t{1}, std::multiplies<int64_t>()) * sizeof(ValueType);
    auto error = cudaMalloc(&mMemory, byteSize);
    if (error != cudaError::cudaSuccess) {
        throw std::runtime_error("Unable to allocate CUDA memory.");
    }
    error = cudaMemset(mMemory, 0, byteSize);
    if (error != cudaError::cudaSuccess) {
        throw std::runtime_error("Unable to zero out CUDA memory.");
    }
}

template <typename ValueType>
CudaIO<ValueType>::~CudaIO()
{
    cudaFree(mMemory);
}

template <typename ValueType>
IOInterface::DataType CudaIO<ValueType>::dataType() const
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

    return DataType::BYTE; // Should never get here
}

template <typename ValueType>
MemoryType CudaIO<ValueType>::memoryType() const
{
    return MemoryType::CUDA;
}

template <typename ValueType>
void* CudaIO<ValueType>::resourcePointer()
{
    return mMemory;
}

template <typename ValueType>
std::vector<std::byte> CudaIO<ValueType>::data() const
{
    std::vector<std::byte> data(byteSize());
    const auto error = cudaMemcpy(data.data(), mMemory, data.size(), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    return data;
}

template <typename ValueType>
void CudaIO<ValueType>::setData(const std::vector<std::byte>& data)
{
    setData(data.data(), data.size());
}

template <typename ValueType>
void CudaIO<ValueType>::setData(const std::byte* data, size_t size)
{
    if (size != byteSize()) {
        throw std::runtime_error("Size of of data does not match size of memory.");
    }
    const auto error = cudaMemcpy(mMemory, data, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (error != cudaError::cudaSuccess) {
        throw std::runtime_error("Unable to copy data from host to device.");
    }
}

template <typename ValueType>
void CudaIO<ValueType>::setDeviceData(const void* data, size_t size)
{
    if (size != byteSize()) {
        throw std::runtime_error("Size of of data does not match size of memory.");
    }
    const auto error = cudaMemcpy(mMemory, data, size, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    if (error != cudaError::cudaSuccess) {
        throw std::runtime_error("Unable to copy data from device to device.");
    }
}

template <typename ValueType>
size_t CudaIO<ValueType>::byteSize() const
{
    return size() * sizeof(ValueType);
}

template <typename ValueType>
size_t CudaIO<ValueType>::size() const
{
    return static_cast<size_t>(std::accumulate(mShape.cbegin(), mShape.cend(), int64_t{1}, std::multiplies<int64_t>()));
}

template <typename ValueType>
const std::vector<int64_t>& CudaIO<ValueType>::shape() const
{
    return mShape;
}

template <typename ValueType>
void CudaIO<ValueType>::reshape(const std::vector<int64_t>& shape)
{
    if (shape == mShape) {
        return;
    }

    const auto newSize = std::accumulate(shape.cbegin(), shape.cend(), int64_t{1}, std::multiplies<int64_t>()) * sizeof(ValueType);
    size_t toCopy = std::min(byteSize(), newSize);

    void* newMemory = nullptr;
    auto error = cudaMalloc(&newMemory, newSize);
    if (error != cudaError::cudaSuccess) {
        throw std::runtime_error("Unable to allocate CUDA memory.");
    }
    error = cudaMemset(newMemory, 0, newSize);
    if (error != cudaError::cudaSuccess) {
        throw std::runtime_error("Unable to zero out CUDA memory.");
    }
    error = cudaMemcpy(newMemory, mMemory, toCopy, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    if (error != cudaError::cudaSuccess) {
        throw std::runtime_error("Unable to copy data from device to device.");
    }
    error = cudaFree(mMemory);
    if (error != cudaError::cudaSuccess) {
        throw std::runtime_error("Unable to free CUDA memory.");
    }

    mShape = shape;
    mMemory = newMemory;
}

template <typename ValueType>
bool CudaIO<ValueType>::hasDynamicShape() const
{
    return std::any_of(mShape.cbegin(), mShape.cend(), [](int64_t s) { return s == 0ll; });
}

template <typename ValueType>
const std::vector<std::string>& CudaIO<ValueType>::names() const
{
    return mNames;
}

template class CudaIO<std::byte>;
template class CudaIO<int8_t>;
template class CudaIO<uint8_t>;
template class CudaIO<int16_t>;
template class CudaIO<uint16_t>;
template class CudaIO<int32_t>;
template class CudaIO<uint32_t>;
template class CudaIO<float>;
template class CudaIO<int64_t>;
template class CudaIO<uint64_t>;
template class CudaIO<double>;

