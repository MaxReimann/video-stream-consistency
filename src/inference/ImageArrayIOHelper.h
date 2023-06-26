/*  ONNXRuntime Model IO Helpers

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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>


class IOInterface;

class ImageArrayIOHelper {
public:
    template <typename IOType>
    static std::shared_ptr<IOType> createImageArrayIO(int32_t width, int32_t height, int32_t channels, int32_t length);

    static int32_t width(const IOInterface& io);
    static int32_t height(const IOInterface& io);
    static int32_t channels(const IOInterface& io);
    static int32_t length(const IOInterface& io);

    static void setWidth(IOInterface& io, int32_t width);
    static void setHeight(IOInterface& io, int32_t height);
    static void setChannels(IOInterface& io, int32_t channels);
    static void setLength(IOInterface& io, int32_t length);

    static void resize(IOInterface& io, int32_t width, int32_t height);

protected:
    const static inline std::vector<std::string> NAMES = {"BatchSize", "Height", "Width", "Channels"};
};

template <typename IOType>
inline std::shared_ptr<IOType> ImageArrayIOHelper::createImageArrayIO(int32_t width,
    int32_t height,
    int32_t channels,
    int32_t length)
{
    std::vector<int64_t> shape{static_cast<int64_t>(length),
        static_cast<int64_t>(height),
        static_cast<int64_t>(width),
        static_cast<int64_t>(channels)};
    return std::make_shared<IOType>(std::move(shape), NAMES);
}