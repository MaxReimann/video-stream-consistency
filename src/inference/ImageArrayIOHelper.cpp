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

#include "ImageArrayIOHelper.h"

#include "IOInterface.h"

int32_t ImageArrayIOHelper::width(const IOInterface& io)
{
    return io.shape()[2];
}

int32_t ImageArrayIOHelper::height(const IOInterface& io)
{
    return io.shape()[1];
}

int32_t ImageArrayIOHelper::channels(const IOInterface& io)
{
    return io.shape()[3];
}

int32_t ImageArrayIOHelper::length(const IOInterface& io)
{
    return io.shape()[0];
}

void ImageArrayIOHelper::setWidth(IOInterface& io, int32_t width)
{
    auto shape = io.shape();
    shape[2] = width;
    io.reshape(shape);
}

void ImageArrayIOHelper::setHeight(IOInterface& io, int32_t height)
{
    auto shape = io.shape();
    shape[1] = height;
    io.reshape(shape);
}

void ImageArrayIOHelper::setChannels(IOInterface& io, int32_t channels)
{
    auto shape = io.shape();
    shape[3] = channels;
    io.reshape(shape);
}

void ImageArrayIOHelper::setLength(IOInterface& io, int32_t length)
{
    auto shape = io.shape();
    shape[0] = length;
    io.reshape(shape);
}

void ImageArrayIOHelper::resize(IOInterface& io, int32_t width, int32_t height)
{
    auto shape = io.shape();
    shape[1] = height;
    shape[2] = width;
    io.reshape(shape);
}

