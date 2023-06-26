// code adapted from https://github.com/pixop/video-compare
// Copyright (C) 2022 Jon Frydensbjerg (email: jon@pixop.com) 
// licensed under GNU General Public License v2.0
#include "string_utils.h"
#include <numeric>

// Borrowed from https://www.techiedelight.com/implode-a-vector-of-strings-into-a-comma-separated-string-in-cpp/
std::string string_join(std::vector<std::string>& strings, const std::string& delim) {
  // return std::accumulate(strings.begin(), strings.end(), std::string(), [&delim](std::string& x, std::string& y) { return x.empty() ? y : x + delim + y; });
  return std::accumulate(
    std::next(strings.begin()), 
    strings.end(), 
    strings[0], 
    [&delim](std::string a, std::string b) {
        return a.empty() ? b : a + delim + b;
    }
  );
}

