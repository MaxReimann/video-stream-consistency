#pragma once

#include <string>
#include <vector>

void ReadFlowFile(std::vector<float>& flow, int& w, int& h, std::string filename);
void WriteFlowFile(const std::vector<float>& flow, int W, int H, std::string filename);
