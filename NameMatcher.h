#pragma once
#ifndef NAME_MATCHER_H
#define NAME_MATCHER_H

#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/img_hash.hpp>

// The major function here, being exposed to the main program
void load_cardlist(std::string filename);
std::string seek_name(cv::Mat m);

#endif