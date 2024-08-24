#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <filesystem>
#include <vector>
#include <string>
#include <Python.h>
#include "NameMatcher.h"
#include "FuzzyWuzzy.hpp"
#include "config.hpp"

// Debugging and testing function
void show_mats(std::vector<cv::Mat> mat_list, std::string label) {
	for (unsigned int i = 0; i < mat_list.size(); i++) {
		cv::imshow(label + " " + std::to_string(i), mat_list[i]);

	}
}

// Helper function to help fill holes in the segmented cards
void fillHoles(cv::Mat& src, cv::Mat dst) {
	cv::Mat tmp = src.clone();
	cv::bitwise_not(src, tmp);
	cv::floodFill(tmp, cv::Point(1, 1), cv::Scalar(0, 0, 0));
	cv::floodFill(tmp, cv::Point(dst.cols - 2, 1), cv::Scalar(0, 0, 0));
	cv::floodFill(tmp, cv::Point(1, dst.rows - 2), cv::Scalar(0, 0, 0));
	cv::floodFill(tmp, cv::Point(dst.cols - 2, dst.rows - 2), cv::Scalar(0, 0, 0));
	dst = (src + tmp);
}



// Apply binary thresholding to the image, and apply processing to it to segment the cards from the surface they are on
void color_to_thresh(cv::Mat & src, cv::Mat & dst) {
	cv::Mat img_bw;
	cv::cvtColor(src, img_bw, cv::ColorConversionCodes::COLOR_BGR2GRAY);
	cv::GaussianBlur(img_bw, img_bw, cv::Size(3, 3), 7);
	cv::Mat canny = img_bw.clone();
	cv::Canny(img_bw, canny, 50, 255);
	cv::imshow("canny", canny);
	cv::Mat elem = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
	cv::dilate(canny, canny, elem, cv::Point(-1, -1), 2);
	cv::erode(canny, canny, elem, cv::Point(-1, -1), 2);
	dst = canny.clone();
	cv::Mat tmp = dst.clone();
	fillHoles(tmp, dst);
	fillHoles(tmp, dst);

	cv::imshow("thresh", dst);

	// Used for screenshotting purposes. Not necessary for the program to work
	cv::imwrite(CONFIG_SAVE_DIRECTORY + "thresh " + std::to_string(saved_file_count++) + ".png", dst);
}


// Using the contours that were found, find all of the objects that could cards
std::vector<cv::RotatedRect> get_candidate_rects(std::vector<std::vector<cv::Point>> contours, std::vector<cv::Vec4i> hierarchy) {
	std::vector<cv::RotatedRect> candidate_rects;
	for (unsigned int i = 0; i < contours.size(); i++) {
		const cv::RotatedRect boundingRect = cv::minAreaRect(contours[i]);
		bool candidate = true;
		if (boundingRect.size.area() < CONFIG_CARD_SIZE_THRESHOLD) { candidate = false; }
		float ratio = std::max(boundingRect.size.width, boundingRect.size.height) / std::min(boundingRect.size.width, boundingRect.size.height);

		if (candidate) {
			cv::Point2f four_corners[4];
			boundingRect.points(four_corners);
			candidate_rects.push_back(boundingRect);
		}
	}
	return candidate_rects;
}



// Apply perspective transformation to each of the rotated rectangles in the image to get those portions
// of the image as upright rectangles. We consider four variants of it: one for each 90-degree rotation
std::vector<cv::Mat> get_warped_candidate_mats(cv::Mat whole_img, std::vector<cv::RotatedRect> candidates) {
	std::vector<cv::Mat> return_vec;
	cv::Mat img_copy = whole_img.clone();
	for (unsigned int i = 0; i < candidates.size(); i++) {
		cv::Point2f four_corners[4];
		cv::Point2f warped_corners[4];
		float width = std::min(candidates[i].size.width, candidates[i].size.height) - 1;
		float height = std::max(candidates[i].size.width, candidates[i].size.height) - 1;
		warped_corners[0] = { 0, height};
		warped_corners[1] = { 0, 0 };
		warped_corners[2] = { width, 0 };
		warped_corners[3] = { width, height};

		candidates[i].points(four_corners);
		for (unsigned int j = 0; j < 4; j++) {
			cv::Point2f tmp = four_corners[0];
			four_corners[0] = four_corners[1];
			four_corners[1] = four_corners[2];
			four_corners[2] = four_corners[3];
			four_corners[3] = tmp;
			cv::Mat rotated_img = whole_img.clone();
			cv::Mat Transform_Matrix = cv::getPerspectiveTransform(four_corners, warped_corners);
			cv::warpPerspective(whole_img, rotated_img, Transform_Matrix, cv::Size((int)width, (int)height));
			return_vec.push_back(rotated_img);
		}

	}
	return return_vec;
}

// For each of the candidates we have, we will examine their text to see if it matches a card in our loaded card list
void seekMatch(std::vector<cv::Mat> warped_mats, std::vector<cv::RotatedRect> rects, cv::Mat frame) {

	for (unsigned int i = 0; i < rects.size(); i++) {
		cv::Point2f four_corners[4];
		rects[i].points(four_corners);

		cv::Scalar color(255, 0, 0);

		std::string card_name;
		card_name = seek_name(warped_mats[i * 4]);
		if (card_name.size() == 0) { card_name = seek_name(warped_mats[i * 4 + 1]); }
		if (card_name.size() == 0) { card_name = seek_name(warped_mats[i * 4 + 2]); }
		if (card_name.size() == 0) { card_name = seek_name(warped_mats[i * 4 + 3]); }
	
		if (card_name.size() != 0) {
			for (unsigned int i = 0; i < 4; i++) { cv::line(frame, four_corners[i], four_corners[(i + 1) % 4], color, 3); }
			cv::putText(frame, card_name, four_corners[0], cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
		}

	}


}

// Removes smaller contours inside larger ones, and utilizes the convex hull of contours, to avoid contours bleeding into background
// Also remove especially large and especially small contours
void cleanContours(const std::vector<std::vector<cv::Point>>& inputContours, std::vector<std::vector<cv::Point>>& outputContours, double minArea) {
	std::vector<bool> keepContour(inputContours.size(), true);

	// Step 1: Remove contours inside of another contour
	for (size_t i = 0; i < inputContours.size(); ++i) {
		// Check contour area
		double area = cv::contourArea(inputContours[i]);
		std::cout << "Area: " << area << std::endl;

		if (area < minArea) {
			keepContour[i] = false;
			continue;
		}

		for (size_t j = 0; j < inputContours.size(); ++j) {
			if (i != j && keepContour[j]) {
				// Check if contour j is inside contour i
				if (cv::pointPolygonTest(inputContours[i], inputContours[j][0], false) >= 0) {
					keepContour[j] = false;
				}
			}
		}
	}

	// Step 2: Generate the convex hull of the remaining contours
	for (size_t i = 0; i < inputContours.size(); ++i) {
		if (keepContour[i]) {
			std::vector<cv::Point> hull;
			cv::convexHull(inputContours[i], hull);
			outputContours.push_back(hull);
		}
	}
}


// Test driver
int main()
{
	load_cardlist(CONFIG_CARD_LIST_FILE);

	for (unsigned int g = 0; g < CONFIG_f_names.size(); g++) {

		std::string image_path = CONFIG_TEST_DIRECTORY + CONFIG_f_names[g];

		cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

		cv::Mat thresh;
		color_to_thresh(img, thresh);

		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;

		cv::Mat contour_mat = img.clone();
		cv::findContours(thresh, contours, hierarchy, cv::RetrievalModes::RETR_TREE, cv::ContourApproximationModes::CHAIN_APPROX_TC89_KCOS);

		std::vector<std::vector<cv::Point>> cleaned_contours;
		cleanContours(contours, cleaned_contours, 40000);

		cv::drawContours(contour_mat, cleaned_contours, -1, cv::Scalar(rand() % 255, rand() % 255, rand() % 255), 3);


		cv::imshow("contours", contour_mat);

		// Save the contour images
		//cv::imwrite(CONFIG_SAVE_DIRECTORY + "contours " + std::to_string(saved_file_count++) + ".png", contour_mat);

		std::vector<cv::RotatedRect> candidate_rects = get_candidate_rects(cleaned_contours, hierarchy);
		std::vector<cv::Mat> warped_mats = get_warped_candidate_mats(img, candidate_rects);

		seekMatch(warped_mats, candidate_rects, img);
		cv::imshow("Output", img);

		// Save the output
		//cv::imwrite(CONFIG_SAVE_DIRECTORY + "output " + std::to_string(saved_file_count++) + ".png", img);


		cv::waitKey();
		cv::destroyAllWindows();
	}


	return 0;
}