#include "NameMatcher.h"
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "fuzzywuzzy.hpp"
#include <cctype>
#include "config.hpp"

std::string rip_text(cv::Mat img);
std::string compare_against_list(std::string s);

// Crop the image to the rough location of where the title text would be (if present),
// and also smooth and binarize the image. This will prepare the image for a tesseract search
cv::Mat thresh_text(cv::Mat input) {
	cv::Mat tmp = input.clone();
	tmp.convertTo(tmp, -1, 0.3, 50);
	cv::resize(tmp, tmp, cv::Size(0, 0), 4.0, 4.0, cv::InterpolationFlags::INTER_AREA);
	tmp = tmp(cv::Rect(0, 0, tmp.cols, (int)(tmp.rows * 0.135)));
	cv::cvtColor(tmp, tmp, cv::ColorConversionCodes::COLOR_BGR2GRAY);
	cv::Mat tmp2 = tmp.clone();
	cv::blur(tmp2, tmp, cv::Size(5, 5));
	cv::threshold(tmp, tmp, 100, 255, cv::THRESH_BINARY);
	return tmp;
}


// A list of card names to be loaded from a file
std::vector<std::string> CardList;

// Remove as much garbage as we can from a text string, to make searches against the card list more accurate
void clean_string(std::string& ref) {
	std::string built = "";
	std::string token = "";
	int token_count = 0;
	for (auto& x : ref) {
		token += x;  
		if (x == ' ') {
			token_count++;
			if (token.size() > 2 && token_count<6) { built += token; }
			token = "";
		}
	}
	if (token_count < 6) { built += token; }
	ref = built;
}

// Loads a vector of card names from a text file, with one line per card name
void load_cardlist(std::string filename) {
	try {
		std::ifstream input(filename);
		for (std::string line; getline(input, line); ) {
			for (auto& x : line) { x = toupper(x); } 
			CardList.push_back(line);
		}
	}
	catch (int exception) { std::cout << "Error: could not load the card list from file." << std::endl; return; }
	std::cout << "Loaded card list." << std::endl;
}

// The fuzzywuzzy partial ratio comparison will be made between a string, and each card in the loaded list,
// to find the string that matches it best
std::string compare_against_list(std::string s) {
	std::string retval = "";
	clean_string(s);
	if (s.size() < 5) { return retval; }
	unsigned int index = 0;
	unsigned int best_score = 0;
	for (unsigned int i = 0; i < CardList.size(); i++) {
		unsigned int new_score = fuzz::token_sort_ratio(CardList[i], s) + fuzz::weighted_ratio(CardList[i], s) + fuzz::ratio(CardList[i], s);
		if (new_score > best_score) {
			best_score = new_score;
			index = i;
		}
		// The following line can be uncommented for debugging purposes
		//if (new_score >= SCORE_THRESHOLD) { std::cout << "Candidate: " << CardList[i] << " ( " << new_score << " " << std::endl; }
	}
	if (best_score >= CONFIG_TESSERACT_SCORE_THRESHOLD) { retval = CardList[index];  }
	// The following line can be uncommented for debugging purposes
	//std::cout << "Labeled as " << retval << " with score of " << best_score << std::endl;
	return retval;
}


// This uses tesseract to rip text from a mat passed to it, and return it as a string
std::string rip_text(cv::Mat img) {

	cv::Mat check = img.clone();

	// Save the candidate image
	//cv::imwrite(CONFIG_SAVE_DIRECTORY + "candidate " + std::to_string(saved_file_count++) + ".png", check);

	check = thresh_text(check);
	cv::resize(check, check, cv::Size(0, 0), 3, 3);
	tesseract::TessBaseAPI tess = tesseract::TessBaseAPI();
	if (tess.Init(CONFIG_TESSERACT_DATA_PATH.c_str(), "eng")) {
		fprintf(stderr, "Could not initialize tesseract.\n");
		exit(1);
	}
	tess.SetImage((uchar*)check.data, check.size().width, check.size().height, check.channels(), check.step1());
	tess.SetVariable("user_defined_dpi", "300");
	tess.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ '");
	tess.SetVariable("user_words_file", "eng.user-words");
	tess.Recognize(0);
	std::string out = std::string(tess.GetUTF8Text());
	std::replace(out.begin(), out.end(), '\n', ' ');

	// Save the binarized image of the area cropped around where the text would be
	//cv::imwrite(CONFIG_SAVE_DIRECTORY + "binarized_text " + std::to_string(saved_file_count++) + ".png", check);
	return out;
}

// Expose all of the above functionality to the main program
std::string seek_name(cv::Mat m) {
	std::string f = rip_text(m);
	return compare_against_list(f);
}

