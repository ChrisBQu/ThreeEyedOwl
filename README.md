# ThreeEyedOwl
This uses computer vision techniques, combined with OCR deep learning via Tesseract, to detect and identify Ashes: Rise of the Phoenixborn cards in an image. 

For more information on the game, or to purchase it, please visit: https://www.plaidhatgames.com/board-games/ashes-rise-phoenixborn/

This tool is entirely unaffiliated with the game, and is completely unofficial. I just really love it, and want it to do well. :)

Building the Source code requirements:
----------------------------------
I used Visual Studio, compiling using the C++17 standard. Earlier versions will likely work.
You will need OpenCV installed and linked to your project file. I used OpenCV 4.5.5. Earlier versions will likely work.
You will need to install and link Tesseract to your project file. See: https://github.com/tesseract-ocr/

Before you run
--------------
There is a file, config.hpp. It has some variables in it that allow you to control where output images should be saved, where the card list is located, and where the tesseract model is. Make sure the directories and files you point the tool at exist. A file card "card_list.txt" is provided, as is the tesseract folder that you will want to use.

Input
-----
This has been developed towards an expected input of: pictures that are 1024 x 768 pixels, taken from a "reasonable" distance above the cards, which are on a flat surface. 

Test Cases
----------
Provided is a folder called "test_cases" which contains 5 tests you can run. You will notice that one of the cards is not detected. Indeed, 16 out of the 17 cases provided pass, and one fails. Here's hoping for improvement in the future. :)

