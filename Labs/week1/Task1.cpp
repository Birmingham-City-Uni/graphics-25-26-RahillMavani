#include <iostream>
#include <lodepng.h>

int setPixel(std::vector<unsigned char>& buffer, int x, int y, int width, int r, int g, int b, int a)
{
	int nChannels = 4;
	int pixelIdex = (y * width + x) * nChannels;

	buffer[pixelIdex + 0] = r;
	buffer[pixelIdex + 1] = g;
	buffer[pixelIdex + 2] = b;
	buffer[pixelIdex + 3] = a;

	return 0;
}

int main()
{
	std::string outputFilename = "output.png";

	const int width = 1920, height = 1080;
	const int nChannels = 4;

	// Setting up an image buffer
	// This std::vector has one 8-bit value for each pixel in each row and column of the image, and
	// for each of the 4 channels (red, green, blue and alpha).
	// Remember 8-bit unsigned values can range from 0 to 255.
	std::vector<uint8_t> imageBuffer(height*width*nChannels);

	// This for loop sets all the pixels of the image to a cyan colour. 
	for(int y = 0; y < height; ++y) 
		for (int x = 0; x < width; ++x) {
			
			if (((x - 960) * (x - 960) + (y - 540) * (y - 540)) < 40000)
			{
				setPixel(imageBuffer, x, y, width, 255, 255, 255, 255);
			}
			else
			{
				setPixel(imageBuffer, x, y, width, 20, 40, 20, 255);

				if (y > 540)
				{
					
					setPixel(imageBuffer, x, y, width, 212, 175, 55, 255);
				}
			}
			
			
		}

	/// *** Lab Tasks ***
	// * Task 1: Try adapting the code above to set the lower half of the image to be a green colour.
	

	
	// * Task 2: Doing the maths above to work out indices is a bit annoying! Write your own setPixel function.
	//           This should take x and y coordinates as input, and red, green, blue and alpha values.
	//           Remember to pass in your imageBuffer. Should it be passed in by reference or by value? Should
	//           the reference be const?
	//           We will use this setPixel function to build our rasteriser in the upcoming labs.
	//			 Test your setPixel function by setting pixels in your image to different colours.



	// * Optional Task 3: Use your setPixel function to draw a circle in the centre of the image. Remember a point is
	//           in a circle if sqrt((x - x_0)^2 + (y - y_0)^2) < radius (here x_0, y_0 are the coordinates at the middle of 
	//           the circle). 
	//           Hint - use a similar for loop to the one above, and add an if statement to check if the current
	//           pixel lies in the circle.
	//           Try modifying the order you draw each component in. If you draw the circle before setting the lower 
	//           part of the image to be green, how does this modify the image?
	

	// * Optional Task 4: Work out how good the compression ratio of the saved PNG image is. PNG images
	//           use *lossless* compression, where all the pixel values of the original image are preserved.
	//           To work out the compression ratio, compare the size of the saved image to the memory
	//           occupied by the image buffer (this is based on the width, height and number of channels).
	//           Try setting the pixels to random values (use rand() and the % operator). What is the 
	//           compression ratio now, and why do you think this is?


	// *** Encoding image data ***
	// PNG files are compressed to save storage space. 
	// The lodepng::encode function applies this compression to the image buffer and saves the result 
	// to the filename given.
	int errorCode;
	errorCode = lodepng::encode(outputFilename, imageBuffer, width, height);
	if (errorCode) { // check the error code, in case an error occurred.
		std::cout << "lodepng error encoding image: " << lodepng_error_text(errorCode) << std::endl;
		return errorCode;
	}

	return 0;
}

