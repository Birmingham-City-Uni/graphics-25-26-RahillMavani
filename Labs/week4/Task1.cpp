// This define is necessary to get the M_PI constant.
#define _USE_MATH_DEFINES
#include <math.h>

#include <iostream>
#include <lodepng.h>
#include "Mesh.hpp"

// ** Introduction to lab 4 **
// It's time to ENTER THE MATRIX
// https://www.youtube.com/watch?v=F1-glc16PHg&t=29s
//
// This week, I'm planning to switch us over to the Eigen linear algebra library
// we've done well so far with our home-baked Vector2.hpp and Vector3.hpp, but as things are 
// getting more complicated, I'd like to save time by using a premade library.
// Eigen is similar to the glm library you use in Game Engine Architecture, but is a larger
// library with a wider range of capabilities.
//
// There are a few things to do this week:
// 1. Get to grips with Eigen, and practice using it to manipulate vectors and matrices
// 2. Implement the matrix-making functions translationMatrix(), scaleMatrix() and rotateYMatrix() below
// 3. Use matrices to translate, rotate and scale the Bunny and Armadillo meshes included in the lab to match
//    the example image.
// 4. Add at least 1 mesh of your own, and transform it into the image. Start to set up your coursework scene.

// This convenience function converts a 3D vector to a 4D homogeneous vector,
// setting the w component to 1.
Eigen::Vector4f vec3ToVec4(const Eigen::Vector3f& v)
{
	Eigen::Vector4f output;
	output << v.x(), v.y(), v.z(), 1.0f;
	return output;
}

// This is the 2D "cross product" function you implemented last week.
float vec2Cross(const Eigen::Vector2f& v0, const Eigen::Vector2f& v1)
{
	return v0.x() * v1.y() - v0.y() * v1.x();
}

void setPixel(std::vector<uint8_t>& image, int x, int y, int width, int height, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255)
{
	int pixelIdx = x + y * width;
	image[pixelIdx * 4 + 0] = r;
	image[pixelIdx * 4 + 1] = g;
	image[pixelIdx * 4 + 2] = b;
	image[pixelIdx * 4 + 3] = a;
}

void drawTriangle(std::vector<uint8_t>& image, int width, int height,
	const Eigen::Vector2f& p0, const Eigen::Vector2f& p1, const Eigen::Vector2f& p2,
	uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255)
{


	// Find a bounding box around the triangle
	int minX, minY, maxX, maxY;
	minX = std::min(std::min(p0.x(), p1.x()), p2.x());
	minY = std::min(std::min(p0.y(), p1.y()), p2.y());
	maxX = std::max(std::max(p0.x(), p1.x()), p2.x());
	maxY = std::max(std::max(p0.y(), p1.y()), p2.y());

	// Constrain it to lie within the image.
	minX = std::min(std::max(minX, 0), width);
	maxX = std::min(std::max(maxX, 0), width);
	minY = std::min(std::max(minY, 0), height);
	maxY = std::min(std::max(maxY, 0), height);

	Eigen::Vector2f edge1 = p1 - p0;
	Eigen::Vector2f edge2 = p2 - p0;
	float triangleArea = 0.5f * vec2Cross(edge2, edge1);
	if (triangleArea < 0) {
		// Triangle is backfacing
		// Exit and quit drawing!
		return;
	}


	for(int x = minX; x <= maxX; ++x) 
		for (int y = minY; y <= maxY; ++y) {
			Eigen::Vector2f p(x, y);

			float a0 = 0.5f * fabsf(vec2Cross(p1 - p2, p - p2));
			float a1 = 0.5f * fabsf(vec2Cross(p0 - p2, p - p2));
			float a2 = 0.5f * fabsf(vec2Cross(p0 - p1, p - p1));

			float b0 = a0 / triangleArea;
			float b1 = a1 / triangleArea;
			float b2 = a2 / triangleArea;

			float sum = b0 + b1 + b2;
			
			if (sum > 1.0001) {
				continue;
			}


			setPixel(image, x, y, width, height, r, g, b, a);
		}


}


void drawMesh(std::vector<unsigned char>& image, const Mesh& mesh, 
	const Eigen::Vector3f& baseColor, const Eigen::Matrix4f& transform, 
	int width, int height)
{
	for (const auto& face : mesh.faces) {
		Eigen::Vector3f
			v0 = mesh.verts[face[0]],
			v1 = mesh.verts[face[1]],
			v2 = mesh.verts[face[2]];

		Eigen::Vector4f tv0, tv1, tv2;

		// *** Your code here ***
		// Transform your vertices by multiplying them with the matrix input to this 
		// function (the "transform" parameter).
		// Save the result in the tv0, tv1, tv2 variables.
		// HINTS:
		// The matrix is 4x4, and the v0, v1, v2 are 3D! You'll need to convert them to 4D 
		// homogeneous vectors first (add a 1 in the w component).
		// You can use the vec3ToVec4 function above to do this.



		tv0 = transform * vec3ToVec4(v0);
		tv1 = transform * vec3ToVec4(v1);
		tv2 = transform * vec3ToVec4(v2);

		Eigen::Vector2f p0(tv0.x() * 250 + width / 2, -tv0.y() * 250 + height / 2);
		Eigen::Vector2f p1(tv1.x() * 250 + width / 2, -tv1.y() * 250 + height / 2);
		Eigen::Vector2f p2(tv2.x() * 250 + width / 2, -tv2.y() * 250 + height / 2);


		Eigen::Vector3f edge1 = tv1.block<3,1>(0,0) - tv0.block<3,1>(0,0);
		Eigen::Vector3f edge2 = tv2.block<3,1>(0,0) - tv0.block<3,1>(0,0);
		Eigen::Vector3f normal = edge1.cross(edge2).normalized();

		float intensity = normal.dot(Eigen::Vector3f(0, 0, 1));
		if (intensity > 0.f) {
			drawTriangle(image, width, height, p0, p1, p2,
				baseColor.x() * intensity * 255,
				baseColor.y() * intensity * 255,
				baseColor.z() * intensity * 255);

		}

	}
}

// ============ TASK 2 =================
// Implement this function that makes a translation matrix
Eigen::Matrix4f translationMatrix(const Eigen::Vector3f& t)
{
	// *** Your code here ***
	Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
	m(0, 3) = t.x();
	m(1, 3) = t.y();
	m(2, 3) = t.z();
	return m;
	return Eigen::Matrix4f::Identity();
}

// Implement this function that makes a uniform scaling matrix
Eigen::Matrix4f scaleMatrix(float s)
{
	// *** Your code here ***
	Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
	m(0, 0) = s;
	m(1, 1) = s;
	m(2, 2) = s;
	return m;
	return Eigen::Matrix4f::Identity();
}

// Implement this function that makes a rotation matrix around the y
// axis
// Hint check: https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
Eigen::Matrix4f rotateYMatrix(float theta)
{
	// *** Your code here ***
	Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
	m(0, 0) = cos(theta);
	m(0, 2) = sin(theta);
	m(2, 0) = -sin(theta);
	m(2, 2) = cos(theta);
	return m;
	return Eigen::Matrix4f::Identity();
}

int main()
{
	// ============ TASK 1 =================
	// The example Eigen code below is in a #pragma region - you can collapse it
	// with the little - sign to hide it if you'd like!
#pragma region LearnEigen

	// Task 1: Getting to Grips with Eigen
	// Let's first look at how to set up and manipulate some vectors and
	// matrices in Eigen.

	// Making vectors
	// You can make a vector in a similar way to our earlier Vector3 and Vector2 classes.
	Eigen::Vector3f v(0, 1, 0);
	// Eigen is actually a heavily templated library - try hovering over Vector3f above, 
	// and you'll see it's actually a typedef (alias) for this:
	Eigen::Matrix<float, 3, 1> v2;
	// In other words, it's a 3x1 float matrix
	// Makes sense, I guess a 3D vector is a 3x1 matrix!
	// You can make matrices of any size this way:
	Eigen::Matrix<float, 6, 4> mySixByFourMatrix;

	// Subtask: Try making a 2D vector of ints using the template <> syntax.
	// Is there a handy typedef for this too?

	// Matrices
	// For the matrix sizes we'll commonly use (3x3 and 4x4) Eigen has typedefs for these too:
	Eigen::Matrix3f myThreeByThreeMatrix;
	Eigen::Matrix4f myFourByFourMatrix;

	// Setting values
	// We've seen you can initialise vectors using a constructor.
	v = Eigen::Vector3f(1, 0, 0);
	// Eigen also has a nice syntax for initialising any matrix:
	v << 1, 0, 0;
	// This does a similar thing, packing the values 1, 0 and 0 into the vector.
	// For matrices especially this is very handy:
	myThreeByThreeMatrix <<
		1, 0, 0,
		0, 1, 0,
		0, 0, 1;
	// Notice this makes an identity matrix, but we could equally have put any values in.
	// There is a shorter way to make an identity matrix:
	Eigen::Matrix3f myIdentityMatrix = Eigen::Matrix3f::Identity();

	// In Eigen, the * will perform matrix multiplication
	// It can multiply two matrices together, or multiply a matrix by a vector.
	Eigen::Matrix3f m = myIdentityMatrix * myIdentityMatrix;
	// We can also easily find the inverse of a matrix:
	myThreeByThreeMatrix <<
		10, 0, 0,
		0, 4, 0,
		0, 0, 1;
	Eigen::Matrix3f myInverse = myThreeByThreeMatrix.inverse();

	// Subtask 3: Try multiplying myThreeByThreeMatrix by myInverse
	// Print out the result.
	// Is it what you would expect?

	// Final advanced tip: the .block<>() function
	// Eigen has a .block method that's super useful for getting or setting a
	// sub-block of a matrix. For example:
	Eigen::Matrix4f myMat4;
	myMat4 <<
		2, 2, 2, 2,
		2, 2, 2, 2,
		2, 2, 2, 2,
		2, 2, 2, 2;
	Eigen::Matrix3f myMat3;
	myMat3 <<
		1, 1, 1,
		1, 1, 1,
		1, 1, 1;
	myMat4.block<3, 3>(0, 0) = myMat3;
	std::cout << "myMat4: " << myMat4 << std::endl;
	// Note the numbers in the triangle brackets say how many rows and columns to get
	// The round brackets give the row, column of the top left corner of the block.
#pragma endregion

	std::string outputFilename = "output.png";

	const int width = 512, height = 512;
	const int nChannels = 4;

	// Setting up an image buffer
	// This std::vector has one 8-bit value for each pixel in each row and column of the image, and
	// for each of the 4 channels (red, green, blue and alpha).
	// Remember 8-bit unsigned values can range from 0 to 255.
	std::vector<uint8_t> imageBuffer(height*width*nChannels);

	// This line sets the memory block occupied by the image to all zeros.
	memset(&imageBuffer[0], 0, width * height * nChannels * sizeof(uint8_t));

	std::string bunnyFilename = "../models/stanford_bunny_simplified.obj";
	std::string dragonFilename = "../models/stanford_dragon_simplified.obj";
	std::string carFilename = "../models/uploads_files_3295091_Lexus+lfa+object.obj";

	Mesh bunnyMesh = loadMeshFile(bunnyFilename);
	Mesh dragonMesh = loadMeshFile(dragonFilename);
	//Mesh carMesh = loadMeshFile(carFilename);


	// ============ TASK 3 =================
	// *** Your Code Here ***
	// These matrices are currently just identity matrices
	// Replace them with your own appropriate matrices to move
	// the bunny and armadillo to match the ones in the example image.
	// TIP: Think about the order of your transforms. Do you want to rotate first,
	//      scale first, or translate first? Does the order matter?



	Eigen::Matrix4f bunnyTransform = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f dragonTransform = Eigen::Matrix4f::Identity();
	//Eigen::Matrix4f carTransform = Eigen::Matrix4f::Identity();

	// Convert 45 degrees to radians for the rotation function
	float angle = 0.0f * (M_PI / 180.0f);

	// Order: Translate * Rotate * Scale
	bunnyTransform = translationMatrix(Eigen::Vector3f(-0.5f, 0, 0)) * rotateYMatrix(angle) * scaleMatrix(0.5f);
	dragonTransform = translationMatrix(Eigen::Vector3f(0.3f, 0.5, 0)) * scaleMatrix(0.8f);
	//carTransform = translationMatrix(Eigen::Vector3f(0.0f, 0, 0)) * scaleMatrix(0.8f);

	// =========== TASK 4 ==============
	// Prepare your own mesh in blender, exporting as OBJ
	// Add code here to load and draw the mesh, transforming it into the scene
	// with an appropriate matrix.

	drawMesh(imageBuffer, bunnyMesh, Eigen::Vector3f(0, 1, 0), bunnyTransform, width, height);
	drawMesh(imageBuffer, dragonMesh, Eigen::Vector3f(0, 1, 1), dragonTransform, width, height);

	//std::vector<float> zBuffer(width* height, std::numeric_limits<float>::max());
	//drawMesh(imageBuffer, carMesh, Eigen::Vector3f(0, 1, 1), carTransform, width, height);

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
