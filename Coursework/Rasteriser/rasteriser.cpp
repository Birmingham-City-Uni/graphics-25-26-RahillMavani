#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <lodepng.h>

#include "Image.hpp"
#include "LinAlg.hpp"
#include "Light.hpp"
#include "Mesh.hpp"
#include "Shading.hpp"


enum ShadingMode {
	PHONG,
	BLINN_PHONG
};


struct Triangle {
	std::array<Eigen::Vector3f, 3> screen; // Coordinates of the triangle in screen space.
	std::array<Eigen::Vector3f, 3> verts; // Vertices of the triangle in world space.
	std::array<Eigen::Vector3f, 3> cam; // Vertices of the triangle in camera space.
	std::array<Eigen::Vector3f, 3> norms; // Normals of the triangle corners in world space.
	std::array<Eigen::Vector2f, 3> texs; // Texture coordinates of the triangle corners.
};


Eigen::Matrix4f projectionMatrix(int height, int width, float horzFov = 70.f * M_PI / 180.f, float zFar = 10.f, float zNear = 0.1f)
{
	float vertFov = horzFov * float(height) / width;
	Eigen::Matrix4f projection;
	projection <<
		1.0f / tanf(0.5f * horzFov), 0, 0, 0,
		0, 1.0f / tanf(0.5f * vertFov), 0, 0,
		0, 0, zFar / (zFar - zNear), -zFar * zNear / (zFar - zNear),
		0, 0, 1, 0;
	return projection;
}

void findScreenBoundingBox(const Triangle& t, int width, int height, int& minX, int& minY, int& maxX, int& maxY)
{
	// Find a bounding box around the triangle
	minX = std::min(std::min(t.screen[0].x(), t.screen[1].x()), t.screen[2].x());
	minY = std::min(std::min(t.screen[0].y(), t.screen[1].y()), t.screen[2].y());
	maxX = std::max(std::max(t.screen[0].x(), t.screen[1].x()), t.screen[2].x());
	maxY = std::max(std::max(t.screen[0].y(), t.screen[1].y()), t.screen[2].y());

	// Constrain it to lie within the image.
	minX = std::min(std::max(minX, 0), width - 1);
	maxX = std::min(std::max(maxX, 0), width - 1);
	minY = std::min(std::max(minY, 0), height - 1);
	maxY = std::min(std::max(maxY, 0), height - 1);
}


void drawTriangle(std::vector<uint8_t>& image, int width, int height,
	std::vector<float>& zBuffer,
	const Triangle& t,
	const std::vector<std::unique_ptr<Light>>& lights,
	const Eigen::Vector3f& albedo, const Eigen::Vector3f& specularColor,
	float specularExponent,
	ShadingMode shadingMode,
	const Eigen::Vector3f& camWorldPos)
{
	int minX, minY, maxX, maxY;
	findScreenBoundingBox(t, width, height, minX, minY, maxX, maxY);

	Eigen::Vector2f edge1 = v2(t.screen[2] - t.screen[0]);
	Eigen::Vector2f edge2 = v2(t.screen[1] - t.screen[0]);
	float triangleArea = 0.5f * vec2Cross(edge2, edge1);
	if (triangleArea < 0) {
		// Triangle is backfacing
		// Exit and quit drawing!
		return;
	}

	for (int x = minX; x <= maxX; ++x)
		for (int y = minY; y <= maxY; ++y) {
			Eigen::Vector2f p(x, y);

			// Find sub-triangle areas
			float a0 = 0.5f * fabsf(vec2Cross(v2(t.screen[1]) - v2(t.screen[2]), p - v2(t.screen[2])));
			float a1 = 0.5f * fabsf(vec2Cross(v2(t.screen[0]) - v2(t.screen[2]), p - v2(t.screen[2])));
			float a2 = 0.5f * fabsf(vec2Cross(v2(t.screen[0]) - v2(t.screen[1]), p - v2(t.screen[1])));

			// find barycentrics
			float b0 = a0 / triangleArea;
			float b1 = a1 / triangleArea;
			float b2 = a2 / triangleArea;

			// If outside triangle, exit early
			float sum = b0 + b1 + b2;
			if (sum > 1.0001) {
				continue;
			}

			// Get the depths from the camera-space position of the 3 corners.
			float depth0 = t.cam[0].z(), depth1 = t.cam[1].z(), depth2 = t.cam[2].z();

			// Work out the depth at the point P
			float dP = b0 * (1.0f / depth0) + b1 * (1.0f / depth1) + b2 * (1.0f / depth2);
			float depthP = 1.0f / dP;

			// Interpolate to find the world-space position of this pixel (correct this version to be 
			// perspective-correct).
			// Don't forget to multiply by depthP!
			Eigen::Vector3f worldP =
				(b0 * (t.verts[0] / depth0) +
					b1 * (t.verts[1] / depth1) +
					b2 * (t.verts[2] / depth2)) * depthP;

			// Interpolate to find the normal of this pixel (correct this version to be 
			// perspective-correct).
			// Tip: you don't need to worry about multiplying by depthP - you'll normalise this anyway!
			Eigen::Vector3f normP =
				(b0 * (t.norms[0] / depth0) +
					b1 * (t.norms[1] / depth1) +
					b2 * (t.norms[2] / depth2)) * depthP;
			normP.normalize();

			// Interpolate to find the correct clip-space depth (correct this version to be perspective-correct)
			// This won't make too much of a difference in this case, but technically this version does use slightly
			// incorrect depths.
			float depth = (b0 * (t.screen[0].z() / depth0) +
				b1 * (t.screen[1].z() / depth1) +
				b2 * (t.screen[2].z() / depth2)) * depthP;
			// *** END YOUR CODE ***

			int depthIdx = static_cast<int>(p.x()) + static_cast<int>(p.y()) * width;
			if (depth > zBuffer[depthIdx]) continue;
			zBuffer[depthIdx] = depth;


			// Work out colour at this position.
			Eigen::Vector3f color = Eigen::Vector3f::Zero();

			Eigen::Vector3f viewDir = (camWorldPos - worldP).normalized();

			// Iterate over lights, and sum to find colour.
			for (auto& light : lights) {

				// Work out the contribution from this light source, and add it to the color variable.

				// Work out the intensity of this light source, at the point worldP.
				Eigen::Vector3f lightIntensity = light->getIntensityAt(worldP);

				// We only need to do the following if the light isn't an ambient light.
				if (light->getType() != Light::Type::AMBIENT) {
					Eigen::Vector3f incomingLightDir = light->getDirection(worldP);

					float specularTerm;
					if (shadingMode == ShadingMode::PHONG) {
						specularTerm = phongSpecularTerm(incomingLightDir, normP, viewDir, specularExponent);
					}
					else {
						specularTerm = blinnPhongSpecularTerm(incomingLightDir, normP, viewDir, specularExponent);
					}

					Eigen::Vector3f specularOut = specularColor * specularTerm;
					specularOut = coeffWiseMultiply(specularOut, lightIntensity);

					// Take the dot product of the normal with the light direction.
					float dotProd = normP.dot(-incomingLightDir);

					// We don't want negative light - if dot product less than 0, set it to 0.
					dotProd = std::max(dotProd, 0.0f);

					// Multiply the light intensity by the dot product.
					Eigen::Vector3f diffuseOut = lightIntensity * dotProd;
					diffuseOut = coeffWiseMultiply(diffuseOut, albedo);

					color += specularOut;
					//color += diffuseOut;
					//color = (incomingLightDir + Eigen::Vector3f::Ones()) / 2;
				}
				else {
					// Light is ambient - just multiply light intensity with albedo.
					color += coeffWiseMultiply(lightIntensity, albedo);
				}
			}

			Color c;
			// Gamma-correcting colours.
			c.r = std::min(powf(color.x(), 1 / 2.2f), 1.0f) * 255;
			c.g = std::min(powf(color.y(), 1 / 2.2f), 1.0f) * 255;
			c.b = std::min(powf(color.z(), 1 / 2.2f), 1.0f) * 255;

			c.a = 255;

			setPixel(image, x, y, width, height, c);
		}
}



void drawMesh(std::vector<unsigned char>& image,
	std::vector<float>& zBuffer,
	const Mesh& mesh,
	const Eigen::Vector3f& albedo, const Eigen::Vector3f& specularColor,
	float specularExponent,
	ShadingMode shadingMode,
	const Eigen::Vector3f& camWorldPos,
	const Eigen::Matrix4f& modelToWorld,
	const Eigen::Matrix4f& worldToCam,
	const Eigen::Matrix4f& camToClip,
	const std::vector<std::unique_ptr<Light>>& lights,
	int width, int height)
{
	for (int i = 0; i < mesh.vFaces.size(); ++i) {
		Eigen::Vector3f
			v0 = mesh.verts[mesh.vFaces[i][0]],
			v1 = mesh.verts[mesh.vFaces[i][1]],
			v2 = mesh.verts[mesh.vFaces[i][2]];
		Eigen::Vector3f
			n0 = mesh.norms[mesh.nFaces[i][0]],
			n1 = mesh.norms[mesh.nFaces[i][1]],
			n2 = mesh.norms[mesh.nFaces[i][2]];

		Triangle t;
		t.verts[0] = (modelToWorld * vec3ToVec4(v0)).block<3, 1>(0, 0);
		t.verts[1] = (modelToWorld * vec3ToVec4(v1)).block<3, 1>(0, 0);
		t.verts[2] = (modelToWorld * vec3ToVec4(v2)).block<3, 1>(0, 0);

		t.cam[0] = (worldToCam * modelToWorld * vec3ToVec4(v0)).block<3, 1>(0, 0);
		t.cam[1] = (worldToCam * modelToWorld * vec3ToVec4(v1)).block<3, 1>(0, 0);
		t.cam[2] = (worldToCam * modelToWorld * vec3ToVec4(v2)).block<3, 1>(0, 0);

		// Work out the clip space coordinates, by multiplying by worldToClip and doing the 
		// perspective divide.
		Eigen::Vector4f vClip0 = camToClip * worldToCam * modelToWorld * vec3ToVec4(v0);
		vClip0 /= vClip0.w();
		Eigen::Vector4f vClip1 = camToClip * worldToCam * modelToWorld * vec3ToVec4(v1);
		vClip1 /= vClip1.w();
		Eigen::Vector4f vClip2 = camToClip * worldToCam * modelToWorld * vec3ToVec4(v2);
		vClip2 /= vClip2.w();

		// Check that all 3 vertices are in the clip box (-1 to 1 in x, y and z) and if not,
		// skip drawing this triangle.
		if (outsideClipBox(vClip0) || outsideClipBox(vClip1) || outsideClipBox(vClip2)) continue;

		// Work out the screen space coordinates based on the image height and width.
		t.screen[0] = Eigen::Vector3f((vClip0.x() + 1.0f) * width / 2, (-vClip0.y() + 1.0f) * height / 2, vClip0.z());
		t.screen[1] = Eigen::Vector3f((vClip1.x() + 1.0f) * width / 2, (-vClip1.y() + 1.0f) * height / 2, vClip1.z());
		t.screen[2] = Eigen::Vector3f((vClip2.x() + 1.0f) * width / 2, (-vClip2.y() + 1.0f) * height / 2, vClip2.z());

		// transform the normals (using the inverse transpose of the upper 3x3 block)
		t.norms[0] = (modelToWorld.block<3, 3>(0, 0).inverse().transpose() * n0).normalized();
		t.norms[1] = (modelToWorld.block<3, 3>(0, 0).inverse().transpose() * n1).normalized();
		t.norms[2] = (modelToWorld.block<3, 3>(0, 0).inverse().transpose() * n2).normalized();

		t.texs[0] = mesh.texs[mesh.tFaces[i][0]];
		t.texs[1] = mesh.texs[mesh.tFaces[i][1]];
		t.texs[2] = mesh.texs[mesh.tFaces[i][2]];

		drawTriangle(image, width, height, zBuffer, t, lights, albedo, specularColor, specularExponent, shadingMode, camWorldPos);
	}
}






int main()
{
	std::string outputFilename = "lexus_render.png";

	// Your requested resolution!
	const int width = 1920, height = 1080;
	const int nChannels = 4;

	std::vector<uint8_t> imageBuffer(height * width * nChannels);
	std::vector<float> zBuffer(height * width);

	//clear the screen to Black and reset Z-Buffer
	Color bg{ 50, 50, 50, 255 };
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			setPixel(imageBuffer, x, y, width, height, bg);
			zBuffer[x + y * width] = 999999.0f;
		}
	}

	//load the Car
	std::cout << "Loading Lexus Model..." << std::endl;
	Mesh carMesh = loadMeshFile("../lexus.obj"); // <-- Make sure lexus.obj is one folder up from the build folder!

	//set up the Camera Lens
	Eigen::Matrix4f projection = projectionMatrix(height, width);

	//move the camera back 8 units
	Eigen::Matrix4f cameraToWorld = translationMatrix(Eigen::Vector3f(0.0f, 0.0f, -8.0f));
	Eigen::Matrix4f worldToCamera = cameraToWorld.inverse();
	Eigen::Vector3f camWorldPos = (cameraToWorld * Eigen::Vector4f(0, 0, 0, 1)).block<3, 1>(0, 0);

	//set up the Lights
	std::vector<std::unique_ptr<Light>> lights;
	lights.emplace_back(new AmbientLight(Eigen::Vector3f(0.2f, 0.2f, 0.2f)));
	lights.emplace_back(new DirectionalLight(Eigen::Vector3f(0.9f, 0.9f, 0.9f), Eigen::Vector3f(0.5f, -1.0f, 1.f)));

	//position the Car
	Eigen::Matrix4f carTransform = translationMatrix(Eigen::Vector3f(0.0f, -1.0f, 0.0f)) * rotateYMatrix(M_PI_4) * scaleMatrix(100.0f);

	//RENDER!
	std::cout << "Rendering Engine Starting..." << std::endl;
	drawMesh(imageBuffer, zBuffer, carMesh,
		Eigen::Vector3f(0.8f, 0.1f, 0.1f), // Base color (Red)
		Eigen::Vector3f(1.0f, 1.0f, 1.0f), // Specular highlight color (White)
		64.f,                              // Shininess (High = sharp mirror reflection)
		BLINN_PHONG,                       // Shading mode
		camWorldPos, carTransform, worldToCamera, projection, lights, width, height);

	//save the Image
	std::cout << "Saving image..." << std::endl;
	int errorCode = lodepng::encode(outputFilename, imageBuffer, width, height);
	if (errorCode) {
		std::cout << "lodepng error encoding image: " << lodepng_error_text(errorCode) << std::endl;
		return errorCode;
	}

	std::cout << "Render Complete! Check lexus_render.png" << std::endl;
	return 0;
}