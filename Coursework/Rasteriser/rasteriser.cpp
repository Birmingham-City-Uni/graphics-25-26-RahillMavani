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


Eigen::Matrix4f projectionMatrix(int height, int width, float horzFov = 70.f * M_PI / 180.f, float zFar = 10000.f, float zNear = 0.1f)
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
	const std::vector<unsigned char>& textureData, unsigned texWidth, unsigned texHeight,
	ShadingMode shadingMode,
	const Eigen::Vector3f& camWorldPos,
	float tileFactor = 1.0f,
	float alpha = 1.0f)

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

			//WEEK 6 TEXTURE MAPPING
			Eigen::Vector3f finalAlbedo = albedo; // Default to the solid math color


			if (!textureData.empty()) {
				//perspective-correct UV interpolation
				Eigen::Vector2f texP =
					(b0 * (t.texs[0] / depth0) +
						b1 * (t.texs[1] / depth1) +
						b2 * (t.texs[2] / depth2)) * depthP;

				//wrap the UVs (prevents crashing if UVs go outside 0 to 1)
				float u = (texP.x() * tileFactor) - floor(texP.x() * tileFactor);
				float v = (texP.y() * tileFactor) - floor(texP.y() * tileFactor);

				//convert UV to actual PNG pixel coordinates (Invert V because images load top-down)
				int tx = static_cast<int>(u * (texWidth - 1));
				int ty = static_cast<int>((1.0f - v) * (texHeight - 1));

				//fetch the RGB color from the Substance Painter texture
				int texIdx = (ty * texWidth + tx) * 4;
				finalAlbedo = Eigen::Vector3f(
					textureData[texIdx] / 255.0f,
					textureData[texIdx + 1] / 255.0f,
					textureData[texIdx + 2] / 255.0f
				);
			}
			
		//calculate how bright this pixel is(Luminance)
			float pixelBrightness = (finalAlbedo.x() * 0.299f) + (finalAlbedo.y() * 0.587f) + (finalAlbedo.z() * 0.114f);

		//texture's brightness
		Eigen::Vector3f mappedSpecularColor = specularColor * pixelBrightness;

			// Iterate over lights, and sum to find colour.
			for (auto& light : lights) {

				Eigen::Vector3f lightIntensity = light->getIntensityAt(worldP);

				if (light->getType() != Light::Type::AMBIENT) {
					Eigen::Vector3f incomingLightDir = light->getDirection(worldP);

					float specularTerm;
					if (shadingMode == ShadingMode::PHONG) {
						specularTerm = phongSpecularTerm(incomingLightDir, normP, viewDir, specularExponent);
					}
					else {
						specularTerm = blinnPhongSpecularTerm(incomingLightDir, normP, viewDir, specularExponent);
					}

					Eigen::Vector3f specularOut = mappedSpecularColor * specularTerm;
					specularOut = coeffWiseMultiply(specularOut, lightIntensity);

					float dotProd = normP.dot(-incomingLightDir);
					dotProd = std::max(dotProd, 0.0f);

				
					Eigen::Vector3f diffuseOut = lightIntensity * dotProd;
					diffuseOut = coeffWiseMultiply(diffuseOut, finalAlbedo);

					color += specularOut;
					color += diffuseOut;
				}
				else {
					color += coeffWiseMultiply(lightIntensity, finalAlbedo);
				}
			}

			//calculate the new pixel color
			float outR = std::min(powf(color.x(), 1 / 2.2f), 1.0f) * 255.0f;
			float outG = std::min(powf(color.y(), 1 / 2.2f), 1.0f) * 255.0f;
			float outB = std::min(powf(color.z(), 1 / 2.2f), 1.0f) * 255.0f;

			//fnid the existing pixel in the image buffer (the background)
			int pIdx = (y * width + x) * 4;
			float bgR = image[pIdx + 0];
			float bgG = image[pIdx + 1];
			float bgB = image[pIdx + 2];

			Color c;

			//mix the new color and background color
			c.r = static_cast<uint8_t>((outR * alpha) + (bgR * (1.0f - alpha)));
			c.g = static_cast<uint8_t>((outG * alpha) + (bgG * (1.0f - alpha)));
			c.b = static_cast<uint8_t>((outB * alpha) + (bgB * (1.0f - alpha)));
			c.a = 255;

			setPixel(image, x, y, width, height, c);
		}
}



void drawMesh(std::vector<unsigned char>& image,
	std::vector<float>& zBuffer,
	const Mesh& mesh,
	const Eigen::Vector3f& albedo, const Eigen::Vector3f& specularColor,
	float specularExponent,
	const std::vector<unsigned char>& textureData, unsigned texWidth, unsigned texHeight,
	ShadingMode shadingMode,
	const Eigen::Vector3f& camWorldPos,
	const Eigen::Matrix4f& modelToWorld,
	const Eigen::Matrix4f& worldToCam,
	const Eigen::Matrix4f& camToClip,
	const std::vector<std::unique_ptr<Light>>& lights,
	int width, int height,
	float tileFactor = 1.0f,
	float alpha = 1.0f)
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

		drawTriangle(image, width, height, zBuffer, t, lights, albedo, specularColor, specularExponent, textureData, texWidth, texHeight, shadingMode, camWorldPos, tileFactor, alpha);
	}
}






int main()
{
	std::string outputFilename = "lexus_render.png";

	//ANTI-ALIASING
	const int finalWidth = 1920, finalHeight = 1080;
	const int aaMultiplier = 2; //2x SSAA

	const int renderWidth = finalWidth * aaMultiplier;
	const int renderHeight = finalHeight * aaMultiplier;
	const int nChannels = 4;

	//internal buffers
	std::vector<uint8_t> renderBuffer(renderHeight * renderWidth * nChannels);
	std::vector<float> zBuffer(renderHeight * renderWidth);

	//clear the high-res screen to Dark Grey and reset Z-Buffer
	Color bg{ 50, 50, 50, 255 };
	for (int y = 0; y < renderHeight; ++y) {
		for (int x = 0; x < renderWidth; ++x) {
			setPixel(renderBuffer, x, y, renderWidth, renderHeight, bg);
			zBuffer[x + y * renderWidth] = 999999.0f;
		}
	}

	//load the Car
	std::cout << "Loading Lexus Model..." << std::endl;
	Mesh carMesh = loadMeshFile("../lexus_body.obj");
	Mesh car2Mesh = loadMeshFile("../lexus_wheels.obj");
	Mesh car3Mesh = loadMeshFile("../lexus_calliper.obj");
	Mesh car4Mesh = loadMeshFile("../lexus_calliper_2.obj");
	Mesh car5Mesh = loadMeshFile("../lexus_glass.obj");
	Mesh car6Mesh = loadMeshFile("../lexus_glass_2.obj");
	Mesh car7Mesh = loadMeshFile("../lexus_grill.obj");
	Mesh car8Mesh = loadMeshFile("../lexus_grill_2.obj");
	Mesh roadMesh = loadMeshFile("../road2.obj");

	//TEXTURE LOADING
	std::cout << "Loading Textures..." << std::endl;
	std::vector<uint8_t> bodyTex, glassTex, glass_2Tex, grillTex, grill_2Tex, wheelsTex, calliperTex, roadTex;

	//give each texture its own width and height variables
	unsigned w1, h1, w2, h2, w3, h3, w4, h4, w5, h5, w6, h6, w7, h7, w8, h8;

	lodepng::decode(bodyTex, w1, h1, "../lexus_body.png");
	lodepng::decode(glassTex, w2, h2, "../lexus_glass.png");
	lodepng::decode(wheelsTex, w3, h3, "../lexus_wheels.png");
	lodepng::decode(calliperTex, w4, h4, "../lexus_calliper.png");
	lodepng::decode(glass_2Tex, w5, h5, "../lexus_glass_2.png");
	lodepng::decode(grillTex, w6, h6, "../lexus_grill.png");
	lodepng::decode(grill_2Tex, w7, h7, "../lexus_grill_2.png");
	lodepng::decode(roadTex, w8, h8, "../road2.png");

	//set up the Camera Lens
	Eigen::Matrix4f projection = projectionMatrix(renderHeight, renderWidth, 70.f * M_PI / 180.f, 10000.f, 0.1f);

	//move the camera back 8 units
	Eigen::Matrix4f cameraToWorld = translationMatrix(Eigen::Vector3f(0.0f, 0.0f, -8.0f));
	Eigen::Matrix4f worldToCamera = cameraToWorld.inverse();
	Eigen::Vector3f camWorldPos = (cameraToWorld * Eigen::Vector4f(0, 0, 0, 1)).block<3, 1>(0, 0);

	//set up the Lights
	std::vector<std::unique_ptr<Light>> lights;
	lights.emplace_back(new AmbientLight(Eigen::Vector3f(0.3f, 0.3f, 0.3f)));
	lights.emplace_back(new DirectionalLight(Eigen::Vector3f(0.9f, 0.9f, 0.9f), Eigen::Vector3f(0.5f, -1.0f, -1.0f)));

	//position the Car
	Eigen::Matrix4f carTransform = translationMatrix(Eigen::Vector3f(0.0f, -1.0f, 0.0f)) * rotateYMatrix(M_PI_4) * scaleMatrix(100.0f);

	Eigen::Matrix4f roadTransform = translationMatrix(Eigen::Vector3f(0.0f, -5.05f, 0.0f)) * scaleMatrix(500.0f);

	//RENDER!
	std::cout << "Rendering Engine Starting..." << std::endl;

	//CAR BODY 
	drawMesh(renderBuffer, zBuffer, carMesh,
		Eigen::Vector3f::Zero(), Eigen::Vector3f(2.0f, 2.0f, 2.0f), 64.f,
		bodyTex, w1, h1, BLINN_PHONG, camWorldPos, carTransform, worldToCamera, projection, lights, renderWidth, renderHeight);

	//WHEELS / RUBBER
	drawMesh(renderBuffer, zBuffer, car2Mesh,
		Eigen::Vector3f::Zero(), Eigen::Vector3f(0.05f, 0.05f, 0.05f), 2.f,
		wheelsTex, w3, h3, BLINN_PHONG, camWorldPos, carTransform, worldToCamera, projection, lights, renderWidth, renderHeight);

	//GRILLS & METAL
	drawMesh(renderBuffer, zBuffer, car7Mesh,
		Eigen::Vector3f::Zero(), Eigen::Vector3f(1.0f, 1.0f, 1.0f), 256.f,
		grillTex, w6, h6, BLINN_PHONG, camWorldPos, carTransform, worldToCamera, projection, lights, renderWidth, renderHeight);
	drawMesh(renderBuffer, zBuffer, car8Mesh,
		Eigen::Vector3f::Zero(), Eigen::Vector3f(1.0f, 1.0f, 1.0f), 256.f,
		grill_2Tex, w7, h7, BLINN_PHONG, camWorldPos, carTransform, worldToCamera, projection, lights, renderWidth, renderHeight);

	//CALLIPERS
	drawMesh(renderBuffer, zBuffer, car3Mesh,
		Eigen::Vector3f::Zero(), Eigen::Vector3f(1.0f, 1.0f, 1.0f), 128.f,
		calliperTex, w4, h4, BLINN_PHONG, camWorldPos, carTransform, worldToCamera, projection, lights, renderWidth, renderHeight);
	drawMesh(renderBuffer, zBuffer, car4Mesh,
		Eigen::Vector3f::Zero(), Eigen::Vector3f(1.0f, 1.0f, 1.0f), 128.f,
		calliperTex, w4, h4, BLINN_PHONG, camWorldPos, carTransform, worldToCamera, projection, lights, renderWidth, renderHeight);


	drawMesh(renderBuffer, zBuffer, roadMesh,
		Eigen::Vector3f(0.2f, 0.2f, 0.2f), 
		Eigen::Vector3f(0.1f, 0.1f, 0.1f), 
		5.0f,                              
		roadTex, w8, h8,             
		BLINN_PHONG, camWorldPos, roadTransform, worldToCamera, projection, lights, renderWidth, renderHeight, 500.0f);

	//GLASS
	drawMesh(renderBuffer, zBuffer, car5Mesh,
		Eigen::Vector3f::Zero(), Eigen::Vector3f(1.5f, 1.5f, 1.5f), 512.f,
		glassTex, w2, h2, BLINN_PHONG, camWorldPos, carTransform, worldToCamera, projection, lights, renderWidth, renderHeight, 1.0f, 0.4f);
	drawMesh(renderBuffer, zBuffer, car6Mesh,
		Eigen::Vector3f::Zero(), Eigen::Vector3f(1.5f, 1.5f, 1.5f), 512.f,
		glass_2Tex, w5, h5, BLINN_PHONG, camWorldPos, carTransform, worldToCamera, projection, lights, renderWidth, renderHeight, 1.0f, 0.4f);

	std::cout << "Road Triangles: " << roadMesh.vFaces.size() << std::endl;

	//SSAA DOWNSAMPLE
	std::cout << "Applying Anti-Aliasing..." << std::endl;
	std::vector<uint8_t> finalImage(finalHeight * finalWidth * nChannels);

	for (int y = 0; y < finalHeight; ++y) {
		for (int x = 0; x < finalWidth; ++x) {
			int r = 0, g = 0, b = 0;

			//sample the 4 pixels (2x2 grid) from the massive render buffer
			for (int dy = 0; dy < aaMultiplier; ++dy) {
				for (int dx = 0; dx < aaMultiplier; ++dx) {
					int rx = x * aaMultiplier + dx;
					int ry = y * aaMultiplier + dy;
					int rIdx = (ry * renderWidth + rx) * 4;

					r += renderBuffer[rIdx + 0];
					g += renderBuffer[rIdx + 1];
					b += renderBuffer[rIdx + 2];
				}
			}

			//average the colors and save to the final 1080p image
			int fIdx = (y * finalWidth + x) * 4;
			finalImage[fIdx + 0] = r / (aaMultiplier * aaMultiplier);
			finalImage[fIdx + 1] = g / (aaMultiplier * aaMultiplier);
			finalImage[fIdx + 2] = b / (aaMultiplier * aaMultiplier);
			finalImage[fIdx + 3] = 255;
		}
	}

	//save the final, smoothed Image
	std::cout << "Saving image..." << std::endl;
	int errorCode = lodepng::encode(outputFilename, finalImage, finalWidth, finalHeight);
	if (errorCode) {
		std::cout << "lodepng error encoding image: " << lodepng_error_text(errorCode) << std::endl;
		return errorCode;
	}

	std::cout << "Render Complete! Check lexus_render.png" << std::endl;
	return 0;
}