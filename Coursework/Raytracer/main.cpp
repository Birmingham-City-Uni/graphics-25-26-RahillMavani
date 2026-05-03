#include <Eigen/Dense>
#include <lodepng.h>
#include <json/json.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "BVHNode.hpp"
#include "Triangle.hpp"
#include "Scene.hpp"
#include "Camera.hpp"
#include "PointLight.hpp"
#include "DirectionalLight.hpp"
#include "LambertianShader.hpp"
#include "TexturedLambertianShader.hpp"
#include "PhongShader.hpp"
#include "MirrorShader.hpp"
#include "TexCoordTestShader.hpp"
#include "Model.hpp"
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>
#include "TexturedPhongShader.hpp"

// Utility functions for constructing basic affine transformation matrices.
Eigen::Matrix4f translationMatrix(const Eigen::Vector3f& t) {
	Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
	m(0, 3) = t.x(); m(1, 3) = t.y(); m(2, 3) = t.z();
	return m;
}

Eigen::Matrix4f scaleMatrix(float s) {
	Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
	m(0, 0) = s; m(1, 1) = s; m(2, 2) = s;
	return m;
}

Eigen::Matrix4f rotateYMatrix(float angle) {
	Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
	m(0, 0) = cos(angle); m(0, 2) = sin(angle);
	m(2, 0) = -sin(angle); m(2, 2) = cos(angle);
	return m;
}

/// <summary>
/// Helper to parse the main configuration JSON.
/// </summary>
nlohmann::json loadConfig(const std::string& filename)
{
	std::ifstream configStream(filename);
	nlohmann::json config = nlohmann::json::parse(configStream);
	return config;
}

/// <summary>
/// Extracts an Eigen::Vector3f from a JSON array element.
/// </summary>
Eigen::Vector3f loadVec3FromConfig(const nlohmann::json& config)
{
	return Eigen::Vector3f(config[0], config[1], config[2]);
}

int main(int argc, char* argv[]) {

	// Initialize scene settings from the external config file
	auto config = loadConfig("../config/config.json");

	const int pixHeight = config["pixHeight"], pixWidth = config["pixWidth"];
	const int nChannels = 4; // RGBA

	// Construct the camera using config parameters (handles FOV and thin-lens DoF internally)
	Camera cam(
		loadVec3FromConfig(config["cameraPos"]),
		loadVec3FromConfig(config["cameraForward"]),
		loadVec3FromConfig(config["cameraUp"]),
		pixWidth, pixHeight,
		config["cameraFov"]);

	// Allocate the final image buffer
	std::vector<uint8_t> outImage(pixHeight * pixWidth * nChannels);

	Eigen::Vector3f
		red(1.f, 0.f, 0.f),
		blue(0.f, 0.f, 1.f),
		aqua(0.f, .8f, .8f),
		lavender(178.f / 255.f, 164.f / 255.f, 212.f / 255.f);

	
	// Load all texture assets required for the Lexus model and environment
	std::vector<uint8_t> bodyTex, roadTex;
	unsigned int w1, h1, w8, h8;
	lodepng::decode(bodyTex, w1, h1, "../models/lexus_body.png");
	lodepng::decode(roadTex, w8, h8, "../models/road2.png");

	std::vector<uint8_t> wheelsTex, glassTex, glass_2Tex, grillTex, grill_2Tex, calliperTex;
	unsigned int w2, h2, w3, h3, w4, h4, w5, h5, w6, h6, w7, h7;
	lodepng::decode(glassTex, w2, h2, "../models/lexus_glass.png");
	lodepng::decode(wheelsTex, w3, h3, "../models/lexus_wheels.png");
	lodepng::decode(calliperTex, w4, h4, "../models/lexus_calliper.png");
	lodepng::decode(glass_2Tex, w5, h5, "../models/lexus_glass_2.png");
	lodepng::decode(grillTex, w6, h6, "../models/lexus_grill.png");
	lodepng::decode(grill_2Tex, w7, h7, "../models/lexus_grill_2.png");

	// Initialize shaders. used TexturedPhongShader for the car body to handle specular highlights.
	TexturedPhongShader bodyShader(&bodyTex, w1, h1, Eigen::Vector3f(2.0f, 2.0f, 2.0f), 64.0f);
	TexturedLambertianShader roadShader(&roadTex, w8, h8);

	TexturedLambertianShader wheelsShader(&wheelsTex, w3, h3);
	TexturedLambertianShader glassShader(&glassTex, w2, h2);
	TexturedLambertianShader glass2Shader(&glass_2Tex, w5, h5);
	TexturedLambertianShader grillShader(&grillTex, w6, h6);
	TexturedLambertianShader grill2Shader(&grill_2Tex, w7, h7);
	TexturedLambertianShader calliperShader(&calliperTex, w4, h4);


	Scene scene;

	// Define transformations to position the car and the road plane correctly in world space
	Eigen::Matrix4f carTransform = translationMatrix(Eigen::Vector3f(0.0f, -1.0f, 0.0f)) * rotateYMatrix(M_PI_4) * scaleMatrix(100.0f);
	Eigen::Matrix4f roadTransform = translationMatrix(Eigen::Vector3f(0.0f, -1.05f, 0.0f)) * scaleMatrix(500.0f);

	// Load object geometry
	Model carBodyModel("../models/lexus_body.obj");
	Model roadModel("../models/road2.obj");

	Model wheelsModel("../models/lexus_wheels.obj");
	Model glassModel("../models/lexus_glass.obj");
	Model glass2Model("../models/lexus_glass_2.obj");
	Model grillModel("../models/lexus_grill.obj");
	Model grill2Model("../models/lexus_grill_2.obj");
	Model calliperModel("../models/lexus_calliper.obj");
	Model calliper2Model("../models/lexus_calliper_2.obj");

	// Wrap geometry and shaders into BVH nodes for accelerated intersection testing, then add to scene
	scene.renderables.push_back(std::make_shared<BVHNode>(carBodyModel, &bodyShader, 4, carTransform));
	scene.renderables.push_back(std::make_shared<BVHNode>(roadModel, &roadShader, 4, roadTransform));

	scene.renderables.push_back(std::make_shared<BVHNode>(wheelsModel, &wheelsShader, 4, carTransform));
	scene.renderables.push_back(std::make_shared<BVHNode>(glassModel, &glassShader, 4, carTransform));
	scene.renderables.push_back(std::make_shared<BVHNode>(glass2Model, &glass2Shader, 4, carTransform));
	scene.renderables.push_back(std::make_shared<BVHNode>(grillModel, &grillShader, 4, carTransform));
	scene.renderables.push_back(std::make_shared<BVHNode>(grill2Model, &grill2Shader, 4, carTransform));
	scene.renderables.push_back(std::make_shared<BVHNode>(calliperModel, &calliperShader, 4, carTransform));
	scene.renderables.push_back(std::make_shared<BVHNode>(calliper2Model, &calliperShader, 4, carTransform));

	// Establish lighting setup
	Eigen::Vector3f ambientLight(0.4f, 0.4f, 0.4f);

	std::vector<std::unique_ptr<Light>> lightSources;

	// Fill light (Point) and main key light (Directional)
	lightSources.push_back(std::make_unique<PointLight>(Eigen::Vector3f(-1.f, 5.f, -5.f), 8.f * Eigen::Vector3f(1.f, 1.f, 1.f)));
	lightSources.push_back(std::make_unique<DirectionalLight>(Eigen::Vector3f(0.5f, -1.0f, -1.0f), 1.2f * Eigen::Vector3f(1.f, 1.f, 1.f)));


	// Randomize scanline processing order to optimize thread workload distribution
	std::vector<unsigned int> scanlines(pixHeight);
	for (int i = 0; i < pixHeight; ++i) scanlines[i] = i;

	if (config["shuffleScanlines"]) {
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(scanlines.begin(), scanlines.end(), g);
	}

	auto startTime = std::chrono::steady_clock::now();

	Ray ray = cam.getRay(531, 325);
	HitInfo hitInfo;
	scene.intersect(ray, 1e-6f, 1e6f, hitInfo, VISIBLE_BITMASK);
	float x = hitInfo.hitT;

	// OpenMP loop for multithreaded rendering
	#pragma omp parallel for
	for (int y = 0; y < pixHeight; ++y) {
		for (int x = 0; x < pixWidth; ++x) {
			Eigen::Vector3f finalColor(0.f, 0.f, 0.f);

			// 2x2 Supersampling grid for Anti-Aliasing
			int gridSize = 2;

			for (int sy = 0; sy < gridSize; ++sy) {
				for (int sx = 0; sx < gridSize; ++sx) {

					// Calculate sub-pixel jitter
					float dx = (sx + 0.5f) / gridSize;
					float dy = (sy + 0.5f) / gridSize;

					// Cast ray through the sub-pixel coordinate
					Ray ray = cam.getRay(x + dx, scanlines[y] + dy);
					HitInfo hitInfo;

					if (scene.intersect(ray, 1e-6f, 1e6f, hitInfo, VISIBLE_BITMASK)) {
						
						// Compute localized shading via the hit material
						Eigen::Vector3f rayColor = hitInfo.shader->getColor(
							hitInfo, &scene,
							lightSources, ambientLight,
							0, config["maxBounces"]);

						// Volumetric integration: Distance Fog
						// Linearly interpolates between the shaded surface color and the ambient fog color based on intersection depth.

						float fogMaxDistance = 30.0f; // Distance where fog becomes 100% thick
						float fogFactor = std::min(hitInfo.hitT / fogMaxDistance, 1.0f);
						Eigen::Vector3f fogColor(0.05f, 0.05f, 0.05f); // Dark grey night fog

						// Mix the car color with the fog based on distance
						finalColor += (rayColor * (1.0f - fogFactor)) + (fogColor * fogFactor);
					}
					else {
						// Ray escaped the scene; return ambient fog color
						finalColor += Eigen::Vector3f(0.05f, 0.05f, 0.05f);
					}
				}
			}

			// Post-Processing: Radial Vignette
			// Calculates normalized distance from screen center to apply a darkening falloff near the edges.
			float centerX = pixWidth / 2.0f;
			float centerY = pixHeight / 2.0f;
			float dist = sqrt(pow(x - centerX, 2) + pow(scanlines[y] - centerY, 2));
			float maxDist = sqrt(pow(centerX, 2) + pow(centerY, 2));

			float vignette = 1.0f - pow(dist / maxDist, 2.0f);
			vignette = std::max(0.2f, vignette);

			finalColor *= vignette;

			// Resolve supersampled rays into final pixel color
			finalColor /= (float)(gridSize * gridSize);

			// Post-Processing: HDR Tone Mapping
			// Applies an exponential exposure curve to compress high dynamic range lighting into the 0-1 display space.

			float exposure = 1.2f;
			finalColor.x() = 1.0f - exp(-finalColor.x() * exposure);
			finalColor.y() = 1.0f - exp(-finalColor.y() * exposure);
			finalColor.z() = 1.0f - exp(-finalColor.z() * exposure);

			// Map float colors to 8-bit output buffer
			int line = (pixHeight - scanlines[y]) - 1;
			outImage[(x + line * pixWidth) * nChannels + 0] = finalColor.x() * 255;
			outImage[(x + line * pixWidth) * nChannels + 1] = finalColor.y() * 255;
			outImage[(x + line * pixWidth) * nChannels + 2] = finalColor.z() * 255;
			outImage[(x + line * pixWidth) * nChannels + 3] = 255;
		}

		// Print progress
		if (omp_get_thread_num() == omp_get_num_threads() - 1) {
			std::clog << "\rScanlines remaining: " << (pixHeight - y) << ' ' << std::flush;
		}
	}

	auto renderTime = std::chrono::steady_clock::now() - startTime;

	std::cout << "Render duration " << std::chrono::duration_cast<std::chrono::milliseconds>(renderTime).count() * 1e-3f << " seconds." << std::endl;

	// *** Save the output image ***
	int errorCode;
	errorCode = lodepng::encode(config["outputFilename"], outImage, pixWidth, pixHeight);
	if (errorCode) { // check the error code, in case an error occurred.
		std::cout << "lodepng error encoding image: " << lodepng_error_text(errorCode) << std::endl;
		return errorCode;
	}

	return 0;
}
