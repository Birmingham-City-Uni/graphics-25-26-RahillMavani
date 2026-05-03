#pragma once
#include "Shader.hpp"
#include <algorithm>
#include <vector>
#include <cmath>
#include <memory>
#include <Eigen/Dense>
#include "GeomUtil.hpp"

// EXTRA SHADER TYPES (Textured Blinn-Phong)
// Combines diffuse texture sampling with Blinn-Phong specular highlights.
class TexturedPhongShader : public Shader
{
private:
	const std::vector<uint8_t>* albedoTexture_;
	const int texWidth_, texHeight_;
	bool shadowTest_;
	Eigen::Vector3f specularColor_;
	float specularExponent_;

public:
	TexturedPhongShader(const std::vector<uint8_t>* albedoTexture, int texWidth, int texHeight, 
						const Eigen::Vector3f& specularColor, float specularExponent, bool shadowTest = true)
		: albedoTexture_(albedoTexture), texWidth_(texWidth), texHeight_(texHeight),
		  specularColor_(specularColor), specularExponent_(specularExponent), shadowTest_(shadowTest)
	{}

	virtual Eigen::Vector3f getColor(const HitInfo& hitInfo, 
		const Renderable* scene, 
		const std::vector<std::unique_ptr<Light>>& lights,
		const Eigen::Vector3f& ambientLight,
		int currBounceCount,
		const int maxBounces) const
	{
		// Fallback: Return bright magenta if texture data is missing
		if (albedoTexture_->empty() || texWidth_ == 0 || texHeight_ == 0) {
			return Eigen::Vector3f(1.0f, 0.0f, 1.0f); 
		}

		// Converts wrapped UV coordinates (0.0 to 1.0) into exact 2D pixel indices to extract the RGB color
		Eigen::Vector2f tex = hitInfo.texCoords;
		float u = tex.x() - std::floor(tex.x());
		float v = tex.y() - std::floor(tex.y());
		int pixX = std::max(0, std::min(static_cast<int>(u * texWidth_), texWidth_ - 1));
		int pixY = std::max(0, std::min(static_cast<int>((1.f - v) * texHeight_), texHeight_ - 1));

		Eigen::Vector3f albedo;
		albedo.x() = static_cast<float>((*albedoTexture_)[(pixX + texWidth_ * pixY) * 4 + 0]) / 255.f;
		albedo.y() = static_cast<float>((*albedoTexture_)[(pixX + texWidth_ * pixY) * 4 + 1]) / 255.f;
		albedo.z() = static_cast<float>((*albedoTexture_)[(pixX + texWidth_ * pixY) * 4 + 2]) / 255.f;

		// Base ambient lighting application
		Eigen::Vector3f color = coefftWiseMul(albedo, ambientLight);

		/* source from AI (Gemini) starts here */
		// Approximates a specular map by calculating the grayscale luminance of the texture.
		// This ensures visually dark areas (like black rubber) reflect less light than bright areas.
		float luminance = (albedo.x() * 0.299f) + (albedo.y() * 0.587f) + (albedo.z() * 0.114f);
		Eigen::Vector3f mappedSpecular = specularColor_ * luminance;
		/* source from AI (Gemini) ends here */


		//LIGHTING (Diffuse + Blinn-Phong Specular)
		for (auto& light : lights) 
		{
			// Cast a shadow ray towards the light source
			if (shadowTest_) 
			{
				if (!light->visibilityCheck(hitInfo.location, scene))
					continue; // Pixel is in shadow, skip adding direct light
			}

			Eigen::Vector3f lightVec = light->getVecToLight(hitInfo.location);
			Eigen::Vector3f viewVec = -hitInfo.inDirection.normalized();
			Eigen::Vector3f normal = hitInfo.normal.normalized();

			// Standard Lambertian diffuse shading (N dot L)
			float nDotL = std::max(normal.dot(lightVec), 0.f);
			Eigen::Vector3f diffuseOut = nDotL * coefftWiseMul(light->getIntensity(hitInfo.location), albedo);

			// Blinn-Phong specular shading: Uses the Half-Vector (N dot H) 
			// Computationally cheaper than pure Phong and prevents unnatural cutoffs at grazing angles.
			Eigen::Vector3f halfVec = (lightVec + viewVec).normalized();
			float nDotH = std::max(normal.dot(halfVec), 0.f);
			float specTerm = std::pow(nDotH, specularExponent_);
			Eigen::Vector3f specularOut = specTerm * coefftWiseMul(light->getIntensity(hitInfo.location), mappedSpecular);

			// Add to final color
			if (nDotL > 0.0f) {
				color += diffuseOut + specularOut;
			}
		}

		return color;
	}
};