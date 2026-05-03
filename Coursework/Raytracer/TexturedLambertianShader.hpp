#pragma once
#include "Shader.hpp"

/// <summary>
/// Lambertian reflectance shader that samples albedo values from a texture.
/// The texture should be stored as an image file (TGAImage instance).
/// </summary>
class TexturedLambertianShader : public Shader
{
private:
	const std::vector<uint8_t>* albedoTexture_;
	const int texWidth_, texHeight_;
	bool shadowTest_;
public:
	TexturedLambertianShader(const std::vector<uint8_t>* albedoTexture, int texWidth, int texHeight, bool shadowTest=true)
		:shadowTest_(shadowTest), albedoTexture_(albedoTexture),
		texWidth_(texWidth), texHeight_(texHeight)
	{}

	virtual Eigen::Vector3f getColor(const HitInfo& hitInfo, 
		const Renderable* scene, 
		const std::vector<std::unique_ptr<Light>>& lights,
		const Eigen::Vector3f& ambientLight,
		int currBounceCount,
		const int maxBounces) const
	{
		// --- 1. SAFETY CHECK: PREVENT CRASH IF PNG FAILED TO LOAD ---
		if (albedoTexture_->empty() || texWidth_ == 0 || texHeight_ == 0) {
			return Eigen::Vector3f(1.0f, 0.0f, 1.0f); // Return bright pink to warn us!
		}

		Eigen::Vector3f albedo;
		Eigen::Vector2f tex = hitInfo.texCoords;

		// --- 2. WRAP UVS: ALLOWS TEXTURES TO TILE PERFECTLY ---
		float u = tex.x() - std::floor(tex.x());
		float v = tex.y() - std::floor(tex.y());

		// --- 3. CONVERT TO PIXEL COORDINATES ---
		int pixX = static_cast<int>(u * texWidth_);
		int pixY = static_cast<int>((1.f - v) * texHeight_);

		// --- 4. STRICT CLAMPING: MUST SUBTRACT 1 TO PREVENT CRASHES ---
		pixX = std::max(0, std::min(pixX, texWidth_ - 1));
		pixY = std::max(0, std::min(pixY, texHeight_ - 1));

		albedo.x() = static_cast<float>((*albedoTexture_)[(pixX + texWidth_ * pixY) * 4 + 0]) / 255.f;
		albedo.y() = static_cast<float>((*albedoTexture_)[(pixX + texWidth_*pixY)*4 + 1]) / 255.f;
		albedo.z() = static_cast<float>((*albedoTexture_)[(pixX + texWidth_*pixY)*4 + 2]) / 255.f;

		Eigen::Vector3f color = coefftWiseMul(albedo, ambientLight);

		for (auto& light : lights) {
			if (shadowTest_) {
				if (!light->visibilityCheck(hitInfo.location, scene))
					continue;
			}
			Eigen::Vector3f lightVec = light->getVecToLight(hitInfo.location);
			float dotProd = std::max(lightVec.dot(hitInfo.normal), 0.f);
			color += dotProd * coefftWiseMul(light->getIntensity(hitInfo.location), albedo);
		}

		return color;
	}
};

