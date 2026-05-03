#pragma once
#include "Light.hpp"

class DirectionalLight : public Light
{
private:
	Eigen::Vector3f direction_, intensity_;
public:
	DirectionalLight(const Eigen::Vector3f& direction, const Eigen::Vector3f& intensity)
		:direction_(direction.normalized()), intensity_(intensity)
	{}


	virtual bool visibilityCheck(const Eigen::Vector3f& location, const Renderable* renderable) const override
	{
		// --- FEATURE 6: SOFT SHADOWS (Area Light Approximation) ---
        /* source from AI (Gemini) starts here */
        // Replaces unnaturally sharp shadows by adding random jitter to the light's 
        // direction vector. The Anti-Aliasing grid tests 4 slightly different 
        // sun angles per pixel, averaging them into a realistic, soft penumbra.
		float randomX = ((rand() % 100) / 100.0f - 0.5f) * 0.1f;
		float randomZ = ((rand() % 100) / 100.0f - 0.5f) * 0.1f;

		// Add the wobble to the light direction so shadows blur at the edges
		Eigen::Vector3f softDirection = -direction_ + Eigen::Vector3f(randomX, 0.0f, randomZ);
		softDirection.normalize();
		/* source from AI (Gemini) ends here */

		Ray shadowRay;
		shadowRay.origin = location;
		shadowRay.direction = softDirection;
		HitInfo info;

		return !renderable->intersect(shadowRay, 1e-4f, 1e4f, info, SHADOW_BITMASK);
	}

	virtual Eigen::Vector3f getIntensity(const Eigen::Vector3f& location) const override
	{
		return intensity_;
	}

	virtual Eigen::Vector3f getVecToLight(const Eigen::Vector3f& location) const override
	{
		return -direction_;
	}

};
