#pragma once
#include "Ray.hpp"
#include <random>

/// <summary>
/// Movable camera class. Provide the camera location, forward direction and an up
/// vector, along with the image dimensions and vertical Field of View angle (radians).
/// The camera can then produce a ray passing through each pixel location.
/// </summary>
class Camera
{
private:
	Eigen::Vector3f location_, bottomLeftPix_, right1pix_, up1pix_;
	Eigen::Vector3f rightVec_, upVec_;
	float lensRadius_;

	Eigen::Vector2f randomInUnitDisk()
	{
		// thread_local ensures OpenMP multithreading doesn't crash the randomizer!
		static thread_local std::mt19937 generator(std::random_device{}());
		std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
		while (true) {
			Eigen::Vector2f p(distribution(generator), distribution(generator));
			if (p.squaredNorm() < 1.0f) return p;
		}
	}

public:
	//DOF
	Camera(
		const Eigen::Vector3f& location,
		const Eigen::Vector3f& forward,
		const Eigen::Vector3f& up,
		int pixWidth, int pixHeight,
		float vertFov,
		float aperture = 0.001f)
		: location_(location)
	{

		lensRadius_ = 0.04f / 2.0f;

		float focusDist = 8.0f;
		if (focusDist < 0.1f) focusDist = 10.0f;

		Eigen::Vector3f forwardVec = forward.normalized();
		rightVec_ = (up.cross(forwardVec)).normalized();
		upVec_ = (forward.cross(rightVec_)).normalized();

		float aspect = static_cast<float>(pixWidth) / static_cast<float>(pixHeight);

		float halfHeight = tan(vertFov / 2);
		float halfWidth = aspect * halfHeight;

		bottomLeftPix_ = location + focusDist * forwardVec - focusDist * (halfWidth * rightVec_ + halfHeight * upVec_);

		right1pix_ = rightVec_ * focusDist * halfWidth * 2.f / static_cast<float>(pixWidth);
		up1pix_ = upVec_ * focusDist * halfHeight * 2.f / static_cast<float>(pixHeight);
	}

	Ray getRay(float pixX, float pixY)
	{
		Eigen::Vector2f randomDisk = lensRadius_ * randomInUnitDisk();
		Eigen::Vector3f offset = rightVec_ * randomDisk.x() + upVec_ * randomDisk.y();

		// Motion blur
		float motionBlurAmount = 0.3f; // how fast is cam moving sideways
		float randomTime = ((rand() % 100) / 100.0f) * motionBlurAmount;
		Eigen::Vector3f motionOffset = rightVec_ * randomTime;

		Ray ray;

		//add the motion offset to the camera's starting position
		ray.origin = location_ + offset + motionOffset;
		Eigen::Vector3f pixelPos = bottomLeftPix_ +
			static_cast<float>(pixX) * right1pix_ +
			static_cast<float>(pixY) * up1pix_;

		ray.direction = (pixelPos - ray.origin).normalized();
		return ray;
	}
};

