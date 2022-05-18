@group(0) @binding(0) var<uniform> camera   : Camera;
@group(0) @binding(1) var SAMPLER           : sampler;
@group(0) @binding(2) var environment       : texture_cube<f32>;
@group(0) @binding(3) var irradiance        : texture_cube<f32>;
@group(0) @binding(4) var specular          : texture_cube<f32>;
@group(0) @binding(5) var brdf              : texture_2d<f32>;

// Adapted from Physically Based Rendering
// Copyright (c) 2022-2022  Bram Kraaijeveld
// Copyright (c) 2017-2018  Michał Siejak (https://github.com/Nadrin/PBR)

let PI = 3.14159265359;
let Fdielectric = vec3<f32>(0.04);

// GGX/Towbridge-Reitz normal distribution function.
// Uses Disney's reparametrization of alpha = roughness^2.
fn ndfGGX(cosLh : f32, roughness : f32) -> f32
{
	let alpha   = roughness * roughness;
	let alphaSq = alpha * alpha;

	let denom = (cosLh * cosLh) * (alphaSq - 1.0) + 1.0;
	return alphaSq / (PI * denom * denom);
}

// Single term for separable Schlick-GGX below.
fn gaSchlickG1(cosTheta : f32, k : f32) -> f32
{
	return cosTheta / (cosTheta * (1.0 - k) + k);
}

// Schlick-GGX approximation of geometric attenuation function using Smith's method.
fn gaSchlickGGX(cosLi : f32, cosLo : f32, roughness : f32) -> f32
{
	let r = roughness + 1.0;
	let k = (r * r) / 8.0; // Epic suggests using this roughness remapping for analytic lights.
	return gaSchlickG1(cosLi, k) * gaSchlickG1(cosLo, k);
}

// Shlick's approximation of the Fresnel factor.
fn fresnelSchlick(F0 : vec3<f32>, cosTheta : f32) -> vec3<f32>
{
	return F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - cosTheta, 5.0);
}

fn rsi(r0 : vec3<f32>, rd : vec3<f32>, s0 : vec3<f32>, sr : f32) -> f32 {
    let a = dot(rd, rd);
    let s0_r0 = r0 - s0;
    let b = 2.0 * dot(rd, s0_r0);
    let c = dot(s0_r0, s0_r0) - (sr * sr);
    return select(-1., (-b - sqrt((b*b) - 4.0*a*c))/(2.0*a), b*b - 4.0*a*c >= 0.0);
}

fn shade(eye: vec3<f32>, pos: vec3<f32>, N: vec3<f32>, albedo: vec3<f32>, roughness: f32, metalness: f32) -> vec3<f32>
{
    // Outgoing light direction (vector from world-space fragment position to the "eye").
	let Lo = normalize(eye - pos);

	// Angle between surface normal and outgoing light direction.
	let cosLo = max(0.0, dot(N, Lo));

	// Specular reflection vector.
	let Lr = 2.0 * cosLo * N - Lo;

	// Fresnel reflectance at normal incidence (for metals use albedo color).
	let F0 = mix(Fdielectric, albedo, metalness);

    // Sample diffuse irradiance at normal direction.
    let ir = textureSample(irradiance, SAMPLER, N).rgb;

    // Calculate Fresnel term for ambient lighting.
    // Since we use pre-filtered cubemap(s) and irradiance is coming from many directions
    // use cosLo instead of angle with light's half-vector (cosLh above).
    // See: https://seblagarde.wordpress.com/2011/08/17/hello-world/
    let F = fresnelSchlick(F0, cosLo);

    // Get diffuse contribution factor (as with direct lighting).
    let kd = mix(vec3<f32>(1.0) - F, vec3<f32>(0.0), metalness);

    // Irradiance map contains exitant radiance assuming Lambertian BRDF, no need to scale by 1/PI here either.
    let diffuseIBL = kd * albedo * ir;

    // Sample pre-filtered specular reflection environment at correct mipmap level.
    let sp = textureSampleLevel(specular, SAMPLER, Lr, roughness * f32(textureNumLevels(specular))).rgb;

    // Split-sum approximation factors for Cook-Torrance specular BRDF.
    var specularBRDF = textureSample(brdf, SAMPLER, vec2<f32>(cosLo, roughness)).rg;
    // specularBRDF = vec2<f32>(.5, .4);

    // Total specular IBL contribution.
    let specularIBL = (F0 * specularBRDF.x + specularBRDF.y) * sp;

    // Total ambient lighting contribution.
    return diffuseIBL + specularIBL;
}

fn image(origin: vec3<f32>, ray: vec3<f32>) -> vec3<f32>
{
    let env = textureSample(environment, SAMPLER, ray).rgb;

    let sphere = rsi(origin, ray, vec3<f32>(0.), 4.);
    let p = origin + sphere * ray;
    let n = normalize(p);

    let ibl = shade(origin, p, n, vec3<f32>(1.), .2, 0.);

    return select(env, ibl, sphere > 0.);
}

@stage(fragment)
fn fragment(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32>
{
    let origin = camera.iV[3].xyz;
    let ray = createRay(uv, camera.iP, camera.iV);

    let color = image(origin, ray);

    return vec4<f32>(1. - exp(-1. * color), 1.);
}