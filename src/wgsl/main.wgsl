@group(0) @binding(0) var<uniform> camera   : Camera;
@group(0) @binding(1) var SAMPLER           : sampler;
@group(0) @binding(2) var environment       : texture_cube<f32>;
@group(0) @binding(3) var irradiance        : texture_cube<f32>;
@group(0) @binding(4) var specular          : texture_cube<f32>;
@group(0) @binding(5) var brdf              : texture_2d<f32>;
@group(0) @binding(6) var terrain           : texture_2d<f32>;

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

struct Ray
{
    origin : vec3<f32>,
    direction : vec3<f32>,
    inverse : vec3<f32>,
};

struct AABB
{
    min : vec3<f32>,
    max : vec3<f32>
};

struct Sphere
{
    origin : vec3<f32>,
    radius : f32
};

struct Hit
{
    position    : vec3<f32>,
    normal      : vec3<f32>,
    hit         : bool
};

struct Material
{
    distance : f32,
    position : vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32
};

fn create_ray(camera: Camera, uv: vec2<f32>) -> Ray
{
    // choose an arbitrary point in the viewing volume
    // z = -1 equals a point on the near plane, i.e. the screen
    let pointNDS = vec3<f32>(uv, -1.);

    // as this is in homogenous space, add the last homogenous coordinate
    let pointNDSH = vec4<f32>(pointNDS, 1.0);
    // transform by inverse projection to get the point in view space
    var dirEye = camera.iP * pointNDSH;

    // since the camera is at the origin in view space by definition,
    // the current point is already the correct direction
    // (dir(0,P) = P - 0 = P as a direction, an infinite point,
    // the homogenous component becomes 0 the scaling done by the
    // w-division is not of interest, as the direction in xyz will
    // stay the same and we can just normalize it later
    dirEye.w = 0.;

    // compute world ray direction by multiplying the inverse view matrix
    let dirWorld = (camera.iV * dirEye).xyz;

    // now normalize direction
    let normDirWorld = normalize(dirWorld);

    return Ray(camera.iV[3].xyz, normDirWorld, 1. / normDirWorld);
}

fn sphere(index: u32, ray: Ray) -> Material
{
    let sphere = Sphere(vec3<f32>(0., 0., f32(index) * 3. - 9.), 1.);

    var hit : Material;

    hit.distance = trace_sphere(ray, sphere);
    hit.position = ray.origin + hit.distance * ray.direction;
    hit.normal = normalize(hit.position - sphere.origin);
    hit.roughness = f32(index) / 4.;
    hit.albedo = vec3<f32>(1.);

    return hit;
}

fn image(ray: Ray) -> vec3<f32>
{
    let env = textureSample(environment, SAMPLER, ray.direction).rgb;

    var hit : Material;
    hit.distance = -1.;

    for (var i=0u; i<7u; i++)
    {
        let h = sphere(i, ray);

        if (h.distance > 0. && (h.distance < hit.distance || hit.distance < 0.))
        {
            hit = h;
        }
    }

    let ibl = shade(ray.origin, hit.position, hit.normal, hit.albedo, hit.roughness, hit.metallic);

    return select(env, ibl, hit.distance > 0.);
}

fn trace_aabb(ray: Ray, aabb: AABB) -> vec2<f32>
{
    let tMin = (aabb.min - ray.origin) * ray.inverse;
    let tMax = (aabb.max - ray.origin) * ray.inverse;
    let t1 = min(tMin, tMax);
    let t2 = max(tMin, tMax);
    let tNear = max(max(t1.x, t1.y), t1.z);
    let tFar = min(min(t2.x, t2.y), t2.z);

    return vec2<f32>(tNear, tFar);
}

fn trace_sphere(ray: Ray, sphere: Sphere) -> f32 {
    let a = dot(ray.direction, ray.direction);
    let s0_r0 = ray.origin - sphere.origin;
    let b = 2.0 * dot(ray.direction, s0_r0);
    let c = dot(s0_r0, s0_r0) - (sphere.radius * sphere.radius);
    return select(-1., (-b - sqrt((b*b) - 4.0*a*c))/(2.0*a), b*b - 4.0*a*c >= 0.0);
}

fn heightmap_aabb(heightmap: texture_2d<f32>, coords: vec2<i32>, level: i32) -> AABB
{
    let height = textureLoad(heightmap, coords, level).x;
    let s = 1 << u32(level);

    let min = coords * s;
    let max = min + s;

    return AABB(
        vec3<f32>(f32(min.x), -1000.  , f32(min.y)),
        vec3<f32>(f32(max.x), height, f32(max.y))
    );
}

fn trace_heightmap(ray: Ray, heightmap: texture_2d<f32>) -> Hit
{
    var hit : Hit;

    let ISIZE = 1. / vec2<f32>(textureDimensions(heightmap));
    let MIPS = i32(textureNumLevels(heightmap));

    let dx = vec2<i32>(1, 0);
    let dy = vec2<i32>(0, 1);
    var dt = 1.;

    // Ray - Heightmap AABB intersection (only sample when needed)
    // TODO: Recursively Sample mipmap AABBs for big performance gain
    let intersect = trace_aabb(ray, heightmap_aabb(heightmap, vec2<i32>(0), MIPS));

    if (intersect.x > intersect.y || intersect.x <= 0.) { return hit; }

    var m0 = vec4<f32>(0.);
    var p0 = vec3<f32>(0.);

    for (var t = intersect.x; t <= intersect.y; t+=dt)
    {
        let p1 = ray.origin + ray.direction * t;
        let m1 = textureLoad(heightmap, vec2<i32>(p1.xz), 0);

        if (p1.y < m1.x)
        {
             hit.hit = true; // Intersection!

            // Find intersection point between this and previous sample point
            let x = (m0.x - p0.y) / ((p1.y - p0.y) - (m1.x - m0.x));

            // Linearly Interpolate Position and Normal
            hit.position = mix(p0, p1, x);
            hit.normal = mix(m0, m1, x).yzw;
            return hit;
        }

        m0 = m1;
        p0 = p1;

        // Less detail the further from the camera this ray is
        // TODO: jitter for improved visual quality
        dt = max(1., .02 * t);
    }

    return hit;
}

@stage(fragment)
fn fragment(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32>
{
    // return textureSample(terrain, SAMPLER, uv);

    var ray = create_ray(camera, uv);

    let tmp = textureSample(terrain, SAMPLER, uv);

    var color = image(ray);

    let hit = trace_heightmap(ray, terrain);

    if (hit.hit)
    {
        color = shade(ray.origin, hit.position, hit.normal, vec3<f32>(1.), 0., 0.);
        // return vec4<f32>(.5 + .5 * normal, 1.);
        // return vec4<f32>(f32(index.x) / 512., f32(index.y) / 512., 0., 1.);
    }

    return vec4<f32>(1. - exp(-1. * color), 1.);
}