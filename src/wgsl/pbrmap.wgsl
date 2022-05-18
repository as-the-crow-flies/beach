// Adapted from Physically Based Rendering (https://github.com/Nadrin/PBR)
// Copyright (c) 2022-2022  Bram Kraaijeveld
// Copyright (c) 2017-2018  Michał Siejak

let PI = 3.14159265359;
let TwoPI = 6.28318530718;
let Epsilon = 0.00001;

let NumSamples = 1024u;

@group(0) @binding(0) var SAMPLER               : sampler;
@group(0) @binding(1) var source                : texture_cube<f32>;
@group(0) @binding(2) var result                : texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var<uniform> ROUGHNESS    : f32;

// Compute Van der Corput radical inverse
// See: http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
fn radicalInverse_VdC(i : u32) -> f32
{
	var bits = (i << 16u) | (i >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return f32(bits) * 2.3283064365386963e-10; // / 0x100000000
}

// Sample i-th point from Hammersley point set of NumSamples points total.
fn sampleHammersley(i : u32) -> vec2<f32>
{
	return vec2<f32>(f32(i) / f32(NumSamples), radicalInverse_VdC(i));
}

// Uniformly sample point on a hemisphere.
// Cosine-weighted sampling would be a better fit for Lambertian BRDF but since this
// compute shader runs only once as a pre-processing step performance is not *that* important.
// See: "Physically Based Rendering" 2nd ed., section 13.6.1.
fn sampleHemisphere(u1 : f32, u2 : f32) -> vec3<f32>
{
	let u1p = sqrt(max(0.0, 1.0 - u1*u1));
	return vec3<f32>(cos(TwoPI*u2) * u1p, sin(TwoPI*u2) * u1p, u1);
}

// Importance sample GGX normal distribution function for a fixed roughness value.
// This returns normalized half-vector between Li & Lo.
// For derivation see: http://blog.tobias-franke.eu/2014/03/30/notes_on_importance_sampling.html
fn sampleGGX(u1: f32, u2: f32, roughness: f32) -> vec3<f32>
{
	let alpha = roughness * roughness;

	let cosTheta = sqrt((1.0 - u2) / (1.0 + (alpha*alpha - 1.0) * u2));
	let sinTheta = sqrt(1.0 - cosTheta*cosTheta); // Trig. identity
	let phi = TwoPI * u1;

	// Convert to Cartesian upon return.
	return vec3<f32>(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

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

// Schlick-GGX approximation of geometric attenuation function using Smith's method (IBL version).
fn gaSchlickGGX_IBL(cosLi : f32, cosLo : f32, roughness : f32) -> f32
{
	let r = roughness;
	let k = (r * r) / 2.0; // Epic suggests using this roughness remapping for IBL lighting.
	return gaSchlickG1(cosLi, k) * gaSchlickG1(cosLo, k);
}

// Calculate normalized sampling direction vector based on current fragment coordinates (gl_GlobalInvocationID.xyz).
// This is essentially "inverse-sampling": we reconstruct what the sampling vector would be if we wanted it to "hit"
// this particular fragment in a cubemap.
// See: OpenGL core profile specs, section 8.13.
fn getSamplingVector(id : vec3<u32>) -> vec3<f32>
{
    let st = vec2<f32>(id.xy) / vec2<f32>(textureDimensions(result));
    let uv = 2.0 * vec2<f32>(st.x, 1.0-st.y) - vec2<f32>(1.0);

   switch (id.z)
   {
    case 0u { return vec3<f32>(1.0,  uv.y, -uv.x); }
    case 1u { return vec3<f32>(-1.0, uv.y,  uv.x); }
    case 2u { return vec3<f32>(uv.x, 1.0, -uv.y); }
    case 3u { return vec3<f32>(uv.x, -1.0, uv.y); }
    case 4u { return vec3<f32>(uv.x, uv.y, 1.0); }
    case 5u { return vec3<f32>(-uv.x, uv.y, -1.0); }
    default { return vec3<f32>(0.); }
   };
}

// Compute orthonormal basis for converting from tanget/shading space to world space.
fn computeBasisVector(N : vec3<f32>) -> vec3<f32>
{
	// Branchless select non-degenerate T.
	var T = cross(N, vec3(0.0, 1.0, 0.0));
	T = mix(cross(N, vec3(1.0, 0.0, 0.0)), T, step(Epsilon, dot(T, T)));
	return normalize(T);
}

// Convert point from tangent/shading space to world space.
fn tangentToWorld(v : vec3<f32>, N : vec3<f32>, S : vec3<f32>, T : vec3<f32>) -> vec3<f32>
{
	return S * v.x + T * v.y + N * v.z;
}

// Computes diffuse irradiance cubemap convolution for image-based lighting.
// Uses quasi Monte Carlo sampling with Hammersley sequence.
@stage(compute) @workgroup_size(8, 8)
fn irmap(@builtin(global_invocation_id) id: vec3<u32>)
{
	let N = normalize(getSamplingVector(id));
	let T = computeBasisVector(N);
	let S = normalize(cross(N, T));

	// Monte Carlo integration of hemispherical irradiance.
	// As a small optimization this also includes Lambertian BRDF assuming perfectly white surface (albedo of 1.0)
	// so we don't need to normalize in PBR fragment shader (so technically it encodes exitant radiance rather than irradiance).
	var irradiance = vec3<f32>(0.);
	for(var i=0u; i<NumSamples; i++) {
		let u  = sampleHammersley(i);
		let Li = tangentToWorld(sampleHemisphere(u.x, u.y), N, S, T);
		let cosTheta = max(0.0, dot(Li, N));

		// PIs here cancel out because of division by pdf.
		irradiance += 2.0 * textureSampleLevel(source, SAMPLER, Li, 0.).rgb * cosTheta;
	}
	irradiance /= vec3<f32>(f32(NumSamples));

	textureStore(result, vec2<i32>(id.xy), i32(id.z), vec4<f32>(irradiance, 1.0));
}

// Pre-filters environment cube map using GGX NDF importance sampling.
// Part of specular IBL split-sum approximation.
@stage(compute) @workgroup_size(8, 8)
fn spmap(@builtin(global_invocation_id) id: vec3<u32>)
{
	// Solid angle associated with a single cubemap texel at zero mipmap level.
	// This will come in handy for importance sampling below.
	let inputSize = vec2<f32>(textureDimensions(source));
	let wt = 4.0 * PI / (6. * inputSize.x * inputSize.y);

	// Approximation: Assume zero viewing angle (isotropic reflections).
    let N = normalize(getSamplingVector(id));
    let T = computeBasisVector(N);
    let S = normalize(cross(N, T));

    let Lo = N;

	var color = vec3<f32>(0.);
	var weight = 0.;

	// Convolve environment map using GGX NDF importance sampling.
	// Weight by cosine term since Epic claims it generally improves quality.
	for(var i=0u; i<NumSamples; i++) {
		let u = sampleHammersley(i);
		let Lh = tangentToWorld(sampleGGX(u.x, u.y, ROUGHNESS), N, S, T);

		// Compute incident direction (Li) by reflecting viewing direction (Lo) around half-vector (Lh).
		let Li = 2.0 * dot(Lo, Lh) * Lh - Lo;

		let cosLi = dot(N, Li);

		if (cosLi > 0.0) {
			// Use Mipmap Filtered Importance Sampling to improve convergence.
			// See: https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch20.html, section 20.4

			let cosLh = max(dot(N, Lh), 0.0);

			// GGX normal distribution function (D term) probability density function.
			// Scaling by 1/4 is due to change of density in terms of Lh to Li (and since N=V, rest of the scaling factor cancels out).
			let pdf = ndfGGX(cosLh, ROUGHNESS) * 0.25;

			// Solid angle associated with this sample.
			let ws = 1.0 / (f32(NumSamples) * pdf);

			// Mip level to sample from.
			let mipLevel = max(0.5 * log2(ws / wt) + 1.0, 0.0);

			color  += textureSampleLevel(source, SAMPLER, Li, mipLevel).rgb * cosLi;
			weight += cosLi;
		}
	}
	color /= weight;

	textureStore(result, vec2<i32>(id.xy), i32(id.z), vec4<f32>(color, 1.0));
}

@stage(compute) @workgroup_size(8, 8)
fn brdf(@builtin(global_invocation_id) id: vec3<u32>)
{
    // Get integration parameters.
    // Make sure viewing angle is non-zero to avoid divisions by zero (and subsequently NaNs).
	let cosLo = max(f32(id.x) / f32(textureDimensions(result).x), Epsilon);
	let roughness = f32(id.y) / f32(textureDimensions(result).y);

	// Derive tangent-space viewing vector from angle to normal (pointing towards +Z in this reference frame).
	let Lo = vec3<f32>(sqrt(1.0 - cosLo*cosLo), 0.0, cosLo);

	// We will now pre-integrate Cook-Torrance BRDF for a solid white environment and save results into a 2D LUT.
	// DFG1 & DFG2 are terms of split-sum approximation of the reflectance integral.
	// For derivation see: "Moving Frostbite to Physically Based Rendering 3.0", SIGGRAPH 2014, section 4.9.2.
	var DFG1 = 0.;
	var DFG2 = 0.;

	for(var i=0u; i<NumSamples; i++) {
		let u = sampleHammersley(i);

		// Sample directly in tangent/shading space since we don't care about reference frame as long as it's consistent.
		let Lh = sampleGGX(u.x, u.y, roughness);

		// Compute incident direction (Li) by reflecting viewing direction (Lo) around half-vector (Lh).
		let Li = 2.0 * dot(Lo, Lh) * Lh - Lo;

		let cosLi   = Li.z;
		let cosLh   = Lh.z;
		let cosLoLh = max(dot(Lo, Lh), 0.0);

		if(cosLi > 0.0) {
			let G  = gaSchlickGGX_IBL(cosLi, cosLo, roughness);
			let Gv = G * cosLoLh / (cosLh * cosLo);
			let Fc = pow(1.0 - cosLoLh, 5.);

			DFG1 += (1. - Fc) * Gv;
			DFG2 += Fc * Gv;
		}
	}

    textureStore(result, vec2<i32>(id.xy), 0, vec4(DFG1, DFG2, 0., 0.) / f32(NumSamples));
}