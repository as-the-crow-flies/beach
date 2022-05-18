let PI: f32 = 3.1415926535897932384626433832795028841971693993751058209749;
let iSteps: i32 = 32;
let jSteps: i32 = 16;

fn rsi(r0 : vec3<f32>, rd : vec3<f32>, sr : f32) -> vec2<f32>
{
    // ray-sphere intersection that assumes
    // the sphere is centered at the origin.
    // No intersection when result.x > result.y
    let a = dot(rd, rd);
    let b = 2.0 * dot(rd, r0);
    let c = dot(r0, r0) - (sr * sr);
    let d = (b*b) - 4.0*a*c;
    if (d < 0.0) { return vec2<f32>(1e5,-1e5); }
    return vec2<f32>(
        (-b - sqrt(d))/(2.0*a),
        (-b + sqrt(d))/(2.0*a)
    );
}

fn stc(radius: f32, inclination: f32, azimuth: f32) -> vec3<f32>
{
    return radius * vec3<f32>(
        sin(inclination) * cos(azimuth),
        cos(inclination),
        sin(inclination) * sin(azimuth)
    );
}

fn atmosphere(r : vec3<f32>, r0 : vec3<f32>, pSun : vec3<f32>, iSun : f32, rPlanet : f32, rAtmos : f32, kRlh : vec3<f32>, kMie : f32, shRlh : f32, shMie : f32, g : f32) -> vec3<f32>
{
    // Normalize the sun and view directions.
    // pSun = normalize(pSun);
    // r = normalize(r);

    // Calculate the step size of the primary ray.
    var p = rsi(r0, r, rAtmos);
    if (p.x > p.y) { return vec3<f32>(0.); }
    p.y = min(p.y, rsi(r0, r, rPlanet).x);
    let iStepSize = (p.y - p.x) / f32(iSteps);

    // Initialize the primary ray time.
    var iTime = 0.0;

    // Initialize accumulators for Rayleigh and Mie scattering.
    var totalRlh = vec3<f32>(0.);
    var totalMie = vec3<f32>(0.);

    // Initialize optical depth accumulators for the primary ray.
    var iOdRlh = 0.0;
    var iOdMie = 0.0;

    // Calculate the Rayleigh and Mie phases.
    let mu = dot(r, pSun);
    let mumu = mu * mu;
    let gg = g * g;
    let pRlh = 3.0 / (16.0 * PI) * (1.0 + mumu);
    let pMie = 3.0 / (8.0 * PI) * ((1.0 - gg) * (mumu + 1.0)) / (pow(1.0 + gg - 2.0 * mu * g, 1.5) * (2.0 + gg));

    // Sample the primary ray.
    for (var i = 0; i < iSteps; i++) {

        // Calculate the primary ray sample position.
        let iPos = r0 + r * (iTime + iStepSize * 0.5);

        // Calculate the height of the sample.
        let iHeight = length(iPos) - rPlanet;

        // Calculate the optical depth of the Rayleigh and Mie scattering for this step.
        let odStepRlh = exp(-iHeight / shRlh) * iStepSize;
        let odStepMie = exp(-iHeight / shMie) * iStepSize;

        // Accumulate optical depth.
        iOdRlh = iOdRlh + odStepRlh;
        iOdMie = iOdMie + odStepMie;

        // Calculate the step size of the secondary ray.
        let jStepSize = rsi(iPos, pSun, rAtmos).y / f32(jSteps);

        // Initialize the secondary ray time.
        var jTime = 0.0;

        // Initialize optical depth accumulators for the secondary ray.
        var jOdRlh = 0.0;
        var jOdMie = 0.0;

        // Sample the secondary ray.
        for (var j = 0; j < jSteps; j++) {

            // Calculate the secondary ray sample position.
            let jPos = iPos + pSun * (jTime + jStepSize * 0.5);

            // Calculate the height of the sample.
            let jHeight = length(jPos) - rPlanet;

            // Accumulate the optical depth.
            jOdRlh = jOdRlh + exp(-jHeight / shRlh) * jStepSize;
            jOdMie = jOdMie + exp(-jHeight / shMie) * jStepSize;

            // Increment the secondary ray time.
            jTime = jTime + jStepSize;
        }

        // Calculate attenuation.
        let attn = exp(-(kMie * (iOdMie + jOdMie) + kRlh * (iOdRlh + jOdRlh)));

        // Accumulate scattering.
        totalRlh = totalRlh + odStepRlh * attn;
        totalMie = totalMie + odStepMie * attn;

        // Increment the primary ray time.
        iTime = iTime + iStepSize;

    }

    // Calculate and return the final color.
    return iSun * (pRlh * kRlh * totalRlh + pMie * kMie * totalMie);
}

fn atmos(r: vec3<f32>, r0: vec3<f32>, pSun: vec3<f32>) -> vec3<f32>
{
    return atmosphere(
           r,                                      // normalized ray direction
           r0,                                     // ray origin
           pSun,                                   // position of the sun
           22.0,                                   // intensity of the sun
           6371e3,                                 // radius of the planet in meters
           6471e3,                                 // radius of the atmosphere in meters
           vec3<f32>(5.5e-6, 13.0e-6, 22.4e-6),    // Rayleigh scattering coefficient
           21e-6,                                  // Mie scattering coefficient
           8e3,                                    // Rayleigh scale height
           1.2e3,                                  // Mie scale height
           0.758                                   // Mie preferred scattering direction
       );
}

@group(0) @binding(0) var result : texture_storage_2d_array<rgba16float, write>;

@stage(compute) @workgroup_size(8, 8)
fn compute(@builtin(global_invocation_id) id: vec3<u32>)
{
    let sun = normalize(stc(1., 1., 0.));
    let earth = vec3<f32>(0., 6371e3, 0.);

    let ray = normalize(getSamplingVector(id, textureDimensions(result)));

    textureStore(result, vec2<i32>(id.xy), i32(id.z), vec4<f32>(atmos(ray, earth, sun), 1.));
}
