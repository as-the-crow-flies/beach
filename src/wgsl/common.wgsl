struct Camera
{
    iV          : mat4x4<f32>,
    iP          : mat4x4<f32>
};

fn createRay(uv: vec2<f32>, PInv : mat4x4<f32>, VInv : mat4x4<f32>) -> vec3<f32>
{
    // choose an arbitrary point in the viewing volume
    // z = -1 equals a point on the near plane, i.e. the screen
    let pointNDS = vec3<f32>(uv, -1.);

    // as this is in homogenous space, add the last homogenous coordinate
    let pointNDSH = vec4<f32>(pointNDS, 1.0);
    // transform by inverse projection to get the point in view space
    var dirEye = PInv * pointNDSH;

    // since the camera is at the origin in view space by definition,
    // the current point is already the correct direction
    // (dir(0,P) = P - 0 = P as a direction, an infinite point,
    // the homogenous component becomes 0 the scaling done by the
    // w-division is not of interest, as the direction in xyz will
    // stay the same and we can just normalize it later
    dirEye.w = 0.;

    // compute world ray direction by multiplying the inverse view matrix
    let dirWorld = (VInv * dirEye).xyz;

    // now normalize direction
    return normalize(dirWorld);
}

// Calculate normalized sampling direction vector based on current fragment coordinates (gl_GlobalInvocationID.xyz).
// This is essentially "inverse-sampling": we reconstruct what the sampling vector would be if we wanted it to "hit"
// this particular fragment in a cubemap.
// See: OpenGL core profile specs, section 8.13.
fn getSamplingVector(id : vec3<u32>, dim: vec2<i32>) -> vec3<f32>
{
    let step = 1. / vec2<f32>(dim);

    let st = vec2<f32>(id.xy) * step;
    let uv = 2. * vec2(st.x, 1. - st.y) - 1.;

    let puv = vec2<f32>(uv.x + step.x, uv.y - step.y);
    let nuv = vec2<f32>(-uv.x - step.x, -uv.y + step.y);

   switch (id.z)
   {
    case 0u { return vec3<f32>( 1.  , puv.y, nuv.x); }
    case 1u { return vec3<f32>(-1.  , puv.y, puv.x); }
    case 2u { return vec3<f32>( puv.x, 1.  , nuv.y); }
    case 3u { return vec3<f32>( puv.x,-1.  , puv.y); }
    case 4u { return vec3<f32>( puv.x, puv.y, 1.  ); }
    case 5u { return vec3<f32>( nuv.x, puv.y,-1.  ); }
    default { return vec3<f32>(0.); }
   };
}