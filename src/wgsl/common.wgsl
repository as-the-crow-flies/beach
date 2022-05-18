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
