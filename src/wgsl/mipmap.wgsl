@group(0) @binding(0) var SAMPLER   : sampler;
@group(0) @binding(1) var source    : texture_2d<f32>;

@stage(fragment)
fn fragment(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32>
{
    return textureSample(source, SAMPLER, .5 - .5 * uv);
}