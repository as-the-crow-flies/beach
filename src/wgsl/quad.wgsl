let QUAD = array<vec2<f32>, 3>(
    vec2<f32>( 1., 1.),
    vec2<f32>(-3., 1.),
    vec2<f32>( 1.,-3.));

struct Quad
{
    @builtin(position) position : vec4<f32>,
    @location(0)       uv       : vec2<f32>
};

@stage(vertex)
fn vertex(@builtin(vertex_index) i : u32) -> Quad
{
    var quad: Quad;
    quad.uv = QUAD[i] * vec2<f32>(-1., 1.);
    quad.position = vec4<f32>(QUAD[i], 0., 1.);

    return quad;
}
