@group(0) @binding(0) var result    : texture_storage_2d<rgba16float, write>;

@stage(compute) @workgroup_size(8, 8)
fn compute(@builtin(global_invocation_id) id: vec3<u32>)
{
    let uv = vec2<f32>(id.xy) / vec2<f32>(textureDimensions(result).xy);
    let height = 100. + 50. * sin(30. * uv.x) + 50. * cos(10. * uv.y);
    textureStore(result, vec2<i32>(id.xy), vec4<f32>(vec3<f32>(height), 1.));
}
