@group(0) @binding(0) var source    : texture_2d_array<f32>;
@group(0) @binding(1) var result    : texture_storage_2d_array<rgba16float, write>;

@stage(compute) @workgroup_size(8, 8)
fn compute(@builtin(global_invocation_id) id: vec3<u32>)
{
    let index = 2 * vec2<i32>(id.xy);
    let layer = i32(id.z);

    textureStore(result, vec2<i32>(id.xy), layer, 0.25 * (
        textureLoad(source, index                  , layer, 0) +
        textureLoad(source, index + vec2<i32>(1, 0), layer, 0) +
        textureLoad(source, index + vec2<i32>(0, 1), layer, 0) +
        textureLoad(source, index + vec2<i32>(1, 1), layer, 0)));
}
