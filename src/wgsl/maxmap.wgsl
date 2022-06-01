@group(0) @binding(0) var source    : texture_2d<f32>;
@group(0) @binding(1) var result    : texture_storage_2d<rgba16float, write>;

@stage(compute) @workgroup_size(8, 8)
fn compute(@builtin(global_invocation_id) id: vec3<u32>)
{
    let size = vec2<u32>(textureDimensions(result));

    if (id.x >= size.x || id.y >= size.y) { return; }

    let index = 2 * vec2<i32>(id.xy);

    textureStore(result, vec2<i32>(id.xy), max(max(max(
        textureLoad(source, index                  , 0),
        textureLoad(source, index + vec2<i32>(1, 0), 0)),
        textureLoad(source, index + vec2<i32>(0, 1), 0)),
        textureLoad(source, index + vec2<i32>(1, 1), 0)));
}
