@group(0) @binding(0) var source    : texture_2d<f32>;
@group(0) @binding(1) var result    : texture_storage_2d<rgba16float, write>;

@stage(compute) @workgroup_size(8, 8)
fn compute(@builtin(global_invocation_id) id: vec3<u32>)
{
    let uv = vec2<i32>(id.xy);
    let dx = vec2<i32>(1, 0);
    let dy = vec2<i32>(0, 1);

    let normal = normalize(vec3<f32>(
        textureLoad(source, uv - dx, 0).x - textureLoad(source, uv + dx, 0).x,
        2.,
        textureLoad(source, uv - dy, 0).x - textureLoad(source, uv + dy, 0).x
    ));

    let height = textureLoad(source, uv, 0).x;

    textureStore(result, uv, vec4<f32>(height, normal));
}

@stage(compute) @workgroup_size(8, 8)
fn copy(@builtin(global_invocation_id) id: vec3<u32>)
{
    let uv = vec2<i32>(id.xy);
    textureStore(result, uv, textureLoad(source, uv, 0));
}
