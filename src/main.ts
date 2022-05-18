import QUAD from './wgsl/quad.wgsl';
import COMMON from './wgsl/common.wgsl';
import MIPMAP from './wgsl/mipmap.wgsl';
import MAIN from './wgsl/main.wgsl';
import ATMOSPHERE from './wgsl/atmosphere.wgsl';
import PBRMAP from './wgsl/pbrmap.wgsl';

import {GPU} from "./gpu";
import {animate, stc} from "./utils";

import {mat4} from 'gl-matrix';

// [+X, -X, +Y, -Y, +Z, -Z]
const DIRECTIONS: [number, number, number][] = [[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]];

(async () => {

  await GPU.init();

  const CUBE_MAP_SIZE = 128;
  const CUBE_MAP_MIPMAPS = Math.log2(CUBE_MAP_SIZE) - Math.log2(16);

  const BRDF_LUT_SIZE = 256;

  const vertex : GPUVertexState = {
    module: await GPU.compile(QUAD), entryPoint:  "vertex"
  }

  let environment = GPU.device.createTexture({
    size: [CUBE_MAP_SIZE, CUBE_MAP_SIZE, 6], dimension: "2d", format: GPU.FORMAT_DATA,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    mipLevelCount: CUBE_MAP_MIPMAPS
  });

  let irmap = GPU.device.createTexture({
    size: [CUBE_MAP_SIZE, CUBE_MAP_SIZE, 6], dimension: "2d", format: GPU.FORMAT_DATA,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    mipLevelCount: CUBE_MAP_MIPMAPS
  });

  let spmap = GPU.device.createTexture({
    size: [CUBE_MAP_SIZE, CUBE_MAP_SIZE, 6], dimension: "2d", format: GPU.FORMAT_DATA,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    mipLevelCount: CUBE_MAP_MIPMAPS
  });

  let brdf = GPU.device.createTexture({
    size: [BRDF_LUT_SIZE, BRDF_LUT_SIZE], dimension: "2d", format: GPU.FORMAT_DATA,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
  });

  let camera = GPU.device.createBuffer({
    size: 34 *  4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  let sampler = GPU.device.createSampler({
    minFilter: "linear", magFilter: "linear", mipmapFilter: "linear"
  });

  // Render Environment to Cube Map
  {
    let environmentPipeline = GPU.device.createRenderPipeline({
      layout: "auto", vertex,
      fragment: {
        module: await GPU.compile(COMMON + ATMOSPHERE),
        entryPoint: "fragment",
        targets: [{ format: GPU.FORMAT_DATA }]
      }
    });

    let mipmapPipeline = GPU.device.createRenderPipeline({
      layout: "auto", vertex,
      fragment: {
        module: await GPU.compile(MIPMAP),
        entryPoint: "fragment",
        targets: [{ format: GPU.FORMAT_DATA }]
      }
    });

    let irPipeline = GPU.device.createComputePipeline({
      layout: "auto", compute: { module: await GPU.compile(PBRMAP), entryPoint: "irmap" }
    });

    let spPipeline = GPU.device.createComputePipeline({
      layout: "auto", compute: { module: await GPU.compile(PBRMAP), entryPoint: "spmap" }
    });

    let brdfPipeline = GPU.device.createComputePipeline({
      layout: "auto", compute: { module: await GPU.compile(PBRMAP), entryPoint: "brdf" }
    });

    let cameraBinding = GPU.device.createBindGroup({
      layout: environmentPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: camera }}]
    });

    let projection = mat4.perspective(mat4.create(), Math.PI / 2, 1, 0.01, 100);
    mat4.invert(projection, projection);

    // Render Cube Face
    for (let [index, direction] of DIRECTIONS.entries())
    {
      let cmd = GPU.device.createCommandEncoder();

      direction[2] += 1E-6; // Looking straight up or down results in artefacts...

      let view = mat4.lookAt(mat4.create(), [0, 0, 0], direction, [0, 1, 0]);
      mat4.invert(view, view);

      GPU.device.queue.writeBuffer(camera, 0, new Float32Array([
        ...view, ...projection, CUBE_MAP_SIZE, CUBE_MAP_SIZE
      ]));

      let environmentPass = cmd.beginRenderPass({
        colorAttachments: [{
          view: environment.createView({
            dimension: "2d", baseArrayLayer: index, arrayLayerCount: 1, baseMipLevel: 0, mipLevelCount: 1 }),
          storeOp: "store", loadOp: "clear", clearValue: [0, 0, 0, 1]
        }]
      });

      environmentPass.setPipeline(environmentPipeline);
      environmentPass.setBindGroup(0, cameraBinding);
      environmentPass.draw(3);
      environmentPass.end();

      GPU.device.queue.submit([cmd.finish()]);

      // Compute Mipmaps
      for (let i=1; i<CUBE_MAP_MIPMAPS; i++)
      {
        let cmd = GPU.device.createCommandEncoder();

        let mipmapBinding = GPU.device.createBindGroup({
          layout: mipmapPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: sampler },
            {
              binding: 1, resource: environment.createView({
                dimension: "2d", baseArrayLayer: index, arrayLayerCount: 1, baseMipLevel: i-1, mipLevelCount: 1 })
            }
          ]
        });

        let mipmapPass = cmd.beginRenderPass({
          colorAttachments: [{
            view: environment.createView({
              dimension: "2d", baseArrayLayer: index, arrayLayerCount: 1, baseMipLevel: i, mipLevelCount: 1
            }),
            storeOp: "store", loadOp: "clear", clearValue: [0, 0, 0, 1]
          }]
        });

        mipmapPass.setPipeline(mipmapPipeline);
        mipmapPass.setBindGroup(0, mipmapBinding);
        mipmapPass.draw(3);
        mipmapPass.end();

        GPU.device.queue.submit([cmd.finish()]);
      }
    }


    let size = CUBE_MAP_SIZE;
    for (let i=0; i<CUBE_MAP_MIPMAPS; i++)
    {
      let cmd = GPU.device.createCommandEncoder();

      let roughness = GPU.device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      GPU.device.queue.writeBuffer(roughness, 0, new Float32Array([i/CUBE_MAP_MIPMAPS]));

      // Compute irradiance map
      let irBinding = GPU.device.createBindGroup({
        layout: irPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: sampler },
          { binding: 1, resource: environment.createView({ baseMipLevel: i, mipLevelCount: 1, dimension: "cube" }) },
          { binding: 2, resource: irmap.createView({ baseMipLevel: i, mipLevelCount: 1, dimension: "2d-array" })}
        ]
      });

      let irPass = cmd.beginComputePass();
      irPass.setPipeline(irPipeline);
      irPass.setBindGroup(0, irBinding);
      irPass.dispatchWorkgroups(size / 8, size / 8, 6);
      irPass.end();

      let spBinding = GPU.device.createBindGroup({
        layout: spPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: sampler },
          { binding: 1, resource: environment.createView({ baseMipLevel: i, mipLevelCount: 1, dimension: "cube" }) },
          { binding: 2, resource: spmap.createView({ baseMipLevel: i, mipLevelCount: 1, dimension: "2d-array" })},
          { binding: 3, resource: { buffer: roughness }}
        ]
      });

      // Compute specular map
      let spPass = cmd.beginComputePass();
      spPass.setPipeline(spPipeline);
      spPass.setBindGroup(0, spBinding);
      spPass.dispatchWorkgroups(size / 8, size / 8, 6);
      spPass.end();

      GPU.device.queue.submit([cmd.finish()]);

      size /= 2;
    }

    // Compute BRDF LUT
    let cmd = GPU.device.createCommandEncoder();

    let brdfBinding = GPU.device.createBindGroup({
      layout: brdfPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 2, resource: brdf.createView({ dimension: "2d-array" })}
      ]
    });

    let brdfPass = cmd.beginComputePass();
    brdfPass.setPipeline(brdfPipeline);
    brdfPass.setBindGroup(0, brdfBinding);
    brdfPass.dispatchWorkgroups(BRDF_LUT_SIZE / 8, BRDF_LUT_SIZE / 8, 1);
    brdfPass.end();

    GPU.device.queue.submit([cmd.finish()]);
  }

  let mainPipeline = GPU.device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: await GPU.compile(QUAD),
      entryPoint: "vertex"
    },
    fragment: {
      module: await GPU.compile(COMMON + MAIN),
      entryPoint: "fragment",
      targets: [{ format: GPU.FORMAT_CANVAS }]
    }
  });

  let binding = GPU.device.createBindGroup({
    layout: mainPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera }},
      { binding: 1, resource: sampler },
      { binding: 2, resource: environment.createView({ dimension: "cube" }) },
      { binding: 3, resource: irmap.createView({ dimension: "cube" })},
      { binding: 4, resource: spmap.createView({ dimension: "cube" })},
      { binding: 5, resource: brdf.createView() }
    ]
  });

  let view = mat4.create();
  let projection = mat4.create();

  let inclination = Math.PI / 2, azimuth = 0., radius = 10;

  window.addEventListener("mousemove", e => {
    if (e.buttons == 1)
    {
      inclination = Math.min(Math.max(inclination + .003 * e.movementY, 0), Math.PI);
      azimuth = (azimuth + .003 * e.movementX) % (2 * Math.PI);
    }
  });

  window.addEventListener("wheel", e => {
    radius += .001 * e.deltaY
  });

  animate(() => {

    azimuth += 0.02;

    let WIDTH = window.innerWidth, HEIGHT = window.innerHeight, NEAR = 0.0001, FAR = 10000, FOV = Math.PI/2;

    mat4.lookAt(view, stc(radius, inclination, azimuth), [0, 0, 0], [0, 1, 0]);
    mat4.perspective(projection, FOV, WIDTH / HEIGHT, NEAR, FAR);

    mat4.invert(view, view);
    mat4.invert(projection, projection);

    GPU.device.queue.writeBuffer(camera, 0, new Float32Array([...view, ...projection, WIDTH, HEIGHT]));

    let cmd = GPU.device.createCommandEncoder();

    let pass = cmd.beginRenderPass({
      colorAttachments: [{
        view: GPU.context.getCurrentTexture().createView(),
        storeOp: "store", loadOp: "clear", clearValue: [0, 0, 0, 1]
      }]
    });

    pass.setPipeline(mainPipeline);
    pass.setBindGroup(0, binding);
    pass.draw(3);
    pass.end();

    GPU.device.queue.submit([cmd.finish()]);
  });
})();
