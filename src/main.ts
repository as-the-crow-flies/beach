import QUAD from './wgsl/quad.wgsl';
import COMMON from './wgsl/common.wgsl';
import MAIN from './wgsl/main.wgsl';

import {GPU} from "./gpu";
import {animate, stc} from "./utils";

import {mat4} from 'gl-matrix';
import {Environment} from "./modules/environment";

(async () => {

  await GPU.init();

  const CUBE_MAP_SIZE = 32;
  const CUBE_MAP_MIPMAPS = Math.log2(CUBE_MAP_SIZE) - Math.log2(16);

  let environment = new Environment(CUBE_MAP_SIZE, CUBE_MAP_MIPMAPS);

  let camera = GPU.device.createBuffer({
    size: 34 *  4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  let sampler = GPU.device.createSampler({
    minFilter: "linear", magFilter: "linear", mipmapFilter: "linear"
  });

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
      { binding: 2, resource: environment.environment.createView({ dimension: "cube" }) },
      { binding: 3, resource: environment.irradiance.createView({ dimension: "cube" })},
      { binding: 4, resource: environment.specular.createView({ dimension: "cube" })},
      { binding: 5, resource: environment.brdf.createView() }
    ]
  });

  let view = mat4.create();
  let projection = mat4.create();

  let cameraInclination = Math.PI / 2, cameraAzimuth = 0., cameraRadius = 10,
      sunInclination = Math.PI / 3, sunAzimuth = Math.PI;

  await environment.render(stc(1, sunInclination, sunAzimuth));

  GPU.canvas.addEventListener('contextmenu', event => event.preventDefault());
  window.addEventListener("mousemove", async e => {
    if (e.buttons == 1)
    {
      cameraInclination = Math.min(Math.max(cameraInclination + .003 * e.movementY, 0), Math.PI);
      cameraAzimuth = (cameraAzimuth + .003 * e.movementX) % (2 * Math.PI);
    }
    else if (e.buttons == 2)
    {
      sunInclination = Math.min(Math.max(sunInclination + .003 * e.movementY, 0), Math.PI);
      sunAzimuth = (sunAzimuth - .003 * e.movementX) % (2 * Math.PI);
      await environment.render(stc(1, sunInclination, sunAzimuth));
    }
  });

  window.addEventListener("wheel", e => {
    cameraRadius += .001 * e.deltaY
  });

  animate(async () => {
    let WIDTH = window.innerWidth, HEIGHT = window.innerHeight, NEAR = 0.0001, FAR = 10000, FOV = Math.PI/2;

    mat4.lookAt(view, stc(cameraRadius, cameraInclination, cameraAzimuth), [0, 0, 0], [0, 1, 0]);
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
