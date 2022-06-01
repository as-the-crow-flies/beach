import QUAD from './wgsl/quad.wgsl';
import COMMON from './wgsl/common.wgsl';
import MAIN from './wgsl/main.wgsl';

import {GPU} from "./gpu";
import {animate, stc} from "./utils";

import {mat4} from 'gl-matrix';
import {Environment} from "./modules/environment";
import {Terrain} from "./modules/terrain";

(async () => {

  await GPU.init();

  const CUBE_MAP_SIZE = 128;
  const CUBE_MAP_MIPMAPS = Math.log2(CUBE_MAP_SIZE) - Math.log2(2);

  const SIMULATION_SIZE = 2048;
  const SIMULATION_MIPMAPS  = Math.log2(SIMULATION_SIZE);

  let environment = new Environment(CUBE_MAP_SIZE, CUBE_MAP_MIPMAPS);
  let terrain = new Terrain(SIMULATION_SIZE, SIMULATION_MIPMAPS);

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
      targets: [{ format: GPU.FORMAT_CANVAS }],
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
      { binding: 5, resource: environment.brdf.createView() },
      { binding: 6, resource: terrain.terrain.createView() }
    ]
  });

  let view = mat4.create();
  let projection = mat4.create();

  let cameraInclination = 0.711197551196597,
      cameraAzimuth = -2.354388980384688,
      cameraRadius = 500,
      cameraOrigin = [SIMULATION_SIZE / 2, 0, SIMULATION_SIZE / 2] as [number, number, number],
      sunInclination = Math.PI / 3, sunAzimuth = Math.PI;

  // Initialize
  await environment.render(stc(1, sunInclination, sunAzimuth));
  await terrain.render();

  GPU.canvas.addEventListener('contextmenu', event => event.preventDefault());
  window.addEventListener("mousemove", async e => {
    if (e.buttons == 1)
    {
      cameraInclination = Math.min(Math.max(cameraInclination + .003 * e.movementY, 1E-6), Math.PI);
      cameraAzimuth = (cameraAzimuth - .003 * e.movementX) % (2 * Math.PI);
    }
    else if (e.buttons == 2)
    {
      sunInclination = Math.min(Math.max(sunInclination + .003 * e.movementY, 0.), Math.PI);
      sunAzimuth = (sunAzimuth - .003 * e.movementX) % (2 * Math.PI);

      await environment.render(stc(1, sunInclination, sunAzimuth));
    }
  });

  window.addEventListener("mousedown", async _ => {
    await environment.render(stc(1, sunInclination, sunAzimuth));
  });

  window.addEventListener("wheel", e => {
    e.preventDefault()
    cameraRadius += .0005 * cameraRadius * e.deltaY
  }, { passive: false });

  animate(async () => {
    let WIDTH = window.innerWidth, HEIGHT = window.innerHeight, NEAR = 0.0001, FAR = 10000, FOV = 2.;

    mat4.lookAt(view, stc(cameraRadius, cameraInclination, cameraAzimuth, cameraOrigin), cameraOrigin, [0, 1, 0]);
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
