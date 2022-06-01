import {GPU} from "../gpu";
import TERRAIN from "../wgsl/terrain.wgsl";
import {Maxmap} from "./maxmap";
import {Normal} from "./normal";

export class Terrain
{
  readonly size;
  readonly mips;

  readonly terrain: GPUTexture;

  private pipeline: GPUComputePipeline | undefined;
  private binding: GPUBindGroup | undefined;

  private static maxmap: Maxmap = new Maxmap();
  private static normal: Normal = new Normal();

  constructor(size: number, mips: number) {
    this.size = size;
    this.mips = mips;

    this.terrain = GPU.device.createTexture({
      size: [size, size], format: "rgba16float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      mipLevelCount: mips
    });
  }

  async render()
  {
    this.pipeline ??= GPU.device.createComputePipeline({
      layout: "auto",
      compute: { module: await GPU.compile(TERRAIN), entryPoint: "compute" }
    });

    this.binding ??= GPU.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: this.terrain.createView({ mipLevelCount: 1 }) }]
    });

    let cmd = GPU.device.createCommandEncoder();

    let pass = cmd.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.binding);
    pass.dispatchWorkgroups(this.size / 8, this.size / 8);
    pass.end();

    GPU.device.queue.submit([cmd.finish()]);

    await Terrain.maxmap.render(this.terrain, this.size, this.mips);
    await Terrain.normal.render(this.terrain, this.size);
  }
}