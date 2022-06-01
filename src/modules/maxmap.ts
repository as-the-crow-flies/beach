import {GPU} from "../gpu";

import MAXMAP from '../wgsl/maxmap.wgsl';

export class Maxmap
{
  private pipeline: GPUComputePipeline | undefined;
  private layout: GPUBindGroupLayout | undefined;

  async render(texture: GPUTexture, size: number, mips: number)
  {
    this.pipeline ??= GPU.device.createComputePipeline({
      layout: "auto", compute: { module: await GPU.compile(MAXMAP), entryPoint: "compute" }
    });

    this.layout ??= this.pipeline.getBindGroupLayout(0);

    let cmd = GPU.device.createCommandEncoder();

    let pass = cmd.beginComputePass();
    pass.setPipeline(this.pipeline);

    let bindings = [];

    let s = size;
    for (let i = 1; i < mips; i++) {
      s /= 2;

      bindings[i] = GPU.device.createBindGroup({
        layout: this.layout,
        entries: [
            { binding: 0, resource: texture.createView({ baseMipLevel: i-1, mipLevelCount: 1 }) },
            { binding: 1, resource: texture.createView({ baseMipLevel: i  , mipLevelCount: 1 }) }
        ]
      });

      pass.setBindGroup(0, bindings[i]);
      pass.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8));
    }

    pass.end();

    GPU.device.queue.submit([cmd.finish()]);
  }
}