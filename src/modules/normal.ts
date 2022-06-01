import {GPU} from "../gpu";
import NORMAL from "../wgsl/normal.wgsl";

export class Normal
{
  private normalPipeline: GPUComputePipeline | undefined;
  private copyPipeline: GPUComputePipeline | undefined;
  private normalLayout: GPUBindGroupLayout | undefined;
  private copyLayout: GPUBindGroupLayout | undefined;
  private texture: GPUTexture | undefined;

  async render(texture: GPUTexture, size: number)
  {
    this.normalPipeline ??= GPU.device.createComputePipeline({
      layout: "auto", compute: { module: await GPU.compile(NORMAL), entryPoint: "compute" }
    });

    this.copyPipeline ??= GPU.device.createComputePipeline({
      layout: "auto", compute: { module: await GPU.compile(NORMAL), entryPoint: "copy" }
    });

    this.normalLayout ??= this.normalPipeline.getBindGroupLayout(0);
    this.copyLayout ??= this.copyPipeline.getBindGroupLayout(0);

    this.texture ??= GPU.device.createTexture({
      size: [size, size], format: "rgba16float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    });

    let cmd = GPU.device.createCommandEncoder();

    let normalBinding = GPU.device.createBindGroup({
      layout: this.normalLayout,
      entries: [
        { binding: 0, resource: texture.createView({ baseMipLevel: 0, mipLevelCount: 1 }) },
        { binding: 1, resource: this.texture.createView({ baseMipLevel: 0, mipLevelCount: 1 }) },
      ]
    });

    let copyBinding = GPU.device.createBindGroup({
      layout: this.copyLayout,
      entries: [
        { binding: 0, resource: this.texture.createView({ baseMipLevel: 0, mipLevelCount: 1 }) },
        { binding: 1, resource: texture.createView({ baseMipLevel: 0, mipLevelCount: 1 }) },
      ]
    })

    let pass = cmd.beginComputePass();
    pass.setPipeline(this.normalPipeline);
    pass.setBindGroup(0, normalBinding);
    pass.dispatchWorkgroups(Math.ceil(size / 8), Math.ceil(size / 8));

    pass.setPipeline(this.copyPipeline);
    pass.setBindGroup(0, copyBinding);
    pass.dispatchWorkgroups(Math.ceil(size / 8), Math.ceil(size / 8));

    pass.end();

    GPU.device.queue.submit([cmd.finish()]);
  }
}