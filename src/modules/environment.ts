import {GPU} from "../gpu";
import COMMON from "../wgsl/common.wgsl";
import ENVIRONMENT from "../wgsl/environment.wgsl";
import MIPMAP from "../wgsl/mipmap.wgsl";
import PBRMAP from "../wgsl/pbrmap.wgsl";
import {stc} from "../utils";

export class Environment
{
  size: number;
  mips: number;

  private readonly sampler: GPUSampler;
  private readonly parameters: GPUBuffer;

  readonly environment: GPUTexture;
  private environmentPipeline: GPUComputePipeline | undefined;
  private environmentBinding: GPUBindGroup | undefined;

  private mipmapPipeline: GPUComputePipeline | undefined;
  private mipmapBindings: GPUBindGroup[] = [];

  readonly irradiance: GPUTexture;
  private irradiancePipeline: GPUComputePipeline | undefined;
  private irradianceBindings: GPUBindGroup[] = [];

  readonly specular: GPUTexture;
  private specularPipeline: GPUComputePipeline | undefined;
  private specularBindings: GPUBindGroup[] = [];

  readonly brdf: GPUTexture;
  private brdfPipeline: GPUComputePipeline | undefined;
  private brdfBinding: GPUBindGroup | undefined;

  constructor(size: number, mips: number) {
    this.size = size;
    this.mips = mips;

    this.sampler = GPU.device.createSampler({
      minFilter: "linear", magFilter: "linear", mipmapFilter: "linear"
    });

    this.parameters = GPU.device.createBuffer({
      size: 13 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    this.environment = GPU.device.createTexture({
      size: [size, size, 6], dimension: "2d", format: GPU.FORMAT_DATA,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      mipLevelCount: mips
    });

    this.irradiance = GPU.device.createTexture({
      size: [size, size, 6], dimension: "2d", format: GPU.FORMAT_DATA,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      mipLevelCount: mips
    });

    this.specular = GPU.device.createTexture({
      size: [size, size, 6], dimension: "2d", format: GPU.FORMAT_DATA,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      mipLevelCount: mips
    });

    this.brdf = GPU.device.createTexture({
      size: [size, size], dimension: "2d", format: GPU.FORMAT_DATA,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    });
  }

  async render(
      sunPosition: [number, number, number] = stc(1., 1., 0),
      sunIntensity: number = 22.0,
      rayleighCoeff: [number, number, number] = [5.5e-6, 13.0e-6, 22.4e-6],
      rayleighScale: number = 8e3,
      planetRadius: number = 6371e3,
      atmosphereRadius: number = 6471e3,
      mieCoeff: number = 21e-6,
      mieScale: number = 1.2e3,
      mieDirection: number = 0.758)
  {
    GPU.device.queue.writeBuffer(this.parameters, 0, new Float32Array([
        ...sunPosition, sunIntensity, ...rayleighCoeff, rayleighScale,
      planetRadius, atmosphereRadius, mieCoeff, mieScale, mieDirection
    ]))

    let cmd = GPU.device.createCommandEncoder();

    if (!this.brdfPipeline)
    {
      this.brdfPipeline = GPU.device.createComputePipeline({
        layout: "auto", compute: { module: await GPU.compile(COMMON + PBRMAP), entryPoint: "brdf" }
      });

      this.brdfBinding = GPU.device.createBindGroup({
        layout: this.brdfPipeline.getBindGroupLayout(0),
        entries: [{ binding: 2, resource: this.brdf.createView({ dimension: "2d-array" })}]
      });

      // Compute BRDF LUT (One Time)
      let brdfComputePass = cmd.beginComputePass();
      brdfComputePass.setPipeline(this.brdfPipeline);
      brdfComputePass.setBindGroup(0, this.brdfBinding);
      brdfComputePass.dispatchWorkgroups(this.size / 8, this.size / 8, 1);
      brdfComputePass.end();
    }

    this.environmentPipeline ??= GPU.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: await GPU.compile(COMMON + ENVIRONMENT),
        entryPoint: "compute"
      }
    });

    this.mipmapPipeline ??= GPU.device.createComputePipeline({
      layout: "auto", compute: { module: await GPU.compile(MIPMAP), entryPoint: "compute" }
    });

    this.irradiancePipeline ??= GPU.device.createComputePipeline({
      layout: "auto", compute: { module: await GPU.compile(COMMON + PBRMAP), entryPoint: "irmap" }
    });

    this.specularPipeline ??= GPU.device.createComputePipeline({
      layout: "auto", compute: { module: await GPU.compile(COMMON + PBRMAP), entryPoint: "spmap" }
    });

    this.environmentBinding ??= GPU.device.createBindGroup({
      layout: this.environmentPipeline.getBindGroupLayout(0),
      entries: [
          { binding: 0, resource: this.environment.createView({ dimension: "2d-array", mipLevelCount: 1})},
          { binding: 1, resource: { buffer: this.parameters }}
      ]
    });

    // Render Environment Cube Map
    let environmentComputePass = cmd.beginComputePass();
    environmentComputePass.setPipeline(this.environmentPipeline);
    environmentComputePass.setBindGroup(0, this.environmentBinding);
    environmentComputePass.dispatchWorkgroups(this.size / 8, this.size / 8, 6);
    environmentComputePass.end();

    let s = this.size;
    for (let i=0; i<this.mips; i++)
    {
      if (i >= 1)  // The first mipmap is already rendered
      {
        this.mipmapBindings[i] ??= GPU.device.createBindGroup({
          layout: this.mipmapPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: this.environment.createView({ dimension: "2d-array", baseMipLevel: i-1, mipLevelCount: 1 })},
            { binding: 1, resource: this.environment.createView({ dimension: "2d-array", baseMipLevel: i  , mipLevelCount: 1 })}
          ]
        });

        // Compute Mipmap
        let mipmapComputePass = cmd.beginComputePass();
        mipmapComputePass.setPipeline(this.mipmapPipeline);
        mipmapComputePass.setBindGroup(0, this.mipmapBindings[i]);
        mipmapComputePass.dispatchWorkgroups(s / 8, s / 8, 6);
        mipmapComputePass.end();
      }

      this.irradianceBindings[i] ??= GPU.device.createBindGroup({
        layout: this.irradiancePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.sampler },
          { binding: 1, resource: this.environment.createView({ baseMipLevel: i, mipLevelCount: 1, dimension: "cube" }) },
          { binding: 2, resource: this.irradiance.createView({ baseMipLevel: i, mipLevelCount: 1, dimension: "2d-array" })}
        ]
      });

      // Compute Irradiance Map
      let irradianceComputePass = cmd.beginComputePass();
      irradianceComputePass.setPipeline(this.irradiancePipeline);
      irradianceComputePass.setBindGroup(0, this.irradianceBindings[i]);
      irradianceComputePass.dispatchWorkgroups(s / 8, s / 8, 6);
      irradianceComputePass.end();

      let roughness = GPU.device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      GPU.device.queue.writeBuffer(roughness, 0, new Float32Array([i/this.mips]));

      this.specularBindings[i] ??= GPU.device.createBindGroup({
        layout: this.specularPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.sampler },
          { binding: 1, resource: this.environment.createView({ baseMipLevel: i, mipLevelCount: 1, dimension: "cube" }) },
          { binding: 2, resource: this.specular.createView({ baseMipLevel: i, mipLevelCount: 1, dimension: "2d-array" })},
          { binding: 3, resource: { buffer: roughness }}
        ]
      });

      // Compute Specular Map
      let specularComputePass = cmd.beginComputePass();
      specularComputePass.setPipeline(this.specularPipeline);
      specularComputePass.setBindGroup(0, this.specularBindings[i]);
      specularComputePass.dispatchWorkgroups(s / 8, s / 8, 6);
      specularComputePass.end();

      s /= 2;
    }

    GPU.device.queue.submit([cmd.finish()]);
  }
}