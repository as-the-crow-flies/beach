export class GPU
{
  static FORMAT_CANVAS : GPUTextureFormat = "bgra8unorm";
  static FORMAT_DATA : GPUTextureFormat = "rgba16float";

  static device: GPUDevice;
  static canvas: HTMLCanvasElement;
  static context: GPUCanvasContext;

  static async init()
  {
    this.device = await (await navigator.gpu.requestAdapter())?.requestDevice()!;
    if (!this.device) throw new Error("Unable to obtain WebGPU Device");
    this.canvas = document.getElementById("canvas") as HTMLCanvasElement;
    this.context = this.canvas.getContext("webgpu") as unknown as GPUCanvasContext;

    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;

    this.context.configure({
      device: this.device,
      format: this.FORMAT_CANVAS,
      size: { width: this.canvas.width * window.devicePixelRatio, height: this.canvas.height * window.devicePixelRatio },
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    });

    console.log(this.canvas.width, this.canvas.height);
  }

  static async compile(source: string)
  {
    const shader = this.device.createShaderModule({ code: source });
    await shader.compilationInfo()
    return shader;
  }
}