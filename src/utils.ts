export const animate = (callback: (t: number, dt: number) => void) => {
  let pt = 0;
  const animation = (time: DOMHighResTimeStamp) =>
  {
    time *= .001;

    requestAnimationFrame(animation);
    callback(time, time - pt);
    pt = time;
  }

  animation(0);
}

export const stc = (radius: number, inclination: number, azimuth: number, origin = [0, 0, 0]) : [number, number, number] => {
  return [
      origin[0] + radius * Math.sin(inclination) * Math.cos(azimuth),
      origin[1] + radius * Math.cos(inclination),
      origin[2] + radius * Math.sin(inclination) * Math.sin(azimuth)
  ]
}