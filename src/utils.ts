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

export const stc = (radius: number, inclination: number, azimuth: number) : [number, number, number] => {
  return [
      radius * Math.sin(inclination) * Math.cos(azimuth),
      radius * Math.cos(inclination),
      radius * Math.sin(inclination) * Math.sin(azimuth)
  ]
}