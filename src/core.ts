export interface ImageBuffer {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export interface Registration {
  scale: number;
  dx: number;
  dy: number;
  cost: number;
  confidence: number;
}

export interface SampleSet {
  values: Float32Array;
  count: number;
}

export interface LutModel {
  size: number;
  values: Float32Array;
}

const clamp = (x: number, min = 0, max = 1) =>
  Math.min(max, Math.max(min, x));

export function srgbToLinear(x: number): number {
  return x <= 0.04045
    ? x / 12.92
    : Math.pow((x + 0.055) / 1.055, 2.4);
}

export function linearToSrgb(x: number): number {
  x = clamp(x);
  return x <= 0.0031308
    ? x * 12.92
    : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
}

export function sampleBilinear(
  image: ImageBuffer,
  u: number,
  v: number,
  output = new Float32Array(3)
): Float32Array {
  const x = clamp(u) * (image.width - 1);
  const y = clamp(v) * (image.height - 1);
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(image.width - 1, x0 + 1);
  const y1 = Math.min(image.height - 1, y0 + 1);
  const tx = x - x0;
  const ty = y - y0;

  const i00 = (y0 * image.width + x0) * 4;
  const i10 = (y0 * image.width + x1) * 4;
  const i01 = (y1 * image.width + x0) * 4;
  const i11 = (y1 * image.width + x1) * 4;

  for (let c = 0; c < 3; c++) {
    const a =
      image.data[i00 + c] * (1 - tx) + image.data[i10 + c] * tx;
    const b =
      image.data[i01 + c] * (1 - tx) + image.data[i11 + c] * tx;
    output[c] = (a * (1 - ty) + b * ty) / 255;
  }

  return output;
}

function buildEdgeMap(image: ImageBuffer, size = 128): Float32Array {
  const luma = new Float32Array(size * size);
  const rgb = new Float32Array(3);

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      sampleBilinear(
        image,
        (x + 0.5) / size,
        (y + 0.5) / size,
        rgb
      );

      const r = srgbToLinear(rgb[0]);
      const g = srgbToLinear(rgb[1]);
      const b = srgbToLinear(rgb[2]);
      luma[y * size + x] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
  }

  const edges = new Float32Array(size * size);
  let mean = 0;

  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      const i = y * size + x;
      const gx =
        -luma[i - size - 1] -
        2 * luma[i - 1] -
        luma[i + size - 1] +
        luma[i - size + 1] +
        2 * luma[i + 1] +
        luma[i + size + 1];

      const gy =
        -luma[i - size - 1] -
        2 * luma[i - size] -
        luma[i - size + 1] +
        luma[i + size - 1] +
        2 * luma[i + size] +
        luma[i + size + 1];

      const magnitude = Math.sqrt(gx * gx + gy * gy);
      edges[i] = magnitude;
      mean += magnitude;
    }
  }

  mean /= size * size;
  const normalizer = Math.max(mean, 1e-5);

  for (let i = 0; i < edges.length; i++) {
    edges[i] = Math.min(4, edges[i] / normalizer);
  }

  return edges;
}

function registrationCost(
  original: Float32Array,
  edited: Float32Array,
  size: number,
  scale: number,
  dx: number,
  dy: number
): number {
  let total = 0;
  let weight = 0;

  for (let y = 4; y < size - 4; y += 2) {
    const v = (y + 0.5) / size;
    const ev = 0.5 + (v - 0.5) * scale + dy;
    if (ev <= 0.02 || ev >= 0.98) continue;

    const ey = Math.round(ev * (size - 1));

    for (let x = 4; x < size - 4; x += 2) {
      const u = (x + 0.5) / size;
      const eu = 0.5 + (u - 0.5) * scale + dx;
      if (eu <= 0.02 || eu >= 0.98) continue;

      const ex = Math.round(eu * (size - 1));
      const a = original[y * size + x];
      const b = edited[ey * size + ex];

      const robust = Math.min(2, Math.abs(a - b));
      const localWeight = 0.2 + Math.min(2, a);
      total += robust * localWeight;
      weight += localWeight;
    }
  }

  return weight > 0 ? total / weight : Number.POSITIVE_INFINITY;
}

export function registerImages(
  originalImage: ImageBuffer,
  editedImage: ImageBuffer
): Registration {
  const size = 128;
  const original = buildEdgeMap(originalImage, size);
  const edited = buildEdgeMap(editedImage, size);

  let best: Registration = {
    scale: 1,
    dx: 0,
    dy: 0,
    cost: registrationCost(original, edited, size, 1, 0, 0),
    confidence: 0
  };

  const initialCost = best.cost;

  for (let scale = 0.8; scale <= 1.2001; scale += 0.05) {
    for (let dy = -0.12; dy <= 0.1201; dy += 0.02) {
      for (let dx = -0.12; dx <= 0.1201; dx += 0.02) {
        const cost = registrationCost(
          original,
          edited,
          size,
          scale,
          dx,
          dy
        );

        if (cost < best.cost) {
          best = { scale, dx, dy, cost, confidence: 0 };
        }
      }
    }
  }

  for (let pass = 0; pass < 2; pass++) {
    const scaleStep = pass === 0 ? 0.015 : 0.005;
    const shiftStep = pass === 0 ? 0.008 : 0.0025;
    const center = { ...best };

    for (let si = -2; si <= 2; si++) {
      for (let yi = -2; yi <= 2; yi++) {
        for (let xi = -2; xi <= 2; xi++) {
          const scale = center.scale + si * scaleStep;
          const dx = center.dx + xi * shiftStep;
          const dy = center.dy + yi * shiftStep;
          const cost = registrationCost(
            original,
            edited,
            size,
            scale,
            dx,
            dy
          );

          if (cost < best.cost) {
            best = { scale, dx, dy, cost, confidence: 0 };
          }
        }
      }
    }
  }

  const improvement =
    initialCost > 1e-6 ? (initialCost - best.cost) / initialCost : 0;

  const absoluteQuality = clamp(1 - best.cost / 1.15);
  best.confidence = clamp(
    absoluteQuality * 0.8 + Math.max(0, improvement) * 0.2
  );

  // Do not introduce a tiny accidental warp when the images already align.
  if (
    Math.abs(best.scale - 1) < 0.012 &&
    Math.abs(best.dx) < 0.006 &&
    Math.abs(best.dy) < 0.006
  ) {
    best.scale = 1;
    best.dx = 0;
    best.dy = 0;
    best.cost = initialCost;
  }

  return best;
}

function spatialGradient(image: ImageBuffer, u: number, v: number): number {
  const du = 1 / Math.max(64, image.width);
  const dv = 1 / Math.max(64, image.height);
  const a = sampleBilinear(image, u - du, v);
  const b = sampleBilinear(image, u + du, v);
  const c = sampleBilinear(image, u, v - dv);
  const d = sampleBilinear(image, u, v + dv);

  return (
    Math.abs(a[0] - b[0]) +
    Math.abs(a[1] - b[1]) +
    Math.abs(a[2] - b[2]) +
    Math.abs(c[0] - d[0]) +
    Math.abs(c[1] - d[1]) +
    Math.abs(c[2] - d[2])
  );
}

export function collectSamples(
  original: ImageBuffer,
  edited: ImageBuffer,
  registration: Registration,
  maxSamples = 140_000
): SampleSet {
  const area = original.width * original.height;
  const stride = Math.max(1, Math.ceil(Math.sqrt(area / maxSamples)));
  const raw: number[] = [];
  const histogram = new Uint32Array(16 * 16 * 16);

  for (let y = Math.floor(stride / 2); y < original.height; y += stride) {
    const v = (y + 0.5) / original.height;
    const ev = 0.5 + (v - 0.5) * registration.scale + registration.dy;
    if (ev < 0 || ev > 1) continue;

    for (let x = Math.floor(stride / 2); x < original.width; x += stride) {
      const u = (x + 0.5) / original.width;
      const eu = 0.5 + (u - 0.5) * registration.scale + registration.dx;
      if (eu < 0 || eu > 1) continue;

      // Strong edges amplify registration and sharpening differences.
      const gradient = spatialGradient(original, u, v);
      if (gradient > 0.42) continue;

      const input = sampleBilinear(original, u, v);
      const output = sampleBilinear(edited, eu, ev);

      const clipped =
        input[0] < 0.002 ||
        input[1] < 0.002 ||
        input[2] < 0.002 ||
        input[0] > 0.998 ||
        input[1] > 0.998 ||
        input[2] > 0.998;

      const rBin = Math.min(15, Math.floor(input[0] * 16));
      const gBin = Math.min(15, Math.floor(input[1] * 16));
      const bBin = Math.min(15, Math.floor(input[2] * 16));
      histogram[(bBin * 16 + gBin) * 16 + rBin]++;

      raw.push(
        input[0],
        input[1],
        input[2],
        output[0],
        output[1],
        output[2],
        clipped ? 0.35 : 1,
        (bBin * 16 + gBin) * 16 + rBin
      );
    }
  }

  const count = raw.length / 8;
  const values = new Float32Array(count * 7);

  for (let i = 0; i < count; i++) {
    const source = i * 8;
    const target = i * 7;
    const density = histogram[raw[source + 7]];

    for (let c = 0; c < 6; c++) {
      values[target + c] = raw[source + c];
    }

    values[target + 6] =
      raw[source + 6] / Math.sqrt(Math.max(1, density));
  }

  return { values, count };
}

function gridIndex(size: number, r: number, g: number, b: number): number {
  return ((b * size + g) * size + r) * 3;
}

function accumulateSamples(
  samples: SampleSet,
  size: number,
  robustModel?: LutModel,
  robustThreshold = 0.04
): { target: Float32Array; confidence: Float32Array } {
  const nodes = size * size * size;
  const target = new Float32Array(nodes * 3);
  const confidence = new Float32Array(nodes);
  const p = new Float32Array(3);

  for (let i = 0; i < samples.count; i++) {
    const offset = i * 7;
    const r = samples.values[offset];
    const g = samples.values[offset + 1];
    const b = samples.values[offset + 2];
    const outR = samples.values[offset + 3];
    const outG = samples.values[offset + 4];
    const outB = samples.values[offset + 5];

    let sampleWeight = samples.values[offset + 6];

    if (robustModel) {
      evaluateLut(robustModel, r, g, b, p);
      const residual = Math.sqrt(
        (p[0] - outR) ** 2 +
        (p[1] - outG) ** 2 +
        (p[2] - outB) ** 2
      );

      sampleWeight *= Math.min(
        1,
        (2.5 * robustThreshold) / Math.max(residual, 1e-5)
      );

      if (residual > Math.max(0.18, robustThreshold * 8)) {
        continue;
      }
    }

    const fr = r * (size - 1);
    const fg = g * (size - 1);
    const fb = b * (size - 1);
    const r0 = Math.min(size - 2, Math.floor(fr));
    const g0 = Math.min(size - 2, Math.floor(fg));
    const b0 = Math.min(size - 2, Math.floor(fb));
    const tr = fr - r0;
    const tg = fg - g0;
    const tb = fb - b0;

    for (let dz = 0; dz <= 1; dz++) {
      for (let dy = 0; dy <= 1; dy++) {
        for (let dx = 0; dx <= 1; dx++) {
          const weight =
            sampleWeight *
            (dx ? tr : 1 - tr) *
            (dy ? tg : 1 - tg) *
            (dz ? tb : 1 - tb);

          const node = ((b0 + dz) * size + (g0 + dy)) * size + r0 + dx;
          const index = node * 3;

          confidence[node] += weight;
          target[index] += outR * weight;
          target[index + 1] += outG * weight;
          target[index + 2] += outB * weight;
        }
      }
    }
  }

  for (let node = 0; node < nodes; node++) {
    const weight = confidence[node];
    if (weight <= 0) continue;

    const index = node * 3;
    target[index] /= weight;
    target[index + 1] /= weight;
    target[index + 2] /= weight;

    confidence[node] = Math.min(4, weight * 250);
  }

  return { target, confidence };
}

function solveRegularizedGrid(
  target: Float32Array,
  confidence: Float32Array,
  size: number,
  iterations = 70
): LutModel {
  const nodes = size * size * size;
  let current = new Float32Array(nodes * 3);
  let next = new Float32Array(nodes * 3);

  for (let b = 0; b < size; b++) {
    for (let g = 0; g < size; g++) {
      for (let r = 0; r < size; r++) {
        const index = gridIndex(size, r, g, b);
        const node = index / 3;

        if (confidence[node] > 0) {
          current[index] = target[index];
          current[index + 1] = target[index + 1];
          current[index + 2] = target[index + 2];
        } else {
          current[index] = r / (size - 1);
          current[index + 1] = g / (size - 1);
          current[index + 2] = b / (size - 1);
        }
      }
    }
  }

  const lambda = 0.72;
  const identityPrior = 0.012;

  for (let iteration = 0; iteration < iterations; iteration++) {
    for (let b = 0; b < size; b++) {
      for (let g = 0; g < size; g++) {
        for (let r = 0; r < size; r++) {
          const index = gridIndex(size, r, g, b);
          const node = index / 3;
          const neighbours: number[] = [];

          if (r > 0) neighbours.push(gridIndex(size, r - 1, g, b));
          if (r + 1 < size) neighbours.push(gridIndex(size, r + 1, g, b));
          if (g > 0) neighbours.push(gridIndex(size, r, g - 1, b));
          if (g + 1 < size) neighbours.push(gridIndex(size, r, g + 1, b));
          if (b > 0) neighbours.push(gridIndex(size, r, g, b - 1));
          if (b + 1 < size) neighbours.push(gridIndex(size, r, g, b + 1));

          const dataWeight = confidence[node];
          const denominator = dataWeight + lambda + identityPrior;

          for (let channel = 0; channel < 3; channel++) {
            let neighbourMean = 0;
            for (const neighbour of neighbours) {
              neighbourMean += current[neighbour + channel];
            }
            neighbourMean /= neighbours.length;

            const identity =
              channel === 0
                ? r / (size - 1)
                : channel === 1
                  ? g / (size - 1)
                  : b / (size - 1);

            next[index + channel] = clamp(
              (
                dataWeight * target[index + channel] +
                lambda * neighbourMean +
                identityPrior * identity
              ) / denominator
            );
          }
        }
      }
    }

    [current, next] = [next, current];
  }

  return { size, values: current };
}

function estimateResidualThreshold(
  samples: SampleSet,
  model: LutModel
): number {
  const residuals: number[] = [];
  const output = new Float32Array(3);
  const step = Math.max(1, Math.floor(samples.count / 20_000));

  for (let i = 0; i < samples.count; i += step) {
    const offset = i * 7;
    evaluateLut(
      model,
      samples.values[offset],
      samples.values[offset + 1],
      samples.values[offset + 2],
      output
    );

    residuals.push(
      Math.sqrt(
        (output[0] - samples.values[offset + 3]) ** 2 +
        (output[1] - samples.values[offset + 4]) ** 2 +
        (output[2] - samples.values[offset + 5]) ** 2
      )
    );
  }

  residuals.sort((a, b) => a - b);
  return Math.max(0.008, residuals[Math.floor(residuals.length / 2)] || 0.04);
}

export function fitLut(samples: SampleSet, size = 17): LutModel {
  const firstAccumulation = accumulateSamples(samples, size);
  const initialModel = solveRegularizedGrid(
    firstAccumulation.target,
    firstAccumulation.confidence,
    size,
    60
  );

  const threshold = estimateResidualThreshold(samples, initialModel);
  const robustAccumulation = accumulateSamples(
    samples,
    size,
    initialModel,
    threshold
  );

  return solveRegularizedGrid(
    robustAccumulation.target,
    robustAccumulation.confidence,
    size,
    85
  );
}

export function evaluateLut(
  model: LutModel,
  r: number,
  g: number,
  b: number,
  output = new Float32Array(3)
): Float32Array {
  const size = model.size;
  const fr = clamp(r) * (size - 1);
  const fg = clamp(g) * (size - 1);
  const fb = clamp(b) * (size - 1);
  const r0 = Math.min(size - 2, Math.floor(fr));
  const g0 = Math.min(size - 2, Math.floor(fg));
  const b0 = Math.min(size - 2, Math.floor(fb));
  const tr = fr - r0;
  const tg = fg - g0;
  const tb = fb - b0;

  output.fill(0);

  for (let dz = 0; dz <= 1; dz++) {
    for (let dy = 0; dy <= 1; dy++) {
      for (let dx = 0; dx <= 1; dx++) {
        const weight =
          (dx ? tr : 1 - tr) *
          (dy ? tg : 1 - tg) *
          (dz ? tb : 1 - tb);

        const index = gridIndex(
          size,
          r0 + dx,
          g0 + dy,
          b0 + dz
        );

        output[0] += model.values[index] * weight;
        output[1] += model.values[index + 1] * weight;
        output[2] += model.values[index + 2] * weight;
      }
    }
  }

  return output;
}

export function applyModelToImage(
  image: ImageBuffer,
  model: LutModel
): Uint8ClampedArray {
  const result = new Uint8ClampedArray(image.data.length);
  const output = new Float32Array(3);

  for (let i = 0; i < image.data.length; i += 4) {
    evaluateLut(
      model,
      image.data[i] / 255,
      image.data[i + 1] / 255,
      image.data[i + 2] / 255,
      output
    );

    result[i] = Math.round(clamp(output[0]) * 255);
    result[i + 1] = Math.round(clamp(output[1]) * 255);
    result[i + 2] = Math.round(clamp(output[2]) * 255);
    result[i + 3] = 255;
  }

  return result;
}

function rgbToLab(r: number, g: number, b: number): [number, number, number] {
  r = srgbToLinear(r);
  g = srgbToLinear(g);
  b = srgbToLinear(b);

  let x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047;
  let y = 0.2126729 * r + 0.7151522 * g + 0.072175 * b;
  let z = (0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883;

  const f = (value: number) =>
    value > 216 / 24389
      ? Math.cbrt(value)
      : (24389 / 27 * value + 16) / 116;

  x = f(x);
  y = f(y);
  z = f(z);

  return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
}

export function calculateMetrics(
  samples: SampleSet,
  model: LutModel
): { mae: number; deltaE: number } {
  const output = new Float32Array(3);
  let absoluteError = 0;
  let deltaE = 0;
  let count = 0;
  const step = Math.max(1, Math.floor(samples.count / 50_000));

  for (let i = 0; i < samples.count; i += step) {
    const offset = i * 7;
    evaluateLut(
      model,
      samples.values[offset],
      samples.values[offset + 1],
      samples.values[offset + 2],
      output
    );

    const expected = [
      samples.values[offset + 3],
      samples.values[offset + 4],
      samples.values[offset + 5]
    ];

    absoluteError +=
      Math.abs(output[0] - expected[0]) +
      Math.abs(output[1] - expected[1]) +
      Math.abs(output[2] - expected[2]);

    const lab1 = rgbToLab(output[0], output[1], output[2]);
    const lab2 = rgbToLab(expected[0], expected[1], expected[2]);

    deltaE += Math.sqrt(
      (lab1[0] - lab2[0]) ** 2 +
      (lab1[1] - lab2[1]) ** 2 +
      (lab1[2] - lab2[2]) ** 2
    );

    count++;
  }

  return {
    mae: count ? absoluteError / (count * 3) : 0,
    deltaE: count ? deltaE / count : 0
  };
}
