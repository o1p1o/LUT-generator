import { describe, expect, it } from "vitest";
import {
  applyModelToImage,
  calculateMetrics,
  collectSamples,
  evaluateLut,
  fitLut,
  type ImageBuffer,
  type LutModel
} from "../src/core";

function syntheticImage(
  width: number,
  height: number,
  transform?: (r: number, g: number, b: number) => number[]
): ImageBuffer {
  const data = new Uint8ClampedArray(width * height * 4);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = (y * width + x) * 4;
      const r = x / Math.max(1, width - 1);
      const g = y / Math.max(1, height - 1);
      const b = ((x * 7 + y * 13) % width) / Math.max(1, width - 1);
      const output = transform ? transform(r, g, b) : [r, g, b];

      data[index] = Math.round(output[0] * 255);
      data[index + 1] = Math.round(output[1] * 255);
      data[index + 2] = Math.round(output[2] * 255);
      data[index + 3] = 255;
    }
  }

  return { width, height, data };
}

describe("LUT fitting", () => {
  it("reconstructs identity from identical images", () => {
    const image = syntheticImage(128, 128);
    const samples = collectSamples(
      image,
      image,
      { scale: 1, dx: 0, dy: 0, cost: 0, confidence: 1 },
      20_000
    );

    const model = fitLut(samples, 17);
    const metrics = calculateMetrics(samples, model);

    expect(metrics.mae).toBeLessThan(0.015);
    expect(metrics.deltaE).toBeLessThan(2);
  });

  it("recovers a smooth cross-channel transformation", () => {
    const original = syntheticImage(160, 120);
    const edited = syntheticImage(160, 120, (r, g, b) => [
      Math.min(1, r * 0.9 + g * 0.06 + 0.025),
      Math.min(1, g * 1.04 + b * 0.025),
      Math.min(1, b * 0.88 + r * 0.08)
    ]);

    const samples = collectSamples(
      original,
      edited,
      { scale: 1, dx: 0, dy: 0, cost: 0, confidence: 1 },
      30_000
    );

    const model = fitLut(samples, 17);
    const metrics = calculateMetrics(samples, model);

    expect(metrics.mae).toBeLessThan(0.04);
  });

  it("evaluates an exact identity model", () => {
    const size = 4;
    const values = new Float32Array(size * size * size * 3);

    for (let b = 0; b < size; b++) {
      for (let g = 0; g < size; g++) {
        for (let r = 0; r < size; r++) {
          const index = ((b * size + g) * size + r) * 3;
          values[index] = r / (size - 1);
          values[index + 1] = g / (size - 1);
          values[index + 2] = b / (size - 1);
        }
      }
    }

    const model: LutModel = { size, values };
    const result = evaluateLut(model, 0.21, 0.53, 0.87);

    expect(result[0]).toBeCloseTo(0.21, 5);
    expect(result[1]).toBeCloseTo(0.53, 5);
    expect(result[2]).toBeCloseTo(0.87, 5);
  });

  it("preserves image dimensions", () => {
    const image = syntheticImage(64, 32);
    const size = 2;
    const model: LutModel = {
      size,
      values: new Float32Array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        1, 1, 0,
        0, 0, 1,
        1, 0, 1,
        0, 1, 1,
        1, 1, 1
      ])
    };

    expect(applyModelToImage(image, model).length).toBe(64 * 32 * 4);
  });
});
