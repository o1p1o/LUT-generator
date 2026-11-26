import type { ImageBuffer, LutModel } from "./core";

export interface ParsedCube {
  model: LutModel;
  domainMin: [number, number, number];
  domainMax: [number, number, number];
}

export function parseCube(text: string): ParsedCube {
  let size = 0;
  let domainMin: [number, number, number] = [0, 0, 0];
  let domainMax: [number, number, number] = [1, 1, 1];
  const rows: number[] = [];
  let saw1d = false;

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.replace(/#.*/, "").trim();
    if (!line) continue;
    const parts = line.split(/\s+/);
    const keyword = parts[0].toUpperCase();
    if (keyword === "TITLE") continue;
    if (keyword === "LUT_1D_SIZE") { saw1d = true; continue; }
    if (keyword === "LUT_3D_SIZE") {
      size = Number(parts[1]);
      if (!Number.isInteger(size) || size < 2 || size > 256) throw new Error("Unsupported LUT_3D_SIZE. Expected an integer from 2 to 256.");
      continue;
    }
    if (keyword === "DOMAIN_MIN" || keyword === "DOMAIN_MAX") {
      if (parts.length !== 4) throw new Error(`Invalid ${keyword} declaration.`);
      const values = parts.slice(1).map(Number);
      if (!values.every(Number.isFinite)) throw new Error(`Invalid ${keyword} values.`);
      if (keyword === "DOMAIN_MIN") domainMin = values as [number, number, number];
      else domainMax = values as [number, number, number];
      continue;
    }
    if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(parts[0])) {
      if (parts.length !== 3) throw new Error("Invalid .cube RGB row: each row must contain exactly three values.");
      const values = parts.map(Number);
      if (!values.every(Number.isFinite)) throw new Error("The .cube file contains a non-finite value.");
      rows.push(...values);
    }
  }

  if (saw1d) throw new Error("This file contains LUT_1D_SIZE. Only 3D .cube LUTs are supported at the moment.");
  if (!size) throw new Error("The .cube file is missing LUT_3D_SIZE.");
  if (rows.length !== size * size * size * 3) throw new Error(`Expected ${size ** 3} RGB entries, found ${rows.length / 3}.`);
  for (let c = 0; c < 3; c++) if (!(domainMax[c] > domainMin[c])) throw new Error("Each DOMAIN_MAX value must be greater than DOMAIN_MIN.");

  return { model: { size, values: new Float32Array(rows) }, domainMin, domainMax };
}

export function parseHaldPng(image: ImageBuffer): ParsedCube {
  if (image.width !== 512 || image.height !== 512) throw new Error("Hald PNG must be exactly 512 × 512 pixels (Level 8).");
  const size = 64;
  const values = new Float32Array(size * size * size * 3);
  // Hald Level 8 is a row-major 64³ cube flattened into a 512 × 512 image.
  // The linear pixel order matches core.ts: R changes fastest, then G, then B.
  for (let pixel = 0; pixel < size ** 3; pixel++) {
    const src = pixel * 4;
    const dst = pixel * 3;
    values[dst] = image.data[src] / 255;
    values[dst + 1] = image.data[src + 1] / 255;
    values[dst + 2] = image.data[src + 2] / 255;
  }
  return { model: { size, values }, domainMin: [0, 0, 0], domainMax: [1, 1, 1] };
}

export function applyParsedLut(image: ImageBuffer, lut: ParsedCube): Uint8ClampedArray {
  const { model, domainMin, domainMax } = lut;
  const output = new Uint8ClampedArray(image.data.length);
  const size = model.size;
  const sample = new Float32Array(3);
  const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
  const index = (r: number, g: number, b: number) => ((b * size + g) * size + r) * 3;

  for (let i = 0; i < image.data.length; i += 4) {
    const coords = [0, 1, 2].map(c => clamp01((image.data[i + c] / 255 - domainMin[c]) / (domainMax[c] - domainMin[c])));
    const fr = coords[0] * (size - 1), fg = coords[1] * (size - 1), fb = coords[2] * (size - 1);
    const r0 = Math.min(size - 2, Math.floor(fr)), g0 = Math.min(size - 2, Math.floor(fg)), b0 = Math.min(size - 2, Math.floor(fb));
    const tr = fr - r0, tg = fg - g0, tb = fb - b0;
    sample.fill(0);
    for (let dz = 0; dz <= 1; dz++) for (let dy = 0; dy <= 1; dy++) for (let dx = 0; dx <= 1; dx++) {
      const w = (dx ? tr : 1 - tr) * (dy ? tg : 1 - tg) * (dz ? tb : 1 - tb);
      const p = index(r0 + dx, g0 + dy, b0 + dz);
      sample[0] += model.values[p] * w;
      sample[1] += model.values[p + 1] * w;
      sample[2] += model.values[p + 2] * w;
    }
    output[i] = Math.round(clamp01(sample[0]) * 255);
    output[i + 1] = Math.round(clamp01(sample[1]) * 255);
    output[i + 2] = Math.round(clamp01(sample[2]) * 255);
    output[i + 3] = image.data[i + 3];
  }
  return output;
}
