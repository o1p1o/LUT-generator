import "./style.css";
import type { ImageBuffer } from "./core";

type SlotName = "original" | "edited";

interface Slot {
  file: File | null;
  objectUrl: string | null;
  input: HTMLInputElement;
  preview: HTMLImageElement;
  meta: HTMLElement;
  drop: HTMLElement;
}

const $ = <T extends Element>(selector: string): T => {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing element: ${selector}`);
  return element;
};

const slots: Record<SlotName, Slot> = {
  original: {
    file: null,
    objectUrl: null,
    input: $("#originalInput"),
    preview: $("#originalPreview"),
    meta: $("#originalMeta"),
    drop: $("#originalDrop")
  },
  edited: {
    file: null,
    objectUrl: null,
    input: $("#editedInput"),
    preview: $("#editedPreview"),
    meta: $("#editedMeta"),
    drop: $("#editedDrop")
  }
};

const generateButton = $<HTMLButtonElement>("#generateButton");
const resetButton = $<HTMLButtonElement>("#resetButton");
const downloadButton = $<HTMLButtonElement>("#downloadButton");
const progressPanel = $<HTMLElement>("#progressPanel");
const progressBar = $<HTMLElement>("#progressBar");
const progressStage = $<HTMLElement>("#progressStage");
const progressPercent = $<HTMLElement>("#progressPercent");
const message = $<HTMLElement>("#message");
const resultSection = $<HTMLElement>("#resultSection");
const lutPreview = $<HTMLImageElement>("#lutPreview");
const comparisonOriginal = $<HTMLImageElement>("#comparisonOriginal");
const comparisonResult = $<HTMLImageElement>("#comparisonResult");
const comparisonToggle = $<HTMLInputElement>("#comparisonToggle");

let worker: Worker | null = null;
let resultBlob: Blob | null = null;
let resultUrl: string | null = null;
let comparisonUrl: string | null = null;
let running = false;

function setMessage(
  text: string,
  kind: "warning" | "error" | "success"
): void {
  message.hidden = !text;
  message.textContent = text;
  message.className = `message message-${kind}`;
}

function setProgress(percent: number, stage: string): void {
  progressPanel.hidden = false;
  progressBar.style.width = `${percent}%`;
  progressStage.textContent = stage;
  progressPercent.textContent = `${Math.round(percent)}%`;

  const progressElement = progressPanel.querySelector<HTMLElement>(
    '[role="progressbar"]'
  );
  progressElement?.setAttribute("aria-valuenow", String(percent));
}

function updateGenerateState(): void {
  generateButton.disabled =
    running || !slots.original.file || !slots.edited.file;
}

function updateUploadLabel(name: SlotName, uploaded: boolean): void {
  const drop = slots[name].drop;
  const title = drop.querySelector<HTMLElement>(".drop-copy strong");
  const hint = drop.querySelector<HTMLElement>(".drop-copy small");

  if (!title || !hint) return;

  title.textContent = uploaded
    ? "File uploaded"
    : name === "original"
      ? "Drop original image"
      : "Drop edited image";

  hint.textContent = uploaded
    ? ""
    : "or click to select PNG, JPEG or WebP";
}

function clearResult(): void {
  resultSection.hidden = true;
  resultBlob = null;

  if (resultUrl) URL.revokeObjectURL(resultUrl);
  if (comparisonUrl) URL.revokeObjectURL(comparisonUrl);

  resultUrl = null;
  comparisonUrl = null;
  lutPreview.removeAttribute("src");
  comparisonResult.removeAttribute("src");
  comparisonToggle.checked = false;
}

function useFile(name: SlotName, file: File): void {
  if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
    setMessage("Please select a PNG, JPEG or WebP image.", "error");
    return;
  }

  const slot = slots[name];

  if (slot.objectUrl) URL.revokeObjectURL(slot.objectUrl);
  slot.objectUrl = URL.createObjectURL(file);
  slot.file = file;
  slot.preview.src = slot.objectUrl;
  slot.preview.hidden = false;
  slot.drop.classList.add("has-image");
  slot.meta.textContent = `${file.name} · ${formatBytes(file.size)}`;

updateUploadLabel(name, true);

  if (name === "original") {
    comparisonOriginal.src = slot.objectUrl;
  }

  clearResult();
  setMessage("", "success");
  updateGenerateState();
}

function bindSlot(name: SlotName): void {
  const slot = slots[name];

  slot.input.addEventListener("change", () => {
    const file = slot.input.files?.[0];
    if (file) useFile(name, file);
  });

  for (const eventName of ["dragenter", "dragover"]) {
    slot.drop.addEventListener(eventName, (event) => {
      event.preventDefault();
      slot.drop.classList.add("is-dragging");
    });
  }

  for (const eventName of ["dragleave", "drop"]) {
    slot.drop.addEventListener(eventName, (event) => {
      event.preventDefault();
      slot.drop.classList.remove("is-dragging");
    });
  }

  slot.drop.addEventListener("drop", (event) => {
    const dragEvent = event as DragEvent;
    const file = dragEvent.dataTransfer?.files[0];
    if (file) useFile(name, file);
  });
}

async function decodeImage(
  source: Blob,
  maxDimension?: number
): Promise<ImageBuffer> {
  const bitmap = await createImageBitmap(source, {
    imageOrientation: "from-image",
    premultiplyAlpha: "none",
    colorSpaceConversion: "default"
  });

  let width = bitmap.width;
  let height = bitmap.height;

  if (maxDimension && Math.max(width, height) > maxDimension) {
    const ratio = maxDimension / Math.max(width, height);
    width = Math.max(1, Math.round(width * ratio));
    height = Math.max(1, Math.round(height * ratio));
  }

  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d", {
    alpha: false,
    colorSpace: "srgb"
  });

  if (!context) throw new Error("Unable to create an image canvas.");

  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();

  const imageData = context.getImageData(0, 0, width, height);

  return {
    width,
    height,
    data: imageData.data
  };
}

async function loadNeutralLut(): Promise<ImageBuffer> {
  const response = await fetch(`${import.meta.env.BASE_URL}neutral_LUT.png`, {
    cache: "force-cache"
  });

  if (!response.ok) {
    throw new Error(
      "neutral_LUT.png was not found in the public directory."
    );
  }

  const blob = await response.blob();
  const image = await decodeImage(blob);

  if (image.width !== 512 || image.height !== 512) {
    throw new Error("neutral_LUT.png must have exact dimensions 512×512.");
  }

  return image;
}

function imageDataUrl(
  pixels: Uint8ClampedArray,
  width: number,
  height: number
): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("2d");
  if (!context) throw new Error("Unable to render preview.");

const imageData = new ImageData(width, height);
imageData.data.set(pixels);
context.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function uint32(value: number): Uint8Array {
  return new Uint8Array([
    (value >>> 24) & 255,
    (value >>> 16) & 255,
    (value >>> 8) & 255,
    value & 255
  ]);
}

const crcTable = (() => {
  const table = new Uint32Array(256);

  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c >>> 0;
  }

  return table;
})();

function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;

  for (const byte of data) {
    crc = crcTable[(crc ^ byte) & 255] ^ (crc >>> 8);
  }

  return (crc ^ 0xffffffff) >>> 0;
}

function concat(parts: Uint8Array[]): Uint8Array {
  const length = parts.reduce((sum, part) => sum + part.length, 0);
  const output = new Uint8Array(length);
  let offset = 0;

  for (const part of parts) {
    output.set(part, offset);
    offset += part.length;
  }

  return output;
}

function pngChunk(type: string, data: Uint8Array): Uint8Array {
  const typeBytes = new TextEncoder().encode(type);
  return concat([
    uint32(data.length),
    typeBytes,
    data,
    uint32(crc32(concat([typeBytes, data])))
  ]);
}

async function encodeRgbPng(
  rgba: Uint8ClampedArray,
  width: number,
  height: number
): Promise<Blob> {
  const rowLength = width * 3 + 1;
  const raw = new Uint8Array(rowLength * height);

  for (let y = 0; y < height; y++) {
    const rowOffset = y * rowLength;
    raw[rowOffset] = 0;

    for (let x = 0; x < width; x++) {
      const source = (y * width + x) * 4;
      const target = rowOffset + 1 + x * 3;

      raw[target] = rgba[source];
      raw[target + 1] = rgba[source + 1];
      raw[target + 2] = rgba[source + 2];
    }
  }

  if (!("CompressionStream" in window)) {
    throw new Error(
      "This browser does not support RGB PNG compression. Use a current version of Chrome, Edge, Firefox or Safari."
    );
  }

const compression = new CompressionStream("deflate");
const writer = compression.writable.getWriter();

const compressedPromise = new Response(
  compression.readable
).arrayBuffer();

await writer.write(raw);
await writer.close();

const compressed = new Uint8Array(
  await compressedPromise
);
  
  const ihdr = new Uint8Array(13);
  ihdr.set(uint32(width), 0);
  ihdr.set(uint32(height), 4);
  ihdr[8] = 8;  // bit depth
  ihdr[9] = 2;  // PNG truecolour RGB, no alpha
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;

  const png = concat([
    new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", compressed),
    pngChunk("IEND", new Uint8Array())
  ]);

const pngBuffer = new ArrayBuffer(png.byteLength);
new Uint8Array(pngBuffer).set(png);

return new Blob([pngBuffer], { type: "image/png" });
}

async function generate(): Promise<void> {
  if (running || !slots.original.file || !slots.edited.file) return;

  running = true;
  updateGenerateState();
  clearResult();
  setMessage("", "success");
  setProgress(2, "Decoding images and converting to browser sRGB…");

  try {
    const [original, edited, neutral] = await Promise.all([
      decodeImage(slots.original.file, 1000),
      decodeImage(slots.edited.file, 1000),
      loadNeutralLut()
    ]);

    worker?.terminate();
    worker = new Worker(new URL("./worker.ts", import.meta.url), {
      type: "module"
    });

    worker.onmessage = async (event) => {
      const payload = event.data;

      if (payload.type === "progress") {
        setProgress(payload.percent, payload.stage);
        return;
      }

      if (payload.type === "error") {
        running = false;
        updateGenerateState();
        setMessage(payload.message, "error");
        worker?.terminate();
        worker = null;
        return;
      }

      if (payload.type !== "result") return;

      try {
        setProgress(98, "Encoding truecolour RGB PNG…");

        const lutPixels = new Uint8ClampedArray(payload.lutPixels);
        const previewPixels = new Uint8ClampedArray(payload.previewPixels);

        resultBlob = await encodeRgbPng(lutPixels, 512, 512);
        resultUrl = URL.createObjectURL(resultBlob);
        lutPreview.src = resultUrl;

        comparisonUrl = imageDataUrl(
          previewPixels,
          payload.previewWidth,
          payload.previewHeight
        );
        comparisonResult.src = comparisonUrl;

        $("#maeMetric").textContent =
          `${(payload.metrics.mae * 255).toFixed(2)} / 255`;
        $("#deltaMetric").textContent =
          payload.metrics.deltaE.toFixed(2);
        $("#registrationMetric").textContent =
          `${Math.round(payload.registration.confidence * 100)}%`;
        $("#samplesMetric").textContent =
          Number(payload.sampleCount).toLocaleString();

        resultSection.hidden = false;
        setProgress(100, "Hald CLUT generated");

        if (payload.warning) {
          setMessage(payload.warning, "warning");
        } else {
          setMessage(
            "LUT generated successfully. Inspect the comparison preview before production use.",
            "success"
          );
        }

        resultSection.scrollIntoView({
          behavior: "smooth",
          block: "start"
        });
      } catch (error) {
        setMessage(
          error instanceof Error ? error.message : "PNG encoding failed.",
          "error"
        );
      } finally {
        running = false;
        updateGenerateState();
        worker?.terminate();
        worker = null;
      }
    };

    worker.onerror = (event) => {
      running = false;
      updateGenerateState();
      setMessage(event.message || "Web Worker failed.", "error");
      worker?.terminate();
      worker = null;
    };

    worker.postMessage(
      { original, edited, neutral },
      [
        original.data.buffer,
        edited.data.buffer,
        neutral.data.buffer
      ]
    );
  } catch (error) {
    running = false;
    updateGenerateState();
    setMessage(
      error instanceof Error ? error.message : "Unable to process images.",
      "error"
    );
  }
}

function reset(): void {
  worker?.terminate();
  worker = null;
  running = false;

  for (const slot of Object.values(slots)) {
    if (slot.objectUrl) URL.revokeObjectURL(slot.objectUrl);
    slot.objectUrl = null;
    slot.file = null;
    slot.input.value = "";
    slot.preview.hidden = true;
    slot.preview.removeAttribute("src");
    slot.meta.textContent = "No image";
    slot.drop.classList.remove("has-image", "is-dragging");
  }

  updateUploadLabel("original", false);
  updateUploadLabel("edited", false);

  comparisonOriginal.removeAttribute("src");
  clearResult();
  progressPanel.hidden = true;
  setMessage("", "success");
  updateGenerateState();
}

bindSlot("original");
bindSlot("edited");

generateButton.addEventListener("click", generate);
resetButton.addEventListener("click", reset);

downloadButton.addEventListener("click", () => {
  if (!resultBlob || !resultUrl) return;

  const link = document.createElement("a");
  link.href = resultUrl;
  link.download = `extracted-hald-clut-${new Date()
    .toISOString()
    .replace(/[:.]/g, "-")}.png`;
  document.body.appendChild(link);
  link.click();
  link.remove();
});

comparisonToggle.addEventListener("change", () => {
  comparisonOriginal.hidden = comparisonToggle.checked;
  comparisonResult.hidden = !comparisonToggle.checked;
});

window.addEventListener("beforeunload", () => {
  worker?.terminate();

  for (const slot of Object.values(slots)) {
    if (slot.objectUrl) URL.revokeObjectURL(slot.objectUrl);
  }

  if (resultUrl) URL.revokeObjectURL(resultUrl);
  if (comparisonUrl?.startsWith("blob:")) {
    URL.revokeObjectURL(comparisonUrl);
  }
});

updateGenerateState();
