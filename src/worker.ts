/// <reference lib="webworker" />

import {
  applyModelToImage,
  calculateMetrics,
  collectSamples,
  fitLut,
  registerImages,
  type ImageBuffer
} from "./core";

interface WorkerInput {
  original: ImageBuffer;
  edited: ImageBuffer;
  neutral: ImageBuffer;
}

const scope = self as DedicatedWorkerGlobalScope;

function progress(percent: number, stage: string): void {
  scope.postMessage({ type: "progress", percent, stage });
}

scope.onmessage = (event: MessageEvent<WorkerInput>) => {
  try {
    const { original, edited, neutral } = event.data;

    if (neutral.width !== 512 || neutral.height !== 512) {
      throw new Error(
        `neutral_LUT.png must be 512×512, received ${neutral.width}×${neutral.height}.`
      );
    }

    progress(8, "Registering image geometry…");
    const registration = registerImages(original, edited);

    progress(25, "Collecting corresponding RGB samples…");
    const samples = collectSamples(
      original,
      edited,
      registration,
      150_000
    );

    if (samples.count < 1_500) {
      throw new Error(
        "Too few reliable corresponding pixels were found. Use images with the same scene and similar framing."
      );
    }

    progress(48, "Rejecting outliers and fitting the 3D color model…");
    const model = fitLut(samples, 17);

    progress(76, "Applying the model to the neutral Hald CLUT…");
    const lutPixels = applyModelToImage(neutral, model);

    progress(88, "Rendering comparison preview…");
    const previewPixels = applyModelToImage(original, model);

    progress(94, "Calculating reconstruction error…");
    const metrics = calculateMetrics(samples, model);

    let warning = "";

    if (registration.confidence < 0.32) {
      warning =
        "Low registration confidence. The images may differ in composition, perspective, local editing, or spatial effects. Inspect the preview before using the LUT.";
    } else if (metrics.deltaE > 10) {
      warning =
        "The reconstruction error is high. The edit may contain masks, exposure blending, retouching, clipping, local contrast, or other effects that cannot be represented by a global 3D LUT.";
    }

    progress(100, "Complete");

    scope.postMessage(
      {
        type: "result",
        lutPixels,
        previewPixels,
        previewWidth: original.width,
        previewHeight: original.height,
        registration,
        metrics,
        sampleCount: samples.count,
        warning
      },
      [lutPixels.buffer, previewPixels.buffer]
    );
  } catch (error) {
    scope.postMessage({
      type: "error",
      message:
        error instanceof Error ? error.message : "Unexpected processing error."
    });
  }
};
