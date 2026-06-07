---
base_model: nvidia/segformer-b0-finetuned-cityscapes-640-1280
library_name: transformers.js
pipeline_tag: image-segmentation
---

https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-640-1280 with ONNX weights to be compatible with Transformers.js.

## Usage (Transformers.js)

If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:
```bash
npm i @huggingface/transformers
```

**Example:** Image segmentation with `Xenova/segformer-b0-finetuned-cityscapes-640-1280`.

```js
import { pipeline } from '@huggingface/transformers';

// Create an image segmentation pipeline
const segmenter = await pipeline('image-segmentation', 'Xenova/segformer-b0-finetuned-cityscapes-640-1280');

// Segment an image
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cityscapes.png';
const output = await segmenter(url);
console.log(output);
// [
//   {
//     score: null,
//     label: 'road',
//     mask: RawImage { ... }
//   },
//   {
//     score: null,
//     label: 'sidewalk',
//     mask: RawImage { ... }
//   },
//   ...
// ]
```

You can visualize the outputs with:
```js
for (const l of output) {
  l.mask.save(`${l.label}.png`);
}
```

---

Note: Having a separate repo for ONNX weights is intended to be a temporary solution until WebML gains more traction. If you would like to make your models web-ready, we recommend converting to ONNX using [🤗 Optimum](https://huggingface.co/docs/optimum/index) and structuring your repo like this one (with ONNX weights located in a subfolder named `onnx`).