# Model Card: Tiny Cats Classifier and Generator

## Intended use

The classifier is intended to distinguish cat images from the negative examples
used in its training dataset. It is not a breed classifier and should not be
used as a general-purpose detector of every possible non-cat. The separate
TinyDiT generator synthesizes cat images conditioned on one of its supported
categories; generated images are illustrative, not evidence about real animals.

## Tasks and labels

### Binary classifier

The ImageFolder training folders and output indices are:

| Output index | Dataset folder | Public label |
|---:|---|---|
| 0 | `cat/` | `cat` |
| 1 | `other/` | `not_cat` |

The model emits two logits. Consumers that need probabilities must apply a
numerically stable softmax (subtract the maximum logit before exponentiation).

### Conditional generator

The generator has 13 conditions: 12 cat breeds and `Other`. Its conditioning
indices are stable and encoded in model weights:

| Index | Condition |
|---:|---|
| 0 | Abyssinian |
| 1 | Bengal |
| 2 | Birman |
| 3 | Bombay |
| 4 | British_Shorthair |
| 5 | Egyptian_Mau |
| 6 | Maine_Coon |
| 7 | Persian |
| 8 | Ragdoll |
| 9 | Russian_Blue |
| 10 | Siamese |
| 11 | Sphynx |
| 12 | Other |
| 13 | Unconditional/null (CFG only; not a user-facing breed condition) |

## Input preprocessing

Classifier inputs are decoded as RGB, resized bilinearly with the shorter side at
256 pixels while preserving aspect ratio, and centre-cropped to 224 × 224.
Pixel values are scaled to `[0, 1]`, normalized channel-wise with ImageNet mean
`[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`, then
represented as float32 NCHW. ImageNet normalization does not bound tensor values
to `[-1, 1]`. The canonical preprocessing and task metadata are recorded in
`configs/model_contract.json` and mirrored to the frontend generated copy.
## Metrics and evaluation

Classifier quality is measured as binary classification on held-out evaluation
data, including accuracy, precision/recall/F1 and confusion-matrix metrics. The
repository evaluation report dated 2026-08-12 records 1,473/1,478 correct
predictions (accuracy 0.9966; macro-F1 0.9961), with confusion matrix
`[[476, 3], [2, 997]]` in `cat`, `other` order. This is a snapshot for that
checkpoint and split, not a guarantee of current artifact performance.
Evaluation uses the configured validation/test split; a persisted split
manifest can be supplied to `src/eval.py` to reproduce membership. Metrics are
dataset- and split-dependent; generator quality is assessed separately using
generation metrics and visual inspection; classifier metrics do not measure
image fidelity.

## Limitations and risks

- The classifier's negative class consists of dogs, not a representative sample
  of all non-cats. Images of other animals, objects, drawings, or unusual inputs
  may be misclassified.
- The classifier predicts only cat/not-cat. **Breed classification is out of
  scope**; breed names are generator conditions, not classifier labels.
- Dataset and demographic coverage are limited; performance can vary with
  image quality, pose, lighting, domain, and breed representation.
- Generator output is synthetic and may contain artifacts or fail to match the
  requested condition. Do not use it for consequential decisions.

## Quantization

The project publishes an FP32 classifier and an INT8 quantized classifier for
smaller/faster deployment. Quantization can change predictions and may reduce
accuracy; evaluate the exact artifact and target runtime before deployment.
The generator is a separate synthesis model and its output should not be
interpreted as classifier output.
