# ADR-004: Frontend Cat Model Update

## Status
Accepted

## Date
2026-02-23

## Context
The frontend was adapted from the landscape model (amins01/tiny-models) but still contains references to:
- Landscape model images in `frontend/public/`
- Old text content about semantic image generation
- Remnant components and utilities

## Decision
We will update all frontend assets to use the cat classifier model:
1. Remove landscape/public sample images
2. Update/remove documentation text about generation
3. Clean up unused utilities and components

## Consequences
### Positive
- Cleaner codebase without dead assets
- Consistent branding for cat classifier
- Smaller bundle size

### Negative
- None identified

## Alternatives Considered
1. Keep all assets - rejected (dead weight)
2. Create separate docs page - out of scope

## Notes
### Completed Actions (2026-02-23)
- Removed 10 unused landscape images from `frontend/public/`
- Removed 4 unused ONNX model files from `frontend/public/models/`
- Removed unused sentence-transformers directory `frontend/public/all-MiniLM-L6-v2/`
- Updated `index.html` title to "Tiny Cats Classifier"
- Verified build succeeds with clean codebase
- Base path correctly set to `/tiny-cats-model/`
