#!/usr/bin/env bash
# data/download.sh — Download and prepare the Oxford IIIT Pet cats dataset
# Usage: bash data/download.sh
# No credentials required.

set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
CATS_DIR="${DATA_DIR}/cats"

echo "==> Preparing cats dataset in: ${CATS_DIR}"
mkdir -p "${CATS_DIR}/cat"
mkdir -p "${CATS_DIR}/other"

# Oxford IIIT Pet Dataset (public, no auth required)
PET_IMAGES_URL="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
PET_ANNOTS_URL="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "==> Downloading images archive..."
curl -sL -o "${TMP_DIR}/images.tar.gz" "${PET_IMAGES_URL}"

echo "==> Downloading annotations archive..."
curl -sL -o "${TMP_DIR}/annotations.tar.gz" "${PET_ANNOTS_URL}"

echo "==> Extracting images..."
tar -xzf "${TMP_DIR}/images.tar.gz" -C "${TMP_DIR}"

echo "==> Filtering cats (breeds starting with lowercase = cat breeds in Oxford Pet)..."
# In Oxford IIIT Pet: cat breeds have filenames starting with a capital letter
# e.g., Abyssinian_001.jpg, Bengal_001.jpg, etc.
# Dog breeds start with lowercase.
# We'll use the class list: Abyssinian, Bengal, Birman, Bombay, British_Shorthair,
# Egyptian_Mau, Maine_Coon, Persian, Ragdoll, Russian_Blue, Siamese, Sphynx

CAT_BREEDS=(
    "Abyssinian"
    "Bengal"
    "Birman"
    "Bombay"
    "British_Shorthair"
    "Egyptian_Mau"
    "Maine_Coon"
    "Persian"
    "Ragdoll"
    "Russian_Blue"
    "Siamese"
    "Sphynx"
)

echo "==> Copying cat images to ${CATS_DIR}/cat/"
for breed in "${CAT_BREEDS[@]}"; do
    find "${TMP_DIR}/images" -name "${breed}_*.jpg" -exec cp {} "${CATS_DIR}/cat/" \;
done

echo "==> Copying a sample of other (dog) images to ${CATS_DIR}/other/"
find "${TMP_DIR}/images" -name "*.jpg" | grep -v -E "$(IFS='|'; echo "${CAT_BREEDS[*]}")" \
    | head -n 500 \
    | xargs -I{} cp {} "${CATS_DIR}/other/"

CAT_COUNT=$(find "${CATS_DIR}/cat" -name '*.jpg' | wc -l)
OTHER_COUNT=$(find "${CATS_DIR}/other" -name '*.jpg' | wc -l)

echo ""
echo "==> Dataset ready:"
echo "    cat/   : ${CAT_COUNT} images"
echo "    other/ : ${OTHER_COUNT} images"
echo "    Total  : $((CAT_COUNT + OTHER_COUNT)) images"
echo ""
echo "==> Done. Dataset is at: ${CATS_DIR}"
