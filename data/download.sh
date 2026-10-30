#!/usr/bin/env bash
# Download and atomically publish the Oxford IIIT Pet cats dataset.
# Usage: bash data/download.sh [--replace]
# No credentials required.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CATS_DIR="${DEST:-${CATS_DIR:-${SCRIPT_DIR}/cats}}"
CATS_DIR="$(mkdir -p "$(dirname "${CATS_DIR}")" && cd "$(dirname "${CATS_DIR}")" && pwd)/$(basename "${CATS_DIR}")"
PET_IMAGES_URL="${PET_IMAGES_URL:-https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz}"
PET_ANNOTS_URL="${PET_ANNOTS_URL:-https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz}"
DOWNLOAD_REPLACE="${DOWNLOAD_REPLACE:-0}"
VERSION="${DATASET_VERSION:-oxford-iiit-pet-v1}"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
  [[ -n "${PYTHON_BIN}" ]] || { echo "Python interpreter not found" >&2; exit 1; }
fi

for arg in "$@"; do
  case "$arg" in
    --replace) DOWNLOAD_REPLACE=1 ;;
    -h|--help) echo "Usage: bash data/download.sh [--replace]"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "${CATS_DIR}")"
LOCK_FILE="$(dirname "${CATS_DIR}")/.$(basename "${CATS_DIR}").download.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"${LOCK_FILE}"
  flock 9
fi

if [[ -n "${DOWNLOAD_WORKDIR:-}" ]]; then
  mkdir -p "${DOWNLOAD_WORKDIR}"
  WORK_DIR="$(mktemp -d "${DOWNLOAD_WORKDIR%/}/cats-download.XXXXXX")"
else
  WORK_DIR="$(mktemp -d)"
fi
STAGING="$(mktemp -d "$(dirname "${CATS_DIR}")/.cats-stage.XXXXXX")"
BACKUP="$(dirname "${CATS_DIR}")/.cats-backup.$$"
HAD_BACKUP=0
cleanup() {
  status=$?
  if [[ ${status} -ne 0 && ${HAD_BACKUP} -eq 1 && ! -e "${CATS_DIR}" ]]; then
    mv "${BACKUP}" "${CATS_DIR}" || echo "ERROR: failed restoring ${CATS_DIR} from ${BACKUP}" >&2
  fi
  rm -rf "${WORK_DIR}" "${STAGING}"
  if [[ ${HAD_BACKUP} -eq 1 && -e "${BACKUP}" && -e "${CATS_DIR}" ]]; then
    rm -rf "${BACKUP}"
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

fail() { echo "ERROR: $*" >&2; exit 1; }

printf '==> Downloading Oxford Pet archives\n'
curl --fail --location --retry 3 --retry-all-errors --connect-timeout 30 --max-time 1800 \
  --output "${WORK_DIR}/images.tar.gz" "${PET_IMAGES_URL}"
curl --fail --location --retry 3 --retry-all-errors --connect-timeout 30 --max-time 1800 \
  --output "${WORK_DIR}/annotations.tar.gz" "${PET_ANNOTS_URL}"

tar -tzf "${WORK_DIR}/images.tar.gz" >/dev/null || fail "Images archive is not a valid tar.gz"
tar -tzf "${WORK_DIR}/annotations.tar.gz" >/dev/null || fail "Annotations archive is not a valid tar.gz"
IMAGES_SHA256="$(sha256sum "${WORK_DIR}/images.tar.gz" | cut -d ' ' -f1)"
ANNOTS_SHA256="$(sha256sum "${WORK_DIR}/annotations.tar.gz" | cut -d ' ' -f1)"

tar -xzf "${WORK_DIR}/images.tar.gz" -C "${WORK_DIR}"
IMAGES_DIR="${WORK_DIR}/images"
[[ -d "${IMAGES_DIR}" ]] || fail "Images archive did not contain images/"
mkdir -p "${STAGING}/cat" "${STAGING}/other"

BREEDS=(Abyssinian Bengal Birman Bombay British_Shorthair Egyptian_Mau Maine_Coon Persian Ragdoll Russian_Blue Siamese Sphynx)
for breed in "${BREEDS[@]}"; do
  find "${IMAGES_DIR}" -type f -name "${breed}_*.jpg" -exec cp -- {} "${STAGING}/cat/" \;
done
find "${IMAGES_DIR}" -type f -name '*.jpg' -print0 | while IFS= read -r -d '' image; do
  name="$(basename "${image}")"
  is_cat=0
  for breed in "${BREEDS[@]}"; do
    if [[ "${name}" == "${breed}_"* ]]; then is_cat=1; break; fi
  done
  if [[ ${is_cat} -eq 0 ]]; then cp -- "${image}" "${STAGING}/other/"; fi
done

printf '==> Validating staged dataset\n'
"${PYTHON_BIN}" "${REPO_DIR}/scripts/validate_dataset.py" \
  --root "${STAGING}" --expected-breeds 12 --write-manifest \
  --source "${PET_IMAGES_URL} | ${PET_ANNOTS_URL}" --version "${VERSION}"
"${PYTHON_BIN}" -c 'import json,sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from dataset_manifest import read_dataset_manifest,write_dataset_manifest; root=Path(sys.argv[2]); m=read_dataset_manifest(root); m["archive_sha256"]={"images":sys.argv[3],"annotations":sys.argv[4]}; write_dataset_manifest(root,m)' \
  "${REPO_DIR}/src" "${STAGING}" "${IMAGES_SHA256}" "${ANNOTS_SHA256}"

if [[ -e "${CATS_DIR}" ]]; then
  EXISTING_HASH="$("${PYTHON_BIN}" -c 'import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from dataset_manifest import build_dataset_manifest,read_dataset_manifest; root=Path(sys.argv[2]); saved=read_dataset_manifest(root); current=build_dataset_manifest(root,use_cache=False,persist_cache=False); print(saved.get("manifest_hash","") if saved and saved.get("manifest_hash")==current.get("manifest_hash") else "")' "${REPO_DIR}/src" "${CATS_DIR}")"
  NEW_HASH="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(open(sys.argv[1]).read())["manifest_hash"])' "${STAGING}/dataset_manifest.json")"
  if [[ -n "${EXISTING_HASH}" && "${EXISTING_HASH}" == "${NEW_HASH}" ]]; then
    echo "==> Existing dataset has identical content; no-op."
    exit 0
  fi
  if [[ "${DOWNLOAD_REPLACE}" != 1 ]]; then
    fail "Dataset already exists with a different or missing manifest; pass --replace or set DOWNLOAD_REPLACE=1"
  fi
fi

if [[ -e "${CATS_DIR}" ]]; then
  mv "${CATS_DIR}" "${BACKUP}"
  HAD_BACKUP=1
fi
if ! mv "${STAGING}" "${CATS_DIR}"; then
  fail "Could not publish validated dataset"
fi

echo "==> Dataset published at ${CATS_DIR}"
