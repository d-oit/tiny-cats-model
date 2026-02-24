# ADR-011: Modal Container Dependencies Fix

## Status
Proposed

## Date
2026-02-24

## Context
The Modal training failed with error:
```
data/download.sh: line 23: wget: command not found
```

The `modal.yml` configuration specifies `apt_packages: ["wget", "unzip"]` but the actual Modal image in `src/train.py` uses:
```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "tar")
    ...
)
```

However, the `wget` command was not found in the container, indicating either:
1. The apt packages were not properly installed
2. The image build failed silently
3. The container uses a different base image

## Decision
We will implement a Python-based download fallback that doesn't require external tools:

1. **Primary**: Use Python's `urllib.request` for downloading (no external dependencies)
2. **Secondary**: Keep `wget` in apt_install for faster downloads when available
3. **Add**: `tarfile` module for extraction (built-in, no dependency)

### Implementation
```python
import urllib.request
import tarfile
from pathlib import Path

def download_file(url: str, dest: Path) -> None:
    """Download file using urllib with progress."""
    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 100 / totalsize
            print(f"\\rProgress: {percent:.1f}%", end='')
    
    urllib.request.urlretrieve(url, dest, reporthook)

def extract_tar_gz(archive: Path, dest: Path) -> None:
    """Extract tar.gz archive."""
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest)
```

## Consequences

### Positive
- **No external dependencies**: Works with minimal container images
- **More reliable**: Python stdlib is always available
- **Better progress reporting**: Can show download progress
- **Cross-platform**: Works on any Python environment

### Negative
- **Slower**: urllib may be slower than wget for large files
- **No resume**: urllib doesn't support resume like wget -c

### Neutral
- Slightly more code in download script
- Need to handle both methods

## Alternatives Considered

1. **Fix apt_install**: May work but depends on Modal image build
2. **Use requests library**: Adds dependency, not worth it for simple download
3. **Pre-upload dataset to Modal volume**: More complex setup

## Related
- ADR-007: Modal GPU Training Fix
- ADR-010: Modal Training Improvements
- `src/train.py`: Modal image configuration
- `data/download.sh`: Download script

## Implementation Plan
1. Update `data/download.sh` to use Python fallback
2. Or create `data/download.py` as alternative
3. Update Modal image to ensure apt packages install correctly
4. Test both methods
