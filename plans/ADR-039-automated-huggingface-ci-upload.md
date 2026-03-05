# ADR-039: Automated HuggingFace CI Upload

**Date:** 2026-02-27
**Status:** Proposed
**Authors:** AI Agent (GOAP System)
**Related:** GOAP.md Phase 17.4, ADR-026 (HuggingFace Publishing), ADR-035 (Full Model Training)

## Context

### Current State

The tiny-cats-model project has:
- **Upload script:** `src/upload_to_huggingface.py` (660 lines)
- **HuggingFace repo:** d4oit/tiny-cats-model (models uploaded)
- **Manual upload process:** Run script locally after training
- **HF_TOKEN:** Stored as environment variable

### Problem Statement

Current upload workflow is **manual and error-prone**:
1. Train model on Modal
2. Export checkpoint from volume
3. Run upload script locally
4. Verify upload succeeded
5. Update documentation

**Issues:**
- ❌ Easy to forget upload step
- ❌ Manual process introduces delays
- ❌ No automated verification
- ❌ Inconsistent model card updates
- ❌ Requires local HF_TOKEN setup

### Requirements

**2026 Best Practices for Model Deployment:**
1. **Automated deployment** - CI/CD pipeline
2. **Secret management** - Secure token storage
3. **Conditional triggers** - Upload only on success
4. **Verification** - Confirm upload succeeded
5. **Rollback support** - Revert bad uploads
6. **Audit trail** - Track all uploads

## Decision

We will implement **automated HuggingFace upload** in the CI/CD pipeline.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Training Workflow                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Train Model  │───▶│   Evaluate   │───▶│   Upload     │  │
│  │  (Modal)     │    │   Metrics    │    │  (Auto)      │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  checkpoints/         evaluation/         HuggingFace       │
│  tinydit_final.pt     report.json         Hub CDN           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### GitHub Actions Workflow

**File:** `.github/workflows/upload-hub.yml`

```yaml
name: Upload to HuggingFace Hub

on:
  workflow_run:
    workflows: ["Train"]
    types:
      - completed
    branches:
      - main

jobs:
  upload:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    timeout-minutes: 30
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install huggingface_hub
      
      - name: Download latest checkpoint
        uses: dawidd6/action-download-artifact@v3
        with:
          workflow: train.yml
          branch: main
          name: checkpoints
          path: checkpoints/
      
      - name: Download evaluation report
        uses: dawidd6/action-download-artifact@v3
        with:
          workflow: train.yml
          branch: main
          name: evaluation-report
          path: evaluation/
      
      - name: Upload to HuggingFace Hub
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          python src/upload_to_huggingface.py \
            --generator checkpoints/tinydit_final.pt \
            --onnx-generator frontend/public/models/generator_quantized.onnx \
            --evaluation-report evaluation/evaluation_report.json \
            --benchmark-report benchmark_report.json \
            --samples-dir samples/evaluation \
            --repo-id d4oit/tiny-cats-model \
            --commit-message "Auto-upload: $(git rev-parse --short HEAD)"
      
      - name: Verify upload
        run: |
          python -c "
          from huggingface_hub import list_repo_files
          files = list_repo_files('d4oit/tiny-cats-model')
          assert 'generator/model.pt' in files
          assert 'generator/model.onnx' in files
          print('✅ Upload verified successfully')
          "
      
      - name: Upload failure notification
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: 'HuggingFace Upload Failed',
              body: `Upload failed in workflow ${context.workflow}
              
              Check logs: ${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}
              
              Commit: ${context.sha}
              `
            })
```

### Secret Management

**GitHub Secrets Setup:**

```bash
# Add HF_TOKEN secret via CLI
gh secret set HF_TOKEN --body "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Verify secret exists
gh secret list

# Or via GitHub UI:
# Settings → Secrets and variables → Actions → New repository secret
# Name: HF_TOKEN
# Value: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Security Best Practices:**
1. ✅ Use repository secrets (not environment variables)
2. ✅ Limit secret access to specific workflows
3. ✅ Rotate tokens periodically
4. ✅ Use fine-grained tokens (write access only)
5. ✅ Never log token values
6. ✅ Audit secret usage

### Conditional Upload Triggers

**Upload only on specific conditions:**

```yaml
# Upload only on main branch
if: github.ref == 'refs/heads/main'

# Upload only for tagged releases
if: startsWith(github.ref, 'refs/tags/')

# Upload only if loss improved
if: ${{ needs.evaluate.outputs.improved == 'true' }}

# Upload only on schedule (weekly)
on:
  schedule:
    - cron: '0 0 * * 0'  # Every Sunday
```

### Upload Verification

**Post-upload checks:**

```python
# verify_upload.py
from huggingface_hub import HfApi, list_repo_files

api = HfApi()

def verify_upload(repo_id: str) -> bool:
    """Verify all expected files are present."""
    expected_files = [
        "README.md",
        "generator/model.pt",
        "generator/model.onnx",
        "classifier/model.onnx",
        "evaluation/evaluation_report.json",
        "benchmarks/benchmark_report.json",
    ]
    
    files = list_repo_files(repo_id)
    
    for expected in expected_files:
        if expected not in files:
            print(f"❌ Missing: {expected}")
            return False
    
    print("✅ All files present")
    
    # Verify model card
    model_card = api.model_info(repo_id)
    if not model_card.cardData:
        print("⚠️  Model card missing")
        return False
    
    print("✅ Model card present")
    return True

if __name__ == "__main__":
    success = verify_upload("d4oit/tiny-cats-model")
    exit(0 if success else 1)
```

### Rollback Strategy

**Revert bad uploads:**

```yaml
# .github/workflows/rollback.yml
name: Rollback HuggingFace Upload

on:
  workflow_dispatch:
    inputs:
      commit_hash:
        description: 'Commit hash to revert to'
        required: true

jobs:
  rollback:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      
      - name: Install huggingface_hub
        run: pip install huggingface_hub
      
      - name: Revert to previous commit
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          python -c "
          from huggingface_hub import HfApi
          api = HfApi()
          
          # Get commit history
          commits = api.list_repo_commits('d4oit/tiny-cats-model')
          
          # Find target commit
          target = None
          for commit in commits:
              if commit.title.startswith('${{ github.event.inputs.commit_hash }}'):
                  target = commit
                  break
          
          if not target:
              raise ValueError('Commit not found')
          
          print(f'Reverting to commit {target.commit_id}')
          
          # Note: HuggingFace doesn't support git-style revert
          # This is a placeholder for manual intervention
          print('Manual revert required - see documentation')
          "
```

### Audit Trail

**Track all uploads:**

```yaml
# Add to upload workflow
- name: Log upload to audit file
  run: |
    echo "$(date -u) | $(git rev-parse HEAD) | $(git log -1 --pretty=%an)" >> audit.log
    
- name: Upload audit log
  uses: actions/upload-artifact@v4
  with:
    name: audit-log
    path: audit.log
```

**Audit log format:**
```
2026-02-27T19:30:00Z | abc1234 | John Doe
2026-02-28T10:15:00Z | def5678 | Jane Smith
```

## Implementation

### Phase 1: Secret Setup (Pending)
- [ ] Create HF_TOKEN in GitHub Secrets
- [ ] Verify token has write permissions
- [ ] Test secret access in workflow

### Phase 2: Workflow Creation (Pending)
- [ ] Create `.github/workflows/upload-hub.yml`
- [ ] Add artifact download steps
- [ ] Add upload step with HF_TOKEN
- [ ] Add verification step

### Phase 3: Testing (Pending)
- [ ] Test workflow with manual trigger
- [ ] Verify upload succeeds
- [ ] Test failure notification
- [ ] Test rollback procedure

### Phase 4: Documentation (Pending)
- [ ] Update README with upload workflow
- [ ] Document secret setup in AGENTS.md
- [ ] Add troubleshooting guide

## Workflow Integration

### Full Training → Upload Pipeline

```yaml
# .github/workflows/train.yml (excerpt)
name: Train

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  train-dit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Modal training
        uses: modal-labs/modal-action@v1
        with:
          script: src/train_dit.py
          args: --steps 200000
          token: ${{ secrets.MODAL_TOKEN }}
      
      - name: Save checkpoint
        run: |
          modal volume get checkpoints/tinydit_final.pt
      
      - name: Upload checkpoint artifact
        uses: actions/upload-artifact@v4
        with:
          name: checkpoints
          path: checkpoints/
      
      - name: Run evaluation
        run: python src/evaluate_full.py --checkpoint checkpoints/tinydit_final.pt
      
      - name: Upload evaluation artifact
        uses: actions/upload-artifact@v4
        with:
          name: evaluation-report
          path: evaluation_report.json

# Triggers upload-hub.yml on success
```

### Alternative: Single Workflow

```yaml
# Combined training + upload in one workflow
name: Train and Upload

on:
  push:
    branches: [main]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      # ... training steps ...
  
  upload:
    needs: train
    runs-on: ubuntu-latest
    steps:
      # ... upload steps ...
```

## Consequences

### Positive
- ✅ **Fully automated** - No manual intervention
- ✅ **Consistent uploads** - Same process every time
- ✅ **Fast deployment** - Upload immediately after training
- ✅ **Audit trail** - Track all uploads
- ✅ **Failure notification** - Know when uploads fail
- ✅ **Secure** - Tokens managed by GitHub

### Negative
- ⚠️ **CI minutes** - Uses GitHub Actions minutes
- ⚠️ **Complexity** - More workflow files to maintain
- ⚠️ **Artifact storage** - Requires artifact retention
- ⚠️ **Token rotation** - Need to update secrets periodically

### Neutral
- ℹ️ **Upload frequency** - Depends on training frequency
- ℹ️ **Artifact retention** - Default 90 days
- ℹ️ **Concurrent uploads** - Handled by GitHub queue

## Alternatives Considered

### Alternative 1: Manual Upload Only
**Proposal:** Keep manual upload process.

**Rejected because:**
- Error-prone and slow
- Easy to forget
- Not scalable
- Industry standard is automated

### Alternative 2: Upload from Modal
**Proposal:** Upload directly from Modal container after training.

**Rejected because:**
- Requires HF_TOKEN in Modal secrets (another provider)
- Less visibility into upload process
- Harder to verify and rollback
- GitHub Actions has better secret management

### Alternative 3: Upload on Every Commit
**Proposal:** Upload after every push to any branch.

**Rejected because:**
- Too many uploads (spam)
- Wastes CI minutes
- HuggingFace repo clutter
- Better to upload only on main

### Alternative 4: Scheduled Uploads
**Proposal:** Upload once per week on schedule.

**Partially adopted:**
- Can add scheduled re-upload for freshness
- Not primary upload mechanism
- Good for model card updates

## Security Considerations

### Token Permissions

**Minimum required permissions:**
- ✅ `read` - Read repo metadata
- ✅ `write` - Upload files to owned repos
- ❌ `delete` - Not needed (use rollback instead)
- ❌ `admin` - Not needed

### Secret Rotation

**Rotate HF_TOKEN every 90 days:**
```bash
# Generate new token
# https://huggingface.co/settings/tokens

# Update GitHub secret
gh secret set HF_TOKEN --body "hf_new_token_here"

# Verify old token is revoked
# https://huggingface.co/settings/tokens
```

### Access Control

**Limit workflow access:**
```yaml
# Only allow specific branches
on:
  push:
    branches: [main]

# Only allow specific users (optional)
permissions:
  contents: read
```

## Troubleshooting

### Issue: "403 Forbidden"
**Cause:** Invalid or expired HF_TOKEN
**Solution:**
1. Generate new token at https://huggingface.co/settings/tokens
2. Update GitHub secret: `gh secret set HF_TOKEN`
3. Re-run workflow

### Issue: "Artifact not found"
**Cause:** Training workflow didn't upload artifact
**Solution:**
1. Check training workflow completed successfully
2. Verify artifact name matches
3. Check artifact retention period

### Issue: "Upload timeout"
**Cause:** Large model files
**Solution:**
1. Increase workflow timeout
2. Use resumable uploads
3. Split large files

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Upload automation | 100% automated | Workflow runs |
| Upload success rate | >95% | CI run history |
| Time to upload | <10 minutes | Workflow duration |
| Failure detection | Immediate | Notification latency |
| Rollback time | <30 minutes | Manual + workflow |

## Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Phase 1 | 30 minutes | Secret setup |
| Phase 2 | 2-3 hours | Workflow creation |
| Phase 3 | 1-2 hours | Testing |
| Phase 4 | 1 hour | Documentation |
| **Total** | **4.5-6.5 hours** | **~1 day** |

## References

- GitHub Actions Secrets: https://docs.github.com/en/actions/security-guides/encrypted-secrets
- HuggingFace Hub API: https://huggingface.co/docs/huggingface_hub
- ADR-026: HuggingFace Model Publishing
- ADR-035: Full Model Training Plan

## Appendix: Complete Workflow File

```yaml
# .github/workflows/upload-hub.yml
name: Upload to HuggingFace Hub

on:
  workflow_run:
    workflows: ["Train"]
    types: [completed]
    branches: [main]

jobs:
  upload:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    timeout-minutes: 30
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install huggingface_hub
      
      - name: Download checkpoint
        uses: dawidd6/action-download-artifact@v3
        with:
          workflow: train.yml
          branch: main
          name: checkpoints
          path: checkpoints/
      
      - name: Download evaluation
        uses: dawidd6/action-download-artifact@v3
        with:
          workflow: train.yml
          branch: main
          name: evaluation-report
          path: evaluation/
      
      - name: Upload to HuggingFace
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          python src/upload_to_huggingface.py \
            --generator checkpoints/tinydit_final.pt \
            --onnx-generator frontend/public/models/generator_quantized.onnx \
            --evaluation-report evaluation/evaluation_report.json \
            --benchmark-report benchmark_report.json \
            --samples-dir samples/evaluation \
            --repo-id d4oit/tiny-cats-model \
            --commit-message "Auto-upload: $(git rev-parse --short HEAD)"
      
      - name: Verify upload
        run: |
          python -c "
          from huggingface_hub import list_repo_files
          files = list_repo_files('d4oit/tiny-cats-model')
          assert 'generator/model.pt' in files
          print('✅ Upload verified')
          "
      
      - name: Notify on failure
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: '🚨 HuggingFace Upload Failed',
              body: `Upload failed in workflow run ${context.runId}
              
              [View logs](${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId})
              
              Commit: ${context.sha}
              Triggered by: ${context.actor}
              `
            })
```
