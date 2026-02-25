# Security Policy for Training Monitoring

## Security Checks

The training monitor performs real-time scanning of all output for security issues.

### 1. Secret Exposure

**Detection Pattern:**
```regex
(api_key|token|password|secret)\s*=\s*["\'][^"\']+["\']
```

**Examples (DO NOT COMMIT THESE):**
```python
# BAD - Will trigger alert
api_key = "sk-1234567890"
token = "ghp_xxxxxxxxxxxx"

# GOOD - Use environment variables
api_key = os.environ.get("API_KEY")
```

**Action:** Immediate alert, log redaction recommended

### 2. Credential Leak

**Detection Pattern:**
```regex
(AWS_|GCP_|AZURE_|MODAL_)[A-Z_]+
```

**Examples:**
```bash
# BAD - Credentials in output
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export MODAL_TOKEN_ID=xxx

# GOOD - Use Modal secrets
modal.Secret.from_name("my-secrets")
```

**Action:** Critical alert, halt training, rotate credentials

### 3. Path Traversal

**Detection Pattern:**
```regex
\.\.[/\\]
```

**Examples:**
```python
# BAD - Potential path traversal
open("../../etc/passwd")

# GOOD - Use safe path joining
from pathlib import Path
safe_path = Path("/data") / "cats" / filename
```

**Action:** Warning alert, log for review

### 4. Shell Injection

**Detection Pattern:**
```regex
[;&|`$()]
```

**Examples:**
```python
# BAD - Shell injection risk
os.system(f"rm -rf {user_input}")

# GOOD - Use subprocess with list
subprocess.run(["rm", "-rf", safe_path])
```

**Action:** Warning alert, review required

## Security Best Practices

### Modal Secrets

```python
# Store secrets securely
modal.Secret.from_name("my-secrets")

# Access in function
@app.function(secrets=[modal.Secret.from_name("my-secrets")])
def train():
    import os
    api_key = os.environ["API_KEY"]  # Safe
```

### Environment Variables

```bash
# Set locally (never commit to .env)
export API_KEY="your-key-here"

# In Modal, use:
modal.Secret.from_dotenv("my-secrets")
```

### Volume Security

```python
# Mount only necessary directories
volume = modal.Volume.from_name("data")

@app.function(volumes={"/data": volume})
def train():
    # Access only mounted paths
    pass
```

## Incident Response

If security issue detected:

1. **Stop training immediately**
2. **Rotate exposed credentials**
3. **Review logs for scope**
4. **Update security patterns if needed**
5. **Document incident**

## Compliance

- Never log full credential values
- Redact sensitive patterns in reports
- Store security logs separately
- Review alerts within 24 hours
