
## Key Features of This SECURITY.md

This security policy follows industry best practices and is specifically tailored for your ML project:

### GitHub-Only Communication
- Uses GitHub Security Advisories as the primary reporting channel
- No email addresses or external websites required
- Leverages GitHub's built-in private vulnerability reporting feature

### ML/AI-Specific Security
- Addresses model poisoning and adversarial attacks
- Covers model integrity and provenance tracking
- Includes dataset validation considerations
- Warns about privacy leakage through model outputs

### Clear Process
- Defines supported versions
- Provides step-by-step reporting instructions
- Sets clear response timelines (48h initial, 7d assessment, 30-90d resolution)
- Follows coordinated disclosure principles

### Practical Guidance
- Security checks for dependencies (`pip-audit`)
- Secret detection tools (`detect-secrets`)
- GitHub Actions secrets management
- Modal training security considerations

### Professional Structure
- Acknowledgment policy for security researchers
- References to OWASP ML Security and PyTorch security docs
- Versioned and dated for transparency

You can create this file by visiting `https://github.com/d-oit/tiny-cats-model/new/main` and pasting the content, or use the GitHub CLI: `gh repo clone d-oit/tiny-cats-model && cd tiny-cats-model && [create file] && git add SECURITY.md && git commit -m "docs: add SECURITY.md" && git push`.
