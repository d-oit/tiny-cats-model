# ADR-018: TypeScript 5.8 Upgrade for Frontend Build

**Date:** 2026-02-25
**Status:** Implemented
**Authors:** AI Agent

## Context

The CI pipeline was failing on the `Build Frontend` job with the error:
```
tsconfig.node.json(21,5): error TS5023: Unknown compiler option 'erasableSyntaxOnly'.
```

Investigation revealed:
- `package.json` specified `"typescript": "~5.7.2"`
- `tsconfig.node.json` used `"erasableSyntaxOnly": true`
- The `erasableSyntaxOnly` flag was introduced in **TypeScript 5.8** (released February 2025)

## Decision

Upgrade TypeScript to version 5.8+ in `frontend/package.json`:

```json
"typescript": "~5.8.0"
```

## Implementation

1. Updated `frontend/package.json` to use TypeScript 5.8.0
2. Reinstalled dependencies (`npm install`)
3. Verified build succeeds (`npm run build`)
4. Ran quality gate - all checks pass

## Consequences

- ✅ Frontend builds successfully in CI
- ✅ Uses modern TypeScript 5.8 features (erasableSyntaxOnly)
- ✅ Follows 2026 best practices for TypeScript
- ⚠️ Requires Node.js 18+ (already required by Vite 6)

## References

- [TypeScript 5.8 Release Notes](https://devblogs.microsoft.com/typescript/announcing-typescript-5-8/)
- [erasableSyntaxOnly Flag](https://egghead.io/use-the-erasable-syntax-only-type-script-compilation-flag)
