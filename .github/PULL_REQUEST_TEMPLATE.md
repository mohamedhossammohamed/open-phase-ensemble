## Description of Changes

Please provide a summary of the changes introduced by this pull request.

## Invariant Verification Checklist

- [ ] **Zero-Lookahead Invariant**: Changes strictly preserve element-by-element streaming zero-lookahead processing.
- [ ] **Determinism**: Seed fixing (`SEED=42`) is preserved.
- [ ] **Unit & Integration Tests**: All unit and integration tests pass cleanly (`PYTHONPATH=src pytest tests/`).
- [ ] **Documentation**: Updated MkDocs site documentation if applicable.
