---
title: Security Checks
description: Public security automation and local security checks
---

# Security Checks

TensorCast runs public security automation without private credentials:

- CodeQL for Python and C/C++;
- gitleaks secret scanning using `.gitleaks.toml`;
- `pip-audit` against the locked Python dependency set;
- OpenSSF Scorecard;
- Dependabot for GitHub Actions and Python dependency updates.

Local checks:

```bash
gitleaks detect --no-git --redact
uv lock
uv export --locked --all-extras --dev --format requirements-txt --output-file /tmp/tensorcast-requirements.txt
uv tool run pip-audit --progress-spinner off -r /tmp/tensorcast-requirements.txt
```

The only secret-scan allowlist is the test-only mTLS fixture in
`daemon/util/grpc_daemon_transport_test.cc`.
