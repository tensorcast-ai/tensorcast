# Security Policy

## Supported Versions

Security fixes are provided for the current public release line. Older
pre-release snapshots and private development branches are not supported as
public security-maintenance targets.

## Reporting A Vulnerability

Please report suspected vulnerabilities privately to `ychu.luo@gmail.com`.

Do not open a public GitHub issue, discussion, or pull request with exploit
details before coordination. Include:

- affected TensorCast version or commit;
- operating system, CUDA, Python, and PyTorch versions where relevant;
- a minimal reproduction or clear impact description;
- whether the issue is already public.

We will acknowledge reports as soon as practical, triage severity, coordinate a
fix or mitigation, and publish public details after users have a reasonable path
to update.

## Scope

In scope:

- TensorCast source code in this repository;
- packaged TensorCast Python wheels and native binaries distributed from the
  official project channels;
- default public examples and configuration files.

Out of scope:

- vulnerabilities in user deployments caused by custom configuration;
- unrelated third-party services or infrastructure;
- historical private commits, old branches, or deleted files outside the
  maintained public release state.
