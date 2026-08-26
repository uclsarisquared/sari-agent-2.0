# CHANGELOG

<!-- version list -->

## v1.6.0 (2026-08-26)

### Features

- **ablation**: Add time-limited run script for a5, a7 and replan-test policies
  ([`967bc5e`](https://github.com/uclsarisquared/sari-agent-2.0/commit/967bc5e98c70c8c25b434ee2b8456995ddf279ac))


## v1.5.0 (2026-08-26)

### Bug Fixes

- **release**: Pin GitPython below 3.1.46 to avoid psr breakage
  ([`3ac667e`](https://github.com/uclsarisquared/sari-agent-2.0/commit/3ac667e95209fd47a27329fdd63ebff24cabae6e))

- **release**: Pin python-semantic-release version to fix GitPython incompatibility
  ([`6851109`](https://github.com/uclsarisquared/sari-agent-2.0/commit/68511094fa24e708df91ea99f9b4ef4454afbb74))

- **watch**: Send custom User-Agent to Discord webhook
  ([`02c0c41`](https://github.com/uclsarisquared/sari-agent-2.0/commit/02c0c414a844492e0b993b51b05f58017c20f9f3))

### Chores

- Untrack editor directories
  ([`593d74b`](https://github.com/uclsarisquared/sari-agent-2.0/commit/593d74b95827a5f1ef7ceceb924acd093d72cc4c))

### Features

- **bench**: Add prompt-first queueing mode
  ([`33f2266`](https://github.com/uclsarisquared/sari-agent-2.0/commit/33f226641d7094548620a474b0e4d2cded40058f))


## v1.4.0 (2026-08-25)

### Chores

- Sync lockfile version
  ([`3968cb1`](https://github.com/uclsarisquared/sari-agent-2.0/commit/3968cb1cdf0a1684fa83b7a5363a342fe578a927))

### Features

- **bench**: Record battery elapsed clock and model identity
  ([`b02ae7e`](https://github.com/uclsarisquared/sari-agent-2.0/commit/b02ae7ef283eeb7dcb5ab953bc387e01f3ee2c20))

- **watch**: Move battery totals into overview panels
  ([`1943775`](https://github.com/uclsarisquared/sari-agent-2.0/commit/1943775a05207b7b7dac52c028741c2cca89f07f))


## v1.3.1 (2026-08-25)

### Bug Fixes

- **auth**: Harden Vertex credential refresh
  ([`3ec5f4c`](https://github.com/uclsarisquared/sari-agent-2.0/commit/3ec5f4cf772dfc05d077b74686b753ad843be793))

- **bench**: Preserve runtime failure diagnostics
  ([`7e5c55a`](https://github.com/uclsarisquared/sari-agent-2.0/commit/7e5c55a57ae9b7998f3456910ce7cdd6d2cdbb2f))

- **context**: Bound findings and event artifacts
  ([`e90cd53`](https://github.com/uclsarisquared/sari-agent-2.0/commit/e90cd530c43cb9120039af0502211ec4bf9f30a4))

- **vision**: Enforce structured bounding boxes
  ([`88f00d5`](https://github.com/uclsarisquared/sari-agent-2.0/commit/88f00d5b9882a92630953fcfad0e90c175bf4c8a))


## v1.3.0 (2026-08-24)

### Features

- **vision**: Treat plain-language target absence as a valid negative detection
  ([`1f81c52`](https://github.com/uclsarisquared/sari-agent-2.0/commit/1f81c52f7bd6bc7307ad2a2cb73cf74ae6379c88))


## v1.2.0 (2026-08-24)

### Features

- **bench**: Add replan-test run to the comprehensive ablation
  ([`23ef4fe`](https://github.com/uclsarisquared/sari-agent-2.0/commit/23ef4feb078e950a3a80eabd71c84891eb0aecd2))


## v1.1.0 (2026-08-24)

### Continuous Integration

- Auto-fill GitHub release notes from commits
  ([`2aba6db`](https://github.com/uclsarisquared/sari-agent-2.0/commit/2aba6dbd9f52df66ec448c644bee69acb91fda45))

- Restore python-semantic-release commit-based release notes
  ([`902da53`](https://github.com/uclsarisquared/sari-agent-2.0/commit/902da53f8add51311cd882d27ca3aea677d1c71a))

- Skip GitHub release when no new version is published
  ([`1816af1`](https://github.com/uclsarisquared/sari-agent-2.0/commit/1816af1971d2172df0db3e1db31da9f1f20ac084))

### Documentation

- Deprecate the moondream pointing path
  ([`0129997`](https://github.com/uclsarisquared/sari-agent-2.0/commit/0129997bdf8f7aed13dd6677a0f4aea2f46315b3))

- Note VLM bbox support is limited to Gemini and Qwen
  ([`03f4131`](https://github.com/uclsarisquared/sari-agent-2.0/commit/03f4131b71e102c2e7a2cce3a70cac322325b712))

### Features

- Add thinking-aware token budgets and provider-aware structured completion
  ([`c866dd4`](https://github.com/uclsarisquared/sari-agent-2.0/commit/c866dd4930ced19e68ccb0765431226aa9598355))

- **agent**: Preserve Gemini thought signature across actor history
  ([`65e34c7`](https://github.com/uclsarisquared/sari-agent-2.0/commit/65e34c77e2bbb9fbf946d3f1285b6e66b36c4763))

### Refactoring

- Route runtime and mapping calls through structured completion
  ([`48d0036`](https://github.com/uclsarisquared/sari-agent-2.0/commit/48d0036b5cb50d33b5dda9fde08a8c4cd02a7ee6))

### Testing

- Cover thinking-aware budgets, structured fallback, and signature retention
  ([`6aed012`](https://github.com/uclsarisquared/sari-agent-2.0/commit/6aed012e0dd5fbef1c2127902b8ff4d105c6c3ca))


## v1.0.3 (2026-08-24)

### Bug Fixes

- **agent**: Drop wrong coordinate-order claim from detect prompts
  ([`c8e9848`](https://github.com/uclsarisquared/sari-agent-2.0/commit/c8e984831f823117bb29daae8f4aa98c1dc22e92))


## v1.0.2 (2026-08-24)

### Bug Fixes

- **agent**: Branch bbox coord order on Vertex backend
  ([`4946302`](https://github.com/uclsarisquared/sari-agent-2.0/commit/4946302b6587e5c30aa3cda2738477fda1d66144))

- **agent**: Support Pro-tier Vertex thinking levels
  ([`51532f3`](https://github.com/uclsarisquared/sari-agent-2.0/commit/51532f309ee8cb9eacd762b4d69ab61a22290012))


## v1.0.1 (2026-08-24)

### Bug Fixes

- Make watch battery date match the UTC run stamp
  ([`1642646`](https://github.com/uclsarisquared/sari-agent-2.0/commit/1642646bdf3049e51c2dd3d53f6c250ae434dea3))


## v1.0.0 (2026-08-24)

- Initial Release
