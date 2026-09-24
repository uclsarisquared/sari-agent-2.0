# CHANGELOG

<!-- version list -->

## v1.8.6 (2026-09-24)

### Bug Fixes

- **analysis**: Repair script paths, bash 3.2 runner and tie-aware Spearman
  ([`a6b61e3`](https://github.com/uclsarisquared/sari-agent-2.0/commit/a6b61e38f154f79588477c5d06e98f6f33707e9b))

- **configs**: Make comprehensive ablation runner work on macOS bash
  ([`e60df22`](https://github.com/uclsarisquared/sari-agent-2.0/commit/e60df2274276d6f4ae12735619f6818bee4425dd))

### Continuous Integration

- Serialize release runs and align semantic-release floor
  ([`362a878`](https://github.com/uclsarisquared/sari-agent-2.0/commit/362a87883b90b11060427fdac3ccadd9d6a05245))

### Documentation

- Describe VL bbox pointing as primary and moondream as deprecated
  ([`25adc14`](https://github.com/uclsarisquared/sari-agent-2.0/commit/25adc14635f69f2741e448ef7725dbe0c259c6c6))


## v1.8.5 (2026-09-24)

### Bug Fixes

- **mapping**: Stop phantom obstacles and fix edge dedup and vantage arrival
  ([`e12cb28`](https://github.com/uclsarisquared/sari-agent-2.0/commit/e12cb28ceeb1ab868e2fb5954f8b2a645eed34c2))

- **watch**: Fix completion notice, log deltas and dashboard state bugs
  ([`a75cfed`](https://github.com/uclsarisquared/sari-agent-2.0/commit/a75cfed1f3065c26fb4229cb0680fc7d5bc08ecf))


## v1.8.4 (2026-09-24)

### Bug Fixes

- **orchestrator**: Preserve leg evidence and fix compare/adaptive bookkeeping
  ([`e465920`](https://github.com/uclsarisquared/sari-agent-2.0/commit/e465920128542c0be9a5a7081e304382938a4645))

- **vision**: Harden checkout alignment, perception timeouts and sim sends
  ([`5303f05`](https://github.com/uclsarisquared/sari-agent-2.0/commit/5303f05f08a28103985d3bf9cac3696eb5891ef9))


## v1.8.3 (2026-09-24)

### Bug Fixes

- **agent-core**: Fix navigation ranking, history, token dump and config validation bugs
  ([`f371a1a`](https://github.com/uclsarisquared/sari-agent-2.0/commit/f371a1a46c6ce721215adc6f1d41f103ef6ae26e))


## v1.8.2 (2026-09-24)

### Chores

- Sync uv.lock version
  ([`185b857`](https://github.com/uclsarisquared/sari-agent-2.0/commit/185b857ea1fe19b61283292ebc8f5b58bb175d62))

### Performance Improvements

- **mapping**: Vectorize scan projection, planner and coverage hot loops
  ([`2655d05`](https://github.com/uclsarisquared/sari-agent-2.0/commit/2655d050939be6c05b837a3013bf2d38ab8763ea))

- **watch**: Cache attempt views, add ETag and gzip to dashboard
  ([`e61e6c1`](https://github.com/uclsarisquared/sari-agent-2.0/commit/e61e6c1b17c058d4cc8a720ab1c27d18103c082c))

### Refactoring

- **agent-core**: Dedupe LLM/runtime helpers and encode frames once
  ([`b819514`](https://github.com/uclsarisquared/sari-agent-2.0/commit/b819514195cd2960c8f098ac25408e6c8143ae62))

- **analysis**: Share ablation script and analysis helpers
  ([`7b90c25`](https://github.com/uclsarisquared/sari-agent-2.0/commit/7b90c2514d9e8a45200df17a74b47eb4e3175e1b))

- **bench**: Share storage helpers and cache winner polling
  ([`c39a509`](https://github.com/uclsarisquared/sari-agent-2.0/commit/c39a5097efefa4b2b7685c5d7ccd5575a6c3a247))

- **orchestrator**: Collapse duplicated guard and inspection code
  ([`97b16a3`](https://github.com/uclsarisquared/sari-agent-2.0/commit/97b16a3dca51c5ad303411c875823291d754b7fd))

- **vision**: Drop dead perception/manipulation code and share sim helpers
  ([`f462b1e`](https://github.com/uclsarisquared/sari-agent-2.0/commit/f462b1e8c0fbf74632705587fb4af64f754d7c7b))


## v1.8.1 (2026-09-05)

### Bug Fixes

- **watch**: Normalize symlinked battery root for frame paths
  ([`c5cc594`](https://github.com/uclsarisquared/sari-agent-2.0/commit/c5cc5944b04a38a0bc7b0ffd6796482f47568d3b))

### Chores

- Large cleanup of comments and dead code
  ([`7e5bfbd`](https://github.com/uclsarisquared/sari-agent-2.0/commit/7e5bfbd86e213ad05501944ca358905564ad1ec2))

### Testing

- **bench**: Cover orphan command matching on every platform
  ([`78e310a`](https://github.com/uclsarisquared/sari-agent-2.0/commit/78e310a61e7401fdb12ad716c85b6da07e90f30d))

- **bench**: Run package-local tests in the validation suite
  ([`e77b6b0`](https://github.com/uclsarisquared/sari-agent-2.0/commit/e77b6b0cfe957dba5cf5646e30735d436ad23c50))

- **validation**: Refresh agent and mapping test contracts
  ([`e02b4a9`](https://github.com/uclsarisquared/sari-agent-2.0/commit/e02b4a953aa28707e80fb562834d6c140c83923c))


## v1.8.0 (2026-09-02)

### Bug Fixes

- **watch**: Separate watcher configuration
  ([`c14dea6`](https://github.com/uclsarisquared/sari-agent-2.0/commit/c14dea6cb6700d91a748239f11e7271bf90ef5f1))

### Chores

- **deps**: Refresh uv lockfile markers
  ([`014ffd8`](https://github.com/uclsarisquared/sari-agent-2.0/commit/014ffd8d2a9f4ea37daaa5afb0fa863a4d66a31b))

### Features

- **ablation**: Run comprehensive prompt battery
  ([`92ea6ea`](https://github.com/uclsarisquared/sari-agent-2.0/commit/92ea6eae489a8b20332c7f27e1bd2245ada2d726))

- **bench**: Add safe benchmark artifact optimization
  ([`226b51c`](https://github.com/uclsarisquared/sari-agent-2.0/commit/226b51c77d7bd83dad38cbec3023f680c013f7a2))

- **watch**: Send completion notifications with replay clips
  ([`829daf5`](https://github.com/uclsarisquared/sari-agent-2.0/commit/829daf5434a7b71e53c8466d5dd237e5eb03b183))


## v1.7.1 (2026-08-30)

### Bug Fixes

- **bench**: Include agent response in Discord alerts
  ([`977a8f5`](https://github.com/uclsarisquared/sari-agent-2.0/commit/977a8f599f75444acb775afc5edea5713b7b439b))


## v1.7.0 (2026-08-27)

### Features

- **ocr**: Add configurable CUDA backend
  ([`c8223a9`](https://github.com/uclsarisquared/sari-agent-2.0/commit/c8223a9c2efc5d6113ad67bbedc376582f849df5))


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
