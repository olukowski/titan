# Contributing to Titan

Thanks for helping build Titan. Titan has no backwards-compatibility guarantees, so prefer the
best design over preserving old behavior.

## Workflow

1. Fork `titan-engine/titan` and create a branch from `main`.
2. Make one focused change. Run the checks from [`AGENTS.md`](AGENTS.md#verify-before-you-finish).
3. Open a pull request against `main` and fill in the template. CI must pass (the `ci` check).
4. PRs are squash-merged, so the PR title becomes the commit title on `main`.

## AI-assisted contributions

AI-assisted contributions are welcome. You, the author, are responsible for everything you
submit:

- **Readability.** You understand the code and can explain every line.
- **Correctness.** It does what the PR says, including edge cases.
- **Verification.** You ran the checks and say how you verified the change.
- **Focus.** Keep PRs small and on-topic; no unrelated refactors or generated noise.

Agents should follow [`AGENTS.md`](AGENTS.md).

## Issues

Titan uses GitHub's native **issue types** for what kind of issue it is: `Bug`, `Feature`, or
`Task`. The issue forms set the type for you.

**Labels** describe what an issue touches or its state, such as `wasm`, `performance`,
`binary-size`, `breaking`, `blocked`, or `good first issue`. There are no `bug` or
`enhancement` labels, because issue types cover those.

## Security

Report vulnerabilities privately as described in [`SECURITY.md`](SECURITY.md), not in public
issues.

## License

Titan is dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
Titan by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
