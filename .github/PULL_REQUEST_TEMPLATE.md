## Summary

<!-- What problem does this solve, and why this approach? Link issues with "Closes #123". -->

## Changes

-

## Verification

<!-- Commands you ran and what you observed. -->

- [ ] `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
      `cargo test --workspace`, and `cargo check --workspace --target wasm32-unknown-unknown` pass
- [ ] The diff is minimal, readable, and focused on one change
- [ ] No leftover debug code (`dbg!`, `println!`, `todo!`, commented-out code)
