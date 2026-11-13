## Description

<!-- Provide a brief description of the changes introduced by this PR. -->

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] CI/CD or tooling enhancement

## Behavior and Compatibility

<!-- For runtime changes, describe the observable contract and the path through
callers, adapters, storage, and readers. Include a concrete before/after example
where useful, plus any API or adapter migration requirements.
For documentation or cosmetic changes, mark runtime sections N/A. -->

## Verification Evidence

<!-- Map changed contracts to behavioral tests. Name the real internal components
exercised and relevant failure, retry, denied-scope, or boundary scenarios.
For bug fixes, state whether the regression failed before the fix.
List commands and results, plus skipped checks and unverified assumptions. -->

## Independent Review

<!-- For authorization, concurrency, or recovery changes, identify the independent
reviewer and outcome before merging. Otherwise mark N/A. -->

## Checklist

- [ ] My code adheres to the project's coding style guidelines (`ruff check .` and `ruff format --check .`)
- [ ] Changed runtime contracts and compatibility requirements are documented above (or N/A explained)
- [ ] Behavioral tests exercise the changed component boundaries and relevant failure scenarios (or N/A explained)
- [ ] Verification commands, results, and skipped checks are recorded above
- [ ] I have updated documentation where appropriate
