# TitusAI Style and Code Quality Guide

## Purpose

This document defines the preferred coding style, cleanup principles, and implementation discipline for TitusAI.

TitusAI is a small educational language-model project. The code should make the model, data, training, and inference easy to understand. Infrastructure should stay boring. Production-style machinery should not be added unless TitusAI has a concrete need for it.

The goal is not minimum line count. The goal is minimum cognitive load while preserving correctness and useful performance.

## 1. Core Principle

Prefer simple, direct, readable code whose intent is obvious on first inspection.

Readability is more important than fashionable Python idioms, formatter defaults, abstraction for its own sake, or patterns copied from larger production systems.

When deciding whether a change is simpler, ask:

1. Does it remove information the reader does not need?
2. Does it avoid repeating static information?
3. Does it make the data flow easier to follow?
4. Does an abstraction remove real complexity, or merely move it elsewhere?
5. Does the code solve a requirement TitusAI actually has?
6. Does any extra complexity buy a meaningful performance improvement?

A shorter file is not automatically simpler. Fewer functions are not automatically simpler. Fewer data structures are not automatically simpler.

## 2. TitusAI Scope and Guardrails

TitusAI is a learning resource, not a production platform.

Keep changes scoped to the task being worked on. Do not turn a cleanup into a repository-wide redesign.

Model architecture, parameter choices, training hyperparameters, corpus composition, and other learning decisions are not style details. Do not change them during cleanup or optimization work unless that change has been discussed and explicitly approved.

Do not introduce infrastructure for hypothetical future scale. Build what the current project needs.

A useful rule for TitusAI is:

> Teach the model, not the plumbing. Keep the plumbing boring.

## 3. Do Not Over-Engineer

Over-engineering is outside the scope of TitusAI.

Avoid adding features merely because they would be useful in a larger production system. In particular, do not add these without a concrete requirement:

- manifests;
- metadata sidecars;
- build plans;
- checkpoint or resumability frameworks;
- custom caching layers;
- temporary-file workflows;
- generalized pipelines;
- plugin systems;
- state machines;
- provenance systems;
- recovery systems for hypothetical failures;
- validation frameworks around simple scripts;
- abstractions intended only for possible future requirements.

If a failed build can reasonably be handled by deleting the partial output and running the script again, that may be sufficient.

Do not confuse more machinery with better engineering.

## 4. Features Versus Optimizations

Unnecessary features should be removed. Useful optimizations should be kept.

Performance-sensitive code may be more complex when that complexity has a clear runtime or resource benefit. Batching, multiprocessing, threaded reads, vectorized operations, and avoiding repeated expensive work can all be appropriate.

Do not remove a meaningful optimization merely to make code visually smaller.

Likewise, do not disguise a feature as an optimization. Atomic-output workflows, resumability, manifests, provenance tracking, generalized recovery logic, and extra metadata are features unless TitusAI actually needs them.

Start from a simple correct implementation. Add optimizations deliberately, one at a time, without turning the surrounding code into a framework.

## 5. Change and Cleanup Discipline

Before changing code:

- read the real implementation;
- identify the actual requirement;
- preserve behavior that is not part of the requested change;
- preserve performance-sensitive behavior unless performance is intentionally being changed;
- inspect direct callers before changing a function contract;
- avoid speculative refactoring;
- review the final diff for accidental behavior changes.

Style cleanup may simplify unnecessarily fragmented control flow, remove genuinely dead code, normalize repeated data, rename unclear variables, or reduce needless abstraction.

Cleanup is not permission to change model architecture, training behavior, dataset composition, or unrelated files.

## 6. General Python Preferences

Unless there is a strong reason otherwise:

- Do not use Python type hints.
- Avoid nested functions.
- Avoid recursion unless it is genuinely the clearest solution.
- Avoid broad `try`/`except` blocks.
- Avoid unnecessary classes.
- Avoid excessive helper functions.
- Avoid defensive coding for hypothetical problems.
- Avoid temporary files unless they are genuinely required.
- Avoid `argparse` and direct `sys.argv` handling unless the file genuinely needs a command-line interface.
- Keep imports at module scope unless runtime initialization requires otherwise.
- Prefer single quotes for normal Python strings.
- Avoid the walrus operator as normal project style.
- Prefer straightforward code over clever code.

These are strong defaults, not blind prohibitions.

## 7. Functions and Abstraction

Functions should exist because they make the program easier to understand.

Do not create a helper merely because an operation can be named. One function call does not need another function wrapped around it unless the wrapper adds meaning or removes substantial repetition.

Avoid fragmenting one simple pipeline into many helpers such as:

```python
get_source_files()
get_local_shard()
filter_batch()
tokenize_jobs()
write_documents()
```

when the reader then has to jump around the file to understand one linear operation.

At the same time, do not remove every function and replace the program with one enormous wall of top-level code.

A function earns its place when it does at least one of these things:

- gives a meaningful name to a coherent operation;
- removes substantial repeated control flow;
- isolates genuinely complicated logic;
- defines a real interface boundary;
- makes the main execution path easier to scan.

The question is not "Can this be a function?" The question is "Does making this a function reduce complexity for the reader?"

## 8. Normalize Repeated Static Information

Do not repeat large pieces of static text when only a small part changes.

Bad:

```python
FILES = [
    (FINEWEB_REPO, 'data/train-00042-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00041-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00091-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00009-of-00100.parquet'),
]
```

The meaningful information is the checkpoint ID. Store that instead:

```python
FINEWEB_CHECKPOINTS = [42, 41, 91, 9]
filename = f'data/train-{checkpoint:05d}-of-00100.parquet'
```

The data structure should contain the information that actually varies.

Do not confuse literal repetition with clarity. Repeating the same prefix, suffix, repository name, or other fixed value dozens of times makes configuration harder to scan and edit.

## 9. Use Data Structures for Fixed Relationships

If several branches merely map one known value to another known value, the relationship is data rather than control flow.

Bad:

```python
if repo_id == FINEWEB_REPO:
    target_tokens = FINEWEB_END
elif repo_id == COSMOPEDIA_REPO:
    target_tokens = COSMOPEDIA_END
else:
    target_tokens = TINYSTORIES_END
```

Better:

```python
TARGET_TOKENS = {
    FINEWEB_REPO: 299_000_000,
    COSMOPEDIA_REPO: 115_000_000,
    TINYSTORIES_REPO: 46_000_000,
}

target_tokens = TARGET_TOKENS[repo_id]
```

If several related configuration values always travel together, a single well-shaped structure may be clearer still:

```python
DATASETS = {
    FINEWEB_REPO: (299_000_000, 'data/train-{:05d}-of-00100.parquet', [42, 41, 91, 9]),
    COSMOPEDIA_REPO: (115_000_000, 'cosmopedia-v2/train-{:05d}-of-00104.parquet', [84, 0, 80, 45]),
}
```

Do not create separate constants such as `FINEWEB_END` or `COSMOPEDIA_END` when they are only implementation artifacts. Prefer the real domain value: `115_000_000` is a Cosmopedia quota; `414_000_000` is merely a cumulative position created by one implementation.

Do not treat "one list" or "one dictionary" as a goal. Use the representation that makes the relationships easiest to understand.

## 10. Control Flow

Prefer direct control flow that can be followed from top to bottom.

Use `if` statements for actual behavior: filtering rows, handling genuinely different execution paths, or avoiding unnecessary work.

Do not use `if`/`elif` chains as verbose lookup tables when a dictionary expresses the relationship directly.

If a condition becomes difficult to read, prepare meaningful boolean values first rather than vertically exploding one large expression.

Early `continue` and `return` statements are useful when they keep the main path shallow and obvious.

## 11. Comprehensions and Loops

Simple comprehensions are welcome:

```python
names = [user.name for user in users]
```

Prefer an explicit loop when a comprehension contains several conditions, multiple `for` clauses, nested structures, side effects, or enough formatting that the reader has to decode it.

Do not use a comprehension merely because it saves lines.

Repeated control flow should not automatically become a generic abstraction. Repeated information should usually be normalized. Keep that distinction clear.

## 12. Collections and Configuration

Short flat collections should normally stay compact:

```python
values = [1, 2, 3, 4, 5]
```

Long flat collections may wrap across lines, but do not put every trivial value on its own line without a readability reason.

Configuration collections should make values that developers are likely to edit easy to find and compare.

Use dictionaries when they naturally model a mapping or grouped configuration. Do not introduce a dictionary solely to avoid two simple statements, but do use one when it removes repetitive lookup control flow or keeps strongly related values together.

For substantial nested dictionaries, use indentation that makes the structure obvious. Very small dictionaries may stay on one line.

## 13. Naming and Variable Lifetime

Use `snake_case` for variables and functions and `UPPER_SNAKE_CASE` for constants.

Prefer descriptive names that make a value understandable without tracing surrounding code.

Avoid single-letter names for meaningful values. Conventional short loop indexes such as `i` are fine when their meaning is obvious and short-lived.

A variable should exist because it:

- gives a value a useful name;
- avoids repeated work;
- simplifies a complicated expression;
- is needed across multiple operations.

Avoid naming implementation artifacts as though they were domain concepts. Prefer a dataset's actual token quota over a cumulative `*_END` value when the latter exists only to support one loop implementation.

Do not keep values alive longer than needed, but do not recompute expensive values merely to avoid a local variable.

## 14. Strings and Docstrings

Use single quotes for normal Python strings.

Preferred:

```python
name = 'cat'
message = f'Found {count} records'
values = ['one', 'two', 'three']
```

Use double quotes only when they genuinely improve readability or Python syntax requires them.

Use triple single quotes for docstrings. Keep the entire docstring on one physical line with a short high-level purpose sentence.

Preferred:

```python
def add(num1, num2):
    ''' Add two numbers together. '''
    return num1 + num2
```

Do not use multiline docstrings to narrate implementation details. The code should explain the implementation.

## 15. Imports and Dependencies

Keep imports at module scope unless initialization order or another real runtime requirement requires otherwise.

Multiple ordinary modules may share one import line when that keeps the import block compact and readable.

Do not use wildcard imports.

Prefer an existing dependency's smallest appropriate API over rebuilding the same machinery in TitusAI.

For example, downloading one known Hugging Face file does not need a custom downloader:

```python
path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type='dataset')
```

Likewise, if `load_dataset('parquet', ...)` cleanly hides Parquet implementation details, TitusAI does not need to expose lower-level Arrow table machinery unless that lower-level API provides a real performance benefit we have chosen to keep.

Do not add dependencies merely to make a trivial operation look more abstract.

## 16. Layout and Formatting

Use vertical space for meaningful structure, not because a formatter prefers every argument on a separate line.

Use roughly 140 characters as a visual line-length guideline, not a hard limit. Keep logically indivisible values such as URLs, model identifiers, or simple calls intact when splitting them would be harder to read.

Keep function signatures on one physical line as a strong default.

Keep simple function calls compact:

```python
result = some_function(first_value, second_value, third_value)
```

Do not put every argument on its own line merely because a call has several arguments.

Use one blank line between top-level code structures and logical phases. Avoid two consecutive blank lines as normal project style.

Automated formatters and linters are review tools, not the authority on readability. Do not make working code worse merely to satisfy a mechanical preference.

## 17. Error Handling and Defensive Code

Keep error handling proportional to real failure modes.

Avoid broad `try`/`except` blocks unless the code is intentionally handling a genuine boundary where arbitrary failures must be captured.

Do not add retries, fallback paths, extra validation layers, or recovery logic for hypothetical problems that have not been observed or required.

Do not add explicit `raise` statements simply to make a script look defensive. Add validation when an invalid state would otherwise produce confusing or dangerous behavior.

Simple scripts are allowed to fail normally and expose the underlying error.

## 18. Dead Code and Comments

Remove genuinely dead code after checking that the expression has no useful side effect.

Do not keep unused variables, parameters, helpers, or configuration merely because they might be useful later.

Comments should add information that cannot be read directly from the code.

Useful comments explain:

- a non-obvious external constraint;
- a subtle performance decision;
- an intentional workaround;
- why an apparently strange operation must remain.

Avoid comments that simply narrate the next statement.

## 19. Performance-Sensitive Code

Performance is a real requirement for training and data preparation. Simplicity does not mean deliberately slow code.

Before adding complexity for performance, identify what work is expensive and whether the optimization addresses it directly.

Justified complexity may include:

- batch tokenization;
- multiprocessing;
- threaded reads;
- vectorized filtering;
- avoiding repeated model or tokenizer loading;
- reducing unnecessary copies of large tensors or datasets;
- using compact numeric formats when appropriate.

Keep the optimization local and obvious. Do not let one optimization grow into a generalized execution framework.

When a performance change is material and measurable, benchmark it rather than assuming a more complicated implementation is faster.

## 20. Data Preparation Code

Data scripts should make the real transformation visible:

```text
known source files
        ↓
download/read
        ↓
filter
        ↓
tokenize
        ↓
write training data
```

Do not bury this flow under generic source classes, manifests, merge stages, temporary parts, or metadata systems unless a real requirement demands them.

Prefer an output format that the next stage can consume directly. Do not add an intermediate text, JSON, or other representation merely because it is easier to inspect if training ultimately needs token IDs.

Store only the source configuration that actually varies. Use simple data structures to describe fixed relationships between repositories, checkpoint IDs, quotas, and filename patterns.

Quality filtering rules are part of the corpus design and should be visible enough for a learner to understand what data is being selected.

## 21. Model, Training, and Inference Code

The core ML path should remain easy to trace.

A learner should be able to find and understand:

- model dimensions and layers;
- attention and feed-forward computation;
- normalization and positional encoding;
- loss computation;
- optimizer and learning-rate behavior;
- batching and context length;
- checkpoint loading/saving where genuinely needed for training;
- token sampling during inference.

Do not hide educationally important ML logic behind generalized configuration frameworks or excessive wrappers.

Performance-specific PyTorch code may be compact or specialized when it materially improves training speed or memory use, but keep the reason for the complexity clear.

Architecture and hyperparameter changes must be treated as intentional project decisions, not incidental cleanup.

## 22. Educational Code

TitusAI code should teach the model or ML concept, not the infrastructure surrounding it.

Prefer code where a learner can identify what data is used, how it is filtered, how it becomes tokens, how the model transforms those tokens, how training updates the model, and how inference produces output.

Avoid architecture whose main lesson is how to maintain the architecture itself.

Do not remove useful names or structure merely to reduce line count. Clear repetition can be better than premature abstraction, while normalized repeated data can be better than literal duplication. Judge both by cognitive load.

## 23. Review Checklist

Before finalizing TitusAI code, check:

- Is the code correct for the task actually requested?
- Is the main execution path easy to follow?
- Did I add any feature that was not requested or required?
- Did I preserve useful optimizations?
- Is repeated static information normalized?
- Is fixed lookup data represented as data rather than repetitive branching?
- Does each function reduce complexity rather than merely relocate it?
- Are collections shaped around the information that actually varies?
- Are names describing real domain concepts rather than implementation artifacts?
- Did I avoid speculative defensive code and unnecessary abstraction?
- Did I keep model architecture, hyperparameters, and corpus decisions unchanged unless explicitly approved?
- Did I avoid unrelated refactoring?
- Does the final diff remain useful as a teaching resource?

## 24. Final Principle

TitusAI code should be unsurprising.

Use abstraction when it removes real complexity.

Use data structures when they make relationships clearer.

Do not repeat static information when only a small value changes.

Do not replace useful structure with walls of code in the name of simplicity.

Do not add features the project did not ask for.

Keep optimizations that earn their complexity.

When in doubt, prefer the smallest clear implementation that directly expresses the requirement.
