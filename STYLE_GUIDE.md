# TitusAI Style and Code Quality Guide

## Purpose

This document defines the preferred coding style and cleanup principles for TitusAI.

TitusAI is an educational project. Code should be easy to scan, easy to reason about, and useful as a teaching resource. The project should not accumulate production-style infrastructure, abstractions, or defensive machinery unless a concrete requirement justifies them.

The goal is not to minimize line count at all costs. The goal is to minimize cognitive load while keeping the code correct and reasonably efficient.

## 1. Core Principle

Prefer simple, direct, readable code that makes its intent obvious without requiring the reader to mentally decode clever syntax or excessive abstraction.

Readability takes priority over fashionable Python idioms, formatter defaults, or architectural patterns that do not solve a real problem.

When deciding whether code is simpler, ask:

1. Does this remove information the reader does not need?
2. Does this avoid repeating static information?
3. Does this make the data flow easier to follow?
4. Does this abstraction remove real complexity, or merely move it elsewhere?
5. Is this code solving a requirement we actually have?

A shorter file is not automatically simpler. A file with fewer functions is not automatically simpler. A file with fewer data structures is not automatically simpler.

## 2. Do Not Over-Engineer

Over-engineering is outside the scope of TitusAI.

Do not add infrastructure merely because it would be useful in a larger production system. In particular, avoid adding:

- manifests;
- metadata files;
- build plans;
- checkpoint or resumability frameworks;
- custom caching layers;
- temporary-file workflows;
- generalized pipelines;
- plugin systems;
- state machines;
- extra validation frameworks;
- recovery systems for hypothetical failures;
- abstractions intended only for possible future requirements.

If a script can be restarted by deleting its output and running it again, that may be entirely sufficient.

Keep the plumbing boring.

## 3. Features Versus Optimizations

Unnecessary features should be removed. Useful optimizations should be kept.

Performance-sensitive code is allowed to be more complex when the complexity produces a real benefit. Examples include batching, multiprocessing, threaded I/O, vectorized operations, or avoiding repeated expensive work.

Do not remove a meaningful optimization merely to make code visually smaller.

Likewise, do not disguise a feature as an optimization. Atomic output files, resumability, manifests, provenance tracking, and generalized recovery logic are features unless the project has a concrete need for them.

Start with the simplest correct implementation. Add optimizations one at a time when their benefit is clear.

## 4. General Python Preferences

Unless there is a strong reason otherwise:

- Do not use Python type hints.
- Avoid nested functions.
- Avoid recursion unless it is genuinely the clearest solution.
- Avoid broad `try`/`except` blocks.
- Avoid unnecessary classes.
- Avoid excessive helper functions.
- Avoid defensive coding for hypothetical problems.
- Avoid temporary files unless they are genuinely required.
- Avoid `argparse` and direct `sys.argv` handling unless the file is genuinely a CLI program.
- Keep imports at module scope unless runtime initialization requires otherwise.
- Prefer single quotes for normal Python strings.
- Prefer straightforward code over clever code.

These are strong defaults, not blind prohibitions.

## 5. Functions and Abstraction

Functions should exist because they make the program easier to understand, not because every operation can technically be given a name.

Avoid splitting one linear operation into many tiny helpers such as:

```python
get_source_files()
get_local_shard()
filter_batch()
tokenize_jobs()
write_documents()
```

when the result forces the reader to jump around the file to understand one simple pipeline.

At the same time, do not respond to this rule by removing every function and creating one enormous wall of top-level code.

A useful function should normally do at least one of these things:

- give a meaningful name to a coherent operation;
- remove substantial repetition;
- isolate genuinely complicated logic;
- define a real interface boundary;
- make the main execution path easier to scan.

The correct question is not "Can this be a function?" It is "Does making this a function reduce complexity for the reader?"

## 6. Normalize Repeated Static Data

Do not repeat large pieces of static text when only a small part changes.

For example, this is noisy:

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
```

Then construct the filename where it is needed:

```python
filename = f'data/train-{checkpoint:05d}-of-00100.parquet'
```

This is simpler because the data structure contains only the values that actually vary.

Do not confuse literal repetition with clarity. Repeating the same long prefix and suffix dozens of times makes configuration harder to scan and harder to edit.

## 7. Use Data Structures for Fixed Lookups

If several branches merely map one known value to another known value, the relationship is data rather than control flow.

Avoid code like:

```python
if repo_id == FINEWEB_REPO:
    target_tokens = FINEWEB_END
elif repo_id == COSMOPEDIA_REPO:
    target_tokens = COSMOPEDIA_END
else:
    target_tokens = TINYSTORIES_END
```

when a direct lookup expresses the relationship more clearly:

```python
TARGET_TOKENS = {
    FINEWEB_REPO: 299_000_000,
    COSMOPEDIA_REPO: 115_000_000,
    TINYSTORIES_REPO: 46_000_000,
}

target_tokens = TARGET_TOKENS[repo_id]
```

Do not create separate constants such as `FINEWEB_END`, `COSMOPEDIA_END`, and `TINYSTORIES_END` when those values only exist to feed an immediately adjacent lookup structure.

Prefer storing the real domain value. For example, `115_000_000` is the Cosmopedia quota. A cumulative value such as `414_000_000` is an implementation artifact and should not be presented as though it were a property of Cosmopedia.

## 8. Use Data Structures When They Reduce Cognitive Load

Do not treat "one list" or "one dictionary" as a goal in itself.

Several small collections may be clearer than one large nested structure when they represent distinct concepts. Conversely, one well-shaped dictionary may be clearer than three constants plus an `if`/`elif` chain.

Choose the representation that lets a reader answer these questions quickly:

- What values exist?
- Which values belong together?
- What actually varies?
- How is each value used?

Data structures should model the information, not satisfy an arbitrary preference for fewer variables.

## 9. Avoid Repetition, but Do Not Abstract Prematurely

Some repetition is clearer than a generic framework.

Do not create a callback system, source class, configurable pipeline, or helper hierarchy merely because two blocks share a few lines.

However, repeated static data and repeated lookup logic should normally be normalized because doing so directly reduces noise.

A useful distinction is:

- repeated *information* is often worth normalizing;
- repeated *control flow* may or may not be worth abstracting.

Judge each case by readability.

## 10. Keep the Main Data Flow Visible

For scripts, the reader should be able to understand the main flow from top to bottom.

For example, a corpus builder should visibly resemble:

```text
checkpoint configuration
        ↓
download file
        ↓
read dataset
        ↓
filter rows
        ↓
tokenize text
        ↓
write output
```

Do not bury this flow under multiple layers of generic orchestration.

Dependencies may hide implementation details that are not important to TitusAI. If `load_dataset('parquet', ...)` can hide Parquet internals cleanly, TitusAI does not need to expose lower-level Arrow table construction merely because the dependency uses Arrow internally.

## 11. Prefer the Smallest Appropriate API

When a dependency already provides the operation we need, use it directly.

For example, downloading one known Hugging Face file is simply:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id='HuggingFaceFW/fineweb_edu_100BT-shuffled',
    filename='data/train-00042-of-00100.parquet',
    repo_type='dataset',
)
```

Do not wrap a one-line dependency call in a helper unless the wrapper adds real meaning or removes meaningful repetition.

Likewise, do not build custom download, cache, retry, or path-management systems when the dependency already handles them adequately for the project.

## 12. Comprehensions

Simple comprehensions are welcome:

```python
names = [user.name for user in users]
```

Avoid comprehensions that combine several kinds of logic, especially when they require vertical formatting or multiple conditions.

Prefer an explicit loop when it is easier to scan.

Do not use a comprehension merely because it is shorter.

## 13. Control Flow

Prefer direct control flow that can be understood from top to bottom.

Avoid large conditions that require the reader to decode several unrelated checks at once.

Do not use an `if`/`elif`/`else` chain for a fixed lookup when a dictionary expresses the same relationship directly.

Do use normal `if` statements when they describe actual behavior, such as filtering rows or choosing between genuinely different execution paths.

The goal is not to eliminate branching. The goal is to avoid using branching as a verbose substitute for data.

## 14. Collections

Short flat collections should normally stay compact:

```python
values = [1, 2, 3, 4, 5]
```

Long collections may wrap across lines, but do not put every trivial value on its own line without a readability reason.

Configuration collections should make the values developers are likely to edit easy to find and compare.

If a collection consists mostly of repeated text with one changing field, reconsider the representation and store the changing field directly.

## 15. Dictionaries

Use dictionaries when they naturally model a mapping or grouped configuration.

Do not introduce a dictionary solely to avoid writing two simple statements. Do introduce one when it removes repetitive lookup control flow or keeps related values together.

For substantial dictionaries, structure the indentation so relationships are obvious.

Very small dictionaries may stay on one line:

```python
data = {'name': 'cat', 'age': 4}
```

## 16. Naming

Use `snake_case` for variables and functions and `UPPER_SNAKE_CASE` for constants.

Prefer descriptive names that make values understandable without tracing surrounding code.

Avoid naming implementation artifacts as though they were domain concepts. For example, prefer a dataset's actual token quota over a cumulative `*_END` value when the latter exists only to support one particular loop implementation.

## 17. Strings

Use single quotes for normal Python strings.

Preferred:

```python
name = 'cat'
message = f'Found {count} records'
values = ['one', 'two', 'three']
```

Use double quotes only when they genuinely improve readability or are required by Python syntax.

## 18. Docstrings

Use triple single quotes.

Keep docstrings on one physical line and describe only the high-level purpose.

Preferred:

```python
def add(num1, num2):
    ''' Add two numbers together. '''
    return num1 + num2
```

Do not use multiline docstrings merely to document obvious implementation details.

## 19. Imports

Keep imports at module scope unless runtime initialization requires another order.

Compact import blocks are preferred when they remain readable.

Do not use wildcard imports.

Do not add dependencies when the standard library or an existing dependency already provides a simple solution.

## 20. Function Signatures and Calls

Keep function signatures on one line as a strong default.

Keep simple function calls compact rather than putting every argument on its own line.

Preferred:

```python
result = some_function(first_value, second_value, third_value)
```

Vertical formatting should communicate meaningful structure, not merely consume space.

## 21. Blank Lines

Use one blank line to separate logical phases and code structures.

Avoid excessive vertical spacing.

Blank lines should tell the reader that one coherent operation has ended and another has begun.

## 22. Comments

Comments should explain information that cannot be read directly from the code.

Avoid narrating obvious statements.

Comments are useful for:

- non-obvious external constraints;
- subtle performance decisions;
- intentional workarounds;
- reasons an apparently strange operation must remain.

Keep comments concise.

## 23. Performance-Sensitive Code

Do not rewrite a measured or clearly important hot path merely because a slower version is aesthetically simpler.

When adding an optimization, keep the surrounding design as simple as practical and make the optimization's purpose obvious.

Examples of justified complexity may include:

- batch tokenization;
- multiprocessing;
- threaded reads;
- vectorized filtering;
- avoiding repeated model or tokenizer loading.

Optimizations should be added because they improve runtime or resource use, not because they appear sophisticated.

## 24. Educational Code

TitusAI is a learning resource. Code should teach the model or ML concept, not distract the reader with infrastructure.

Prefer code where a learner can identify:

- what data is being used;
- how it is filtered;
- how it is tokenized;
- how the model is built;
- how training works;
- how inference works.

Avoid architecture whose main lesson is how to maintain the architecture itself.

## 25. Cleanup Discipline

When cleaning up existing code:

1. Read the real implementation.
2. Identify the actual requirements.
3. Remove features that do not serve those requirements.
4. Preserve meaningful optimizations unless intentionally changing performance.
5. Normalize repeated static information.
6. Replace fixed lookup control flow with data structures when clearer.
7. Remove abstraction that merely relocates complexity.
8. Do not rewrite unrelated code.
9. Review the final diff for accidental behavior changes.

Cleanup is not permission to redesign the entire repository.

## 26. Final Principle

TitusAI code should be unsurprising.

Use abstraction when it removes real complexity.

Use data structures when they make relationships clearer.

Do not repeat static information when only one small value changes.

Do not replace useful structure with walls of code in the name of simplicity.

Do not add features the project did not ask for.

Keep optimizations that earn their complexity.

When in doubt, prefer the smallest clear implementation that directly expresses the requirement.
