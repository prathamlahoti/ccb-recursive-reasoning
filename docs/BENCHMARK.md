# Benchmark Specification

## Provenance

The source of truth for original CCB behavior is the official repository:

- https://github.com/Shubh-Chapra/Complexity_Ceiling_Benchmark
- Pinned commit: `e1e953a7a33a8c2406f936a54f4900f740dea8dd`
- Paper: https://arxiv.org/abs/2606.29278

The three fixed JSON files in `data/ccb_official` are copied from that commit.
Their hashes, upstream paths, and licenses are recorded in the adjacent
`SOURCE.md` and `LICENSE` files.

`ccb_official_v1` means exact compatibility with those generators and records.
`ccb_learn_v1` means a generated learning extension using official operations
and transition semantics. Extension results must never be presented as original
CCB evaluation results.

## Official fixed evaluation set

Each domain contains 400 examples: 40 examples at each depth in
`{5,10,15,20,25,30,35,40,45,50}`. Every record contains the depth, deterministic
seed, operation/event prompt, full ground-truth trace, and final answer.

The fixed 1,200 examples are evaluation data. They must not be used for model
training, early stopping, architecture selection, or hyperparameter search.

## D1: Alien Grid

Initial state is always:

```text
[[1,2,3],[4,5,6],[7,8,9]]
```

The official operation vocabulary, in generator sampling order, is:

1. `ROTATE_90_CW`
2. `SHIFT_ROW_2_LEFT`
3. `SWAP_CORNERS`
4. `REVERSE_GRID`
5. `FLIP_HORIZONTAL`
6. `TRANSPOSE_GRID`
7. `SHIFT_COL_1_UP`

The state must remain a permutation of 1 through 9. A trace contains the full
grid after each operation and excludes the initial grid.

## D2: Symbolic Pointers

Initial state is always `[A=1, B=2, C=3, D=4, E=5, F=6, G=7]`. The official
operation vocabulary, in sampling order, is:

1. `SHIFT_RIGHT`
2. `SHIFT_LEFT`
3. `SWAP_A_G`
4. `SET_D_TO_A_PLUS_B`
5. `SET_C_TO_G_MINUS_E`

Arithmetic is modulo 10. Unlike the earlier paper-only inference, D2 does not
require register values to remain distinct: SET operations may create
duplicates. Each trace state lists all seven registers.

## D3: Social Logic

There are ten people A-J, initially in separate neutral components. Each step
samples an unordered pair. If they are disconnected, the generator samples an
alliance or rivalry and merges their signed components. If already connected,
it emits the relationship implied by their existing factions. Closure uses:

- friend × friend = friend;
- rival × rival = friend;
- friend × rival = rival;
- rival × friend = rival.

Trace states list every non-neutral unordered pair in alphabetical order using
`AB:F` or `AB:R` notation.

## Exact compatibility gate

`python -m ccb verify-official` verifies each imported file's SHA-256 hash,
regenerates every record from its depth and seed, and requires exact equality
of prompt, every trace string, and final answer. The current result is:

| Domain | Records | Exact matches |
|---|---:|---:|
| D1 | 400 | 400 |
| D2 | 400 | 400 |
| D3 | 400 | 400 |

Any mismatch is a release blocker.

## Learning extensions

The definitive generated suite uses official transition semantics:

| Split | Depths | Examples/depth | Purpose |
|---|---|---:|---|
| train | 1-20 | 100 | parameter fitting |
| validation | 1-20 | 25 | in-distribution model selection |
| test_depth | 25, 30, 35, 40, 45, 50 | 100 | depth extrapolation |
| test_strong | 60, 80, 100 | 100 | strong extrapolation |

Every candidate is passed through an official-evaluation firewall. A candidate
is rejected if its `(depth, seed)`, canonical program hash, or canonical full
instance hash occurs in the 1,200 official records. The manifest stores the
three overlap counts for every split and generation fails unless all are zero.

D1 and D2 use fixed official initial states. Their shallow program spaces are
therefore very small: D1 has only seven distinct depth-1 programs and D2 only
five. Repeated programs within training and program overlap between shallow
train/validation cells are mathematically unavoidable at the chosen sample
counts. These collisions are measured, not hidden. The extrapolation suites
use non-overlapping depths, and official-test overlap is always forbidden. D3
has a much larger event and query space and the current definitive splits are
instance-disjoint.

Structural extensions are separately versioned:

| Domain | Training restriction | Test requirement |
|---|---|---|
| D1 | Excludes ordered pair `ROTATE_90_CW` → `SHIFT_ROW_2_LEFT` | Requires that pair |
| D2 | Excludes `SET_D_TO_A_PLUS_B` → `SET_C_TO_G_MINUS_E` | Requires that pair |
| D3 | Six/eight agents with chain topology | Ten/twelve agents with star topology |

D3 structural testing combines topology and graph-size shift and must be
reported as such.

### Repaired D1 semantic structural suite

The original D1 structural row above is retained only as a token-level
diagnostic. It is not a semantic holdout: for example,
`TRANSPOSE_GRID -> FLIP_HORIZONTAL` implements `ROTATE_90_CW`.

Use the repaired suite for method claims:

```text
ccb generate-structural --domain d1 --semantic-d1 --output <directory>
```

It uses randomized valid initial grids, protects the official fixed records,
and excludes from train/validation every contiguous program segment whose net
grid permutation equals the held composition
`ROTATE_90_CW -> SHIFT_ROW_2_LEFT`. It also generates test first and rejects
any train/validation example reusing a concrete
`(before state, operation, after state)` transition from a protected test
split. The manifest records semantic exposure and all cross-split transition
overlap counts; both must be zero for train/validation exposure and for every
cross-split transition overlap.

## Metrics and audits

Implemented metrics include final exact accuracy, full-trace exact accuracy,
transition accuracy, state-element accuracy, validity, first divergence
`k*`, TFBC both overall and conditional on a correct final answer, per-step
retention, success horizons, normalized depth AUC, exact
Clopper-Pearson intervals, Wilson intervals, and bootstrap retention intervals.
Recursive models also report final exact accuracy after every loop.

Required audits include seed/content hashes, cross-split instance and program
overlap, operation n-grams, endpoint entropy, identity rate, D1 reachability,
and explicit structural-feature exposure counts.
