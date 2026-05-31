# D2E-FDM-1 Reproduction PoC Spec

# Goal

D2E에서 다루는 모든 게임에 대해 FDM-1 style approach가 재현 가능한지 확인한다.

이 PoC의 목표는 단순히 한 번의 end-to-end run을 성공시키는 것이 아니다.
목표는 **Video Encoder / IDM / FDM 각 training stage별로 현실적인 candidate들을 ablate하고**, 그 결과를 통해 아래 research questions에 답하는 것이다.

## Research Questions

1. D2E의 heterogeneous game distribution 전체에서 video encoder / IDM / FDM pipeline을 end-to-end로 구성할 수 있는가?

2. IDM이 2D, 3D, FPS, open-world, sandbox, top-down, side-scroller, UI/menu-heavy gameplay 전반에 대해 action pseudo-label을 생성할 수 있는가?

3. IDM pseudo-label로 학습한 FDM이 GT action label로 학습한 FDM에 얼마나 근접하는가?

4. FDM이 특정 게임에 overfit하지 않고 held-out game에서도 non-trivial action prediction을 수행하는가?

5. FDM-1식 video-action pretraining recipe가 D2E 전체 distribution에서 scale trend를 보이는가?

---

# High-level Pipeline

```text
D2E recordings across all games
  ├── video frames
  ├── audio, optional
  ├── keyboard events
  ├── mouse raw deltas
  ├── mouse clicks / coordinates
  ├── button states
  └── active window info

Step 1. Video Encoder candidates
  D2E video
  → V-JEPA 2 based video encoder variants
  → compressed video tokens

Step 2. IDM candidates
  compressed video tokens + masked action slots
  → recover keyboard / mouse / click action labels

Step 3. Pseudo-labeling
  video-only input
  → IDM-predicted action tokens / MCAP-style events

Step 4. FDM candidates
  past video tokens + past action tokens
  → next action token / next action bin prediction

Step 5. Offline Evaluation
  D2E-style action metrics
  GT-label FDM vs pseudo-label FDM
  per-game / per-category / held-out-game generalization
  scale trend analysis
```

---

# Dataset

## Source

Use D2E-480p as the primary training dataset.

Reason:

* 480p is more practical for vision-action model training.
* D2E-Original can be reserved for high-resolution ablations.
* Dataset revision must be pinned because published D2E counts differ across project page, Original, and 480p releases.

Required metadata to log:

* dataset name
* dataset revision / commit hash
* number of games
* total hours
* per-game hours
* per-game action statistics
* train/val/test split manifest

---

# Scope

Use all D2E games, not only 3D games.

The model should be evaluated across the full game distribution, including:

* FPS / shooter games
* first-person and third-person open-world games
* sandbox / crafting / survival games
* top-down games
* side-scroller or platformer-like games
* 2D action games
* farming / life-sim games
* driving / vehicle games
* UI/menu-heavy segments
* spectator / inactive / non-gameplay-like segments if present

The reproduction target is:

```text
FDM-1-style generalist video-action model over the full D2E game distribution
```

---

# Dataset Splits

## Split A: Recording-level in-distribution split

Purpose:

* Measure whether video encoder / IDM / FDM can learn D2E action prediction when train and test share the same game distribution.

Recommended:

* train: 80%
* validation: 10%
* test: 10%

Split unit:

* recording-level split
* clips from the same recording must not appear in both train and test.

Requirement:

* preserve per-game distribution as much as possible.
* every game with enough data should appear in train/val/test.

Report:

* micro-average over all clips
* macro-average over games
* per-game metrics

Macro-average is important because large games should not dominate the reported result.

---

## Split B: Held-out-game split

Purpose:

* Test whether IDM/FDM can generalize to unseen games.

Recommended:

* hold out multiple games across different genres/control schemes.

Example held-out categories:

```text
1. one FPS / shooter game
2. one sandbox or open-world game
3. one top-down or 2D game
4. one UI/menu-heavy or slower-paced game
5. one low-resource game if available
```

Evaluation:

* train on all non-held-out games
* validate on held-out validation game(s)
* test on held-out test game(s)

Report:

* per-held-out-game metrics
* average held-out-game score
* degradation relative to in-distribution test

This split is the primary split for answering RQ4.

---

## Split C: Pseudo-label simulation split

Purpose:

* Mimic FDM-1’s “IDM labels unlabeled internet video, then FDM trains on pseudo-labels” recipe within D2E.

Procedure:

```text
1. D2E labeled subset A:
   use GT actions to train IDM

2. D2E pseudo-label subset B:
   hide GT actions
   use IDM to generate pseudo-labels

3. Train FDM variants:
   FDM-GT: trained on GT labels
   FDM-Pseudo: trained on IDM pseudo-labels
   FDM-FilteredPseudo: trained on confidence-filtered pseudo-labels
   FDM-Mix: trained on GT + pseudo-labels

4. Evaluate all variants on the same GT-labeled held-out test set
```

Main comparison:

```text
FDM-Pseudo vs FDM-GT
```

This split is the primary split for answering RQ3.

---

## Split D: Data scale split

Purpose:

* Test whether the FDM-1-style recipe shows scale trend on D2E.

Create training subsets by hours or recordings:

```text
1%
5%
10%
25%
50%
100%
```

For each scale, train/evaluate:

* IDM candidate, if compute allows
* FDM-GT
* FDM-Pseudo
* optionally FDM-Mix

Report:

* metric vs data scale
* per-game macro-average vs scale
* held-out-game performance vs scale
* pseudo-label quality vs scale

This split is the primary split for answering RQ5.

---

# Timebase and Sequence Construction

## Base timestep

Use non-overlapping 50ms bins.

Reason:

* D2E official-style IDM evaluation uses 50ms temporal bins.
* 50ms corresponds to 20Hz action prediction.
* It is tractable across all game types and aligns keyboard/mouse events with video.

## Video sampling

For each 50ms bin:

```text
video frames:
  sample 1 representative frame per bin
  default: center frame
```

Optional:

* use multiple frames per 50ms bin for video encoder adaptation.
* compress them into one bin-level token group.

Recommended PoC default:

```text
60fps raw video
→ 20fps sampled visual stream
→ one visual timestep per 50ms action bin
```

## Event aggregation

For each 50ms bin:

* aggregate raw mouse HID deltas:

  * sum `last_x`
  * sum `last_y`

* collect keyboard events:

  * key press
  * key release

* collect mouse button events:

  * left/right/middle down
  * left/right/middle up

* collect scroll events if available

* collect optional state:

  * keyboard state
  * mouse button state
  * cursor absolute position
  * active window / game process metadata

---

# Action Tokenization

## Design principle

D2E contains timestamped input events. Multiple events can occur inside one 50ms bin.

Use bin-level action serialization:

```text
VideoBin_t
  MOUSE_MOVE token
  sparse event token 1
  sparse event token 2
  ...
  sparse event token K
VideoBin_{t+1}
```

For IDM:

* action slots are replaced by `MASK_ACTION`.

For FDM:

* previous action tokens are provided causally.
* model predicts the next bin’s action tokens.

---

## Special tokens

```text
MASK_ACTION
NO_ACTION
PAD_ACTION
BOS_ACTION
EOS_ACTION
EVENT_OVERFLOW
```

Usage:

* `MASK_ACTION`: IDM masked diffusion input
* `NO_ACTION`: explicit no-op action/event
* `PAD_ACTION`: clip packing padding
* `EVENT_OVERFLOW`: more events occurred in a bin than K slots can store

---

## Keyboard tokens

Use virtual-key or physical-key codes from D2E.

```text
KEY_DOWN_<key>
KEY_UP_<key>
```

Recommended:

* physical/virtual key identity, not character identity
* represent modifiers explicitly:

  * `KEY_DOWN_SHIFT`
  * `KEY_DOWN_CTRL`
  * `KEY_DOWN_ALT`
  * `KEY_DOWN_SPACE`

Optional auxiliary target:

* key state multi-hot prediction

Reason:

* many games depend on held keys, not only press/release events.

---

## Mouse movement tokens

Use raw HID deltas as the main movement signal.

For each 50ms bin:

```text
dx = sum(raw_mouse.last_x)
dy = sum(raw_mouse.last_y)
```

Recommended default:

```text
MOUSE_MOVE_BIN_<xbin>_<ybin>
```

Bin design:

* signed exponential bins
* include zero bin
* fit bins on the training set
* use global bins across all games, not per-game bins

Recommended number of bins:

* 49 bins per axis
* 49 x 49 = 2401 compound movement tokens

Ablation:

```text
MOUSE_DX_BIN_<i>
MOUSE_DY_BIN_<j>
```

Use separate axis tokens if compound-token sparsity becomes problematic.

---

## Mouse button tokens

```text
MOUSE_LEFT_DOWN
MOUSE_LEFT_UP
MOUSE_RIGHT_DOWN
MOUSE_RIGHT_UP
MOUSE_MIDDLE_DOWN
MOUSE_MIDDLE_UP
```

Optional:

* mouse button state auxiliary head

---

## Scroll tokens

Use scroll tokens because some games and menus may use wheel input.

Recommended:

```text
SCROLL_UP
SCROLL_DOWN
SCROLL_LEFT
SCROLL_RIGHT
```

If scroll magnitude is important:

```text
SCROLL_DY_BIN_<k>
SCROLL_DX_BIN_<k>
```

---

## Cursor / click auxiliary target

Use an auxiliary next-click-position prediction head.

This is not necessarily part of the action token vocabulary.

```text
NEXT_CLICK_POSITION_BIN_<x>_<y>
NO_CLICK_WITHIN_H
```

Define target:

```text
For each bin t:
  find the next mouse button down event within horizon H.
  if found:
      target = screen coordinate of that click
  else:
      target = NO_CLICK_WITHIN_H
```

Recommended:

* H = 1.0s or 2.0s
* grid = 32 x 18 for 16:9 480p
* ablation = 64 x 36

Loss:

```text
L_total = L_action + λ_click * L_next_click_position
```

Apply click-position loss:

* on bins with mouse movement
* or bins where a click occurs within H
* otherwise use `NO_CLICK_WITHIN_H`

---

# Fixed Action Slots per Bin

Because multiple events can happen inside one 50ms bin, use fixed K action slots.

Recommended default:

```text
K = 8 sparse event slots per 50ms bin
```

Per-bin serialization:

```text
[MOUSE_MOVE_BIN]
[EVENT_SLOT_1]
[EVENT_SLOT_2]
...
[EVENT_SLOT_K]
```

Ordering:

1. mouse movement token first
2. discrete events sorted by timestamp
3. remaining slots filled with `NO_ACTION`
4. sequence padding uses `PAD_ACTION`

Overflow policy:

* preserve mouse button events
* preserve key down events
* preserve key up events if capacity remains
* otherwise emit `EVENT_OVERFLOW`
* log overflow rate globally and per game
* increase K if overflow rate exceeds 0.1%

Ablation:

* K = 4
* K = 8
* K = 16

---

# Video Encoder

## Role

Compress D2E video frames from all games into a shared visual token space suitable for IDM and FDM.

The encoder should handle:

* 3D scenes
* 2D scenes
* menus
* UI overlays
* text-heavy screens
* cursor/crosshair movement
* rapid camera motion
* low-motion or idle segments

## Input

```text
D2E video frames
sampled at 20fps or grouped from 60fps into 50ms bins
```

## Output

```text
compressed video tokens per 50ms bin
```

Recommended default:

```text
4 to 16 video tokens per 50ms bin
hidden size: 512 or 768
```

---

## Video Encoder backbone

Use pretrained V-JEPA 2 as the PoC video encoder backbone.

The main PoC should not train a custom video encoder from scratch.
Instead, it should evaluate whether V-JEPA 2 can be adapted into a useful FDM-1-style compressed video representation for D2E.

---

## Video Encoder candidates

### VE-0: Frozen V-JEPA 2 + linear/shallow probes

Purpose:

* establish whether pretrained V-JEPA 2 features already contain useful information for D2E action prediction.

Training:

* freeze V-JEPA 2
* train only shallow probes for:

  * mouse movement
  * keyboard events
  * mouse buttons
  * next-action prediction

Evaluation:

* frozen feature action probe
* per-game and per-category metrics

Use for:

* RQ1
* RQ2 baseline
* diagnosing whether V-JEPA 2 features preserve gameplay-relevant information

---

### VE-1: Frozen V-JEPA 2 + trainable temporal compressor

Recommended first main candidate.

Architecture:

```text
D2E frames
  → frozen V-JEPA 2
  → trainable temporal compressor / Perceiver resampler
  → fixed number of video tokens per 50ms bin
```

Training:

* freeze V-JEPA 2
* train resampler/compressor
* train IDM/FDM on compressed tokens

Token budget:

* 4 video tokens / 50ms bin default
* ablate 8 and 16 tokens / bin

Pros:

* stable
* efficient
* avoids destroying pretrained representation
* suitable for H200 x4 first-pass experiments

Use for:

* main IDM/FDM runs
* most ablations
* pseudo-label usefulness study

---

### VE-2: V-JEPA 2 last-block / adapter / LoRA finetuning

Architecture:

```text
D2E frames
  → V-JEPA 2 with LoRA/adapters or last-N-block finetuning
  → temporal compressor / resampler
  → compressed video tokens
```

Training:

* freeze lower V-JEPA 2 blocks
* finetune only:

  * LoRA/adapters
  * last N blocks
  * temporal compressor

Purpose:

* adapt V-JEPA 2 to D2E-specific visual features:

  * HUD
  * crosshair
  * cursor
  * UI text
  * menus
  * fast camera motion
  * small objects

Use for:

* testing whether D2E domain adaptation improves IDM/FDM
* comparing against VE-1

---

### VE-3: V-JEPA 2 domain adaptation with masked video objective

Architecture:

```text
D2E video
  → V-JEPA 2 backbone
  → masked latent prediction / JEPA-style adaptation
  → temporal compressor
```

Training objective:

```text
mask temporal spans and/or spatial regions
predict teacher latent features of masked frames
```

Recommended:

* use action labels only for probes or downstream training, not as the primary video encoder objective.
* primary objective should remain self-supervised video representation adaptation.

Purpose:

* test whether self-supervised D2E video adaptation improves downstream action modeling.

Use for:

* RQ5 scale trend
* video encoder ablation

---

### VE-4: End-to-end finetuned V-JEPA 2 with IDM/FDM

Optional, not default.

Architecture:

* V-JEPA 2 + temporal compressor + IDM/FDM are trained jointly.

Purpose:

* test upper-bound downstream performance.

Risks:

* expensive
* harder to attribute improvements
* may overfit to D2E actions
* may reduce interpretability of video encoder stage

Use only after VE-1 / VE-2 are stable.

---

## Video Encoder training objective

For VE-1:

* train compressor through downstream IDM/FDM losses.
* optionally add masked latent prediction loss.

For VE-2 / VE-3:

* use a combination of:

  * masked latent prediction
  * downstream IDM loss
  * downstream FDM loss, if doing joint finetune
  * optional reconstruction or contrastive regularization

Default recommendation:

```text
L_VE =
  L_masked_latent_prediction
```

for domain adaptation, followed by downstream IDM/FDM training.

For joint finetuning:

```text
L_total =
  L_downstream
  + λ_ve * L_masked_latent_prediction
```

---

## Video Encoder evaluation

Primary:

1. frozen V-JEPA 2 action probe
2. V-JEPA 2 + compressor IDM downstream performance
3. V-JEPA 2 + compressor FDM downstream performance
4. VE candidate comparison:

   * VE-0
   * VE-1
   * VE-2
   * VE-3
   * optional VE-4
5. compression / throughput
6. per-game and macro-game performance

Baselines:

* no-op / zero-mouse baseline
* previous-action repeat baseline
* action-history-only baseline
* frozen image encoder + shallow probe, optional
* raw-frame baseline, optional if compute allows

Success criteria:

* V-JEPA 2 features improve action prediction beyond action-only and no-op baselines.
* VE-1 or VE-2 improves IDM/FDM downstream performance over VE-0.
* gains appear across multiple game categories, not only FPS or one high-resource game.
* higher token budget or adaptation shows measurable improvement, unless the task is bottlenecked by labels/action ambiguity.

---

# IDM

## Role

Train an inverse dynamics model on labeled D2E recordings and use it to pseudo-label action labels from video-only gameplay.

## Input

```text
compressed video tokens from a non-causal observation window
+ masked action tokens for target bins
```

Recommended observation design:

```text
target action bin: [t, t+50ms)

condition on video frames around the target:
  past context: t - C_past to t
  future context: t to t + C_future
```

Use future context because inverse dynamics is non-causal.

Recommended future offset:

* τ = 100ms default
* ablate 0ms, 50ms, 100ms, 150ms, 200ms

## Sequence format

```text
VideoBin_{t-N}
ActionSlots_{t-N}
...
VideoBin_t
MASK_ACTION x slots
VideoBin_{t+1}
MASK_ACTION x slots
...
VideoBin_{t+M}
```

The IDM can attend bidirectionally to all video tokens and all action/mask tokens.

## Architecture

Recommended PoC architecture:

```text
video token embeddings
+ action token embeddings
+ temporal position embeddings
+ modality embeddings
→ bidirectional transformer
→ action-token classification heads for masked positions
```

Model sizes:

Tiny:

```text
d_model: 512
layers: 8
heads: 8
context: 10s to 30s
```

Base:

```text
d_model: 768
layers: 12
heads: 12
context: 30s to 120s
```

Start with Tiny for broad ablations, then promote the best candidates to Base.

---

## IDM candidates

### IDM-0: Majority / no-op baseline

Purpose:

* baseline for sparse event dominance.

Prediction:

* zero mouse movement or previous distribution
* no keyboard/mouse-button events

---

### IDM-1: Action-prior baseline

Purpose:

* measure how much can be predicted from game/action statistics without visual information.

Input:

* game-agnostic or per-game action prior
* optional previous action context
* no video

---

### IDM-2: Causal IDM

Purpose:

* baseline against non-causal inverse dynamics.

Input:

* past video only
* no future video context

---

### IDM-3: Non-causal IDM

Main IDM candidate.

Input:

* past + future video context
* masked action slots

Ablations:

* future offset τ = 0ms, 50ms, 100ms, 150ms, 200ms
* context length = 1s, 5s, 10s, 30s
* K action slots = 4, 8, 16
* compound mouse token vs separate dx/dy tokens
* VE candidate = VE-1 vs VE-2 vs VE-3

---

### IDM-4: Non-causal IDM + confidence calibration

Purpose:

* improve pseudo-label quality for FDM training.

Methods:

* temperature scaling
* confidence thresholding
* entropy filtering
* per-token-type thresholding
* sparse event filtering

Used for:

* generating D_PSEUDO_FILTERED

---

### IDM-5: Per-game specialist IDM, optional

Purpose:

* upper-bound per-game labelability.
* compare generalist model vs specialist models.

Use:

* only if compute allows.
* not the main reproduction target.

---

## IDM training objective

Masked denoising / masked diffusion over action slots.

Simplified PoC objective:

```text
randomly mask action slots
predict original action tokens
cross-entropy over masked positions
```

FDM-1-style iterative inference:

```text
1. initialize target action slots as MASK_ACTION
2. predict log probabilities for all masked positions
3. unmask top-k highest-confidence predictions
4. repeat for S steps
```

Recommended:

* S = 8 or 16 steps
* mask ratio sampled from 0.1 to 1.0
* include fully masked target windows

Loss:

```text
L_IDM =
  CE(action token)
  + λ_click * CE(next-click-position optional)
  + λ_state * BCE(key/button state optional)
```

---

## IDM evaluation

Use D2E-style 50ms-bin metrics as primary.

Primary:

* mouse movement Pearson correlation X/Y
* mouse movement scale ratio X/Y
* mouse button accuracy
* keyboard per-key accuracy

Additional:

* masked action NLL
* top-1 / top-k action accuracy
* per-action-type accuracy
* mouse dequantized L1/L2 error
* key event precision / recall / F1
* mouse button precision / recall / F1
* event edit distance per second
* calibration: confidence vs correctness, ECE
* overflow rate for fixed K slots

Report all metrics:

* micro-average over all data
* macro-average over games
* per-game
* per-category
* in-distribution
* held-out-game

Important comparisons:

```text
IDM-NonCausal vs IDM-Causal
IDM-NonCausal vs IDM-ActionPrior
IDM with VE-1 vs VE-2 vs VE-3
IDM with different τ
IDM generalist vs per-game specialist, optional
```

Success criteria:

* IDM-NonCausal should outperform IDM-Causal.
* IDM should outperform no-op and previous-action baselines across most games.
* IDM should produce useful pseudo-labels for multiple game categories.
* Sparse keyboard/button metrics must be reported separately from mouse metrics.

---

# Pseudo-labeling Pipeline

## Purpose

Mimic FDM-1’s pseudo-labeling stage within D2E.

Procedure:

```text
1. Train IDM candidates on labeled subset.
2. Select best IDM variants by validation metrics.
3. Hide labels for pseudo-label subset.
4. Run IDM inference over pseudo-label subset.
5. Write predicted actions as MCAP or packed action-token dataset.
6. Train FDM candidates on pseudo-labeled subset.
```

## Confidence filtering

For each predicted token:

* store probability
* store entropy
* store token type
* store game ID
* store timestamp

Filtering options:

* keep all pseudo-labels
* drop low-confidence sparse events
* replace low-confidence sparse event slots with NO_ACTION
* downweight low-confidence labels in FDM loss
* use per-token-type confidence thresholds

Recommended initial policy:

* keep all mouse movement predictions
* filter only sparse keyboard/mouse-button events below threshold
* tune threshold on validation set, not per test game

Create datasets:

```text
D_GT:
  GT action labels

D_PSEUDO_ALL:
  all IDM predictions

D_PSEUDO_FILTERED:
  confidence-filtered IDM predictions

D_MIX:
  GT + pseudo-labels
```

---

# FDM

## Role

Train a forward dynamics model that predicts the next action given prior video and prior actions across all D2E games.

## Input

```text
past compressed video tokens
+ past action tokens
```

Optional:

* game ID embedding
* active window embedding
* current key/button state tokens

Default PoC:

* do not use audio
* do not use game ID in the main generalization experiment
* allow game ID only as an ablation

Reason:

* game ID can improve in-distribution score but may obscure whether a shared cross-game policy/action model is being learned.

## Output

Main:

* next 50ms bin action sequence

Optional:

* next-click-position auxiliary target
* key/button state auxiliary target
* action chunk over next H bins

## Architecture

Recommended PoC architecture:

```text
video tokens
+ action tokens
+ modality embeddings
+ temporal position embeddings
→ causal transformer
→ next-action heads
```

Model sizes:

Tiny FDM:

```text
d_model: 512
layers: 8
heads: 8
context: 10s to 30s
```

Base FDM:

```text
d_model: 768
layers: 12
heads: 12
context: 30s to 120s
```

Use Tiny for broad ablations.
Use Base only for final selected configurations.

## Context length

Initial:

```text
10s context = 200 action bins at 50ms
```

Ablations:

```text
2s
5s
10s
30s
60s
```

Purpose:

* measure whether longer video/action context improves action prediction across different game types.

---

## FDM candidates

### FDM-0: No-op / zero-mouse baseline

Purpose:

* establish lower bound.

---

### FDM-1: Previous-action repeat baseline

Purpose:

* test action inertia baseline.

---

### FDM-2: ActionOnly transformer

Input:

* past action tokens only

Purpose:

* measure how much performance comes from action autocorrelation.

---

### FDM-3: VideoOnly transformer

Input:

* past video tokens only

Purpose:

* measure whether visual state alone is predictive.

---

### FDM-4: FDM-GT

Input:

* past video tokens + past GT actions

Train labels:

* GT actions

Purpose:

* oracle upper bound at D2E scale.

---

### FDM-5: FDM-Pseudo

Input:

* past video tokens + past IDM-pseudo actions

Train labels:

* IDM pseudo-labels

Purpose:

* FDM-1-style reproduction.

---

### FDM-6: FDM-FilteredPseudo

Input:

* video + confidence-filtered pseudo-actions

Purpose:

* test whether pseudo-label filtering improves downstream FDM.

---

### FDM-7: FDM-Mix

Input:

* video + action tokens

Train labels:

* mixture of GT and pseudo-labels

Purpose:

* test whether limited GT data stabilizes pseudo-label training.

---

### FDM-8: FDM-GT with game ID, ablation only

Input:

* video tokens
* action tokens
* game ID embedding

Purpose:

* estimate upper bound with game conditioning.

Caution:

* not the main result, because game ID can obscure generalist cross-game learning.

---

### FDM-9: FDM-Pseudo with game ID, ablation only

Purpose:

* test whether game conditioning helps pseudo-label-trained FDM.

---

## FDM training objective

Teacher-forced autoregressive next-action prediction.

If using fixed K slots:

```text
L_FDM =
  CE(mouse movement token)
  + mean_k CE(event slot k)
  + λ_click * CE(next click position)
  + λ_state * BCE(next key/button state)
```

Class imbalance:

* no-op and zero movement dominate.
* use class-balanced reporting.
* optionally use loss weighting or balanced sampling.

Recommended:

* report both weighted and unweighted metrics.
* avoid aggressive rare-event weighting early, because it may increase false-positive actions.

---

# Evaluation

## Online rollout policy

Online rollout is out of scope for this PoC.

Reason:

* D2E is a logged dataset, not an interactive environment.
* The planned hardware target is an H200 GPU training cluster, not a game/VM rollout cluster.
* True online control would require reproducible game environments, reset logic, game state instrumentation, and latency-sensitive inference infra.
* This PoC is about validating FDM-1-style video-action pretraining under D2E, not demonstrating game-solving behavior.

Therefore, Phase 1 evaluation must be entirely offline.

No online rollout metric is required for success.

---

## Evaluation modes

### Eval A: Offline IDM action reconstruction

Purpose:

* answer whether IDM can pseudo-label D2E gameplay.

Input:

* video tokens
* masked action slots

Compare:

* IDM predicted actions vs GT D2E actions

Metrics:

* mouse movement Pearson correlation X/Y
* mouse movement scale ratio X/Y
* mouse button accuracy / F1
* keyboard per-key accuracy / F1
* masked action NLL
* top-k action accuracy
* calibration
* per-game macro-average
* held-out-game metrics

Primary RQs:

* RQ1
* RQ2

---

### Eval B: Pseudo-label usefulness

Purpose:

* answer whether IDM pseudo-labels are good enough to train FDM.

Train:

* FDM-GT on GT labels
* FDM-Pseudo on IDM pseudo-labels
* FDM-FilteredPseudo on confidence-filtered pseudo-labels
* FDM-Mix on GT + pseudo-labels

Evaluate:

* same GT-labeled held-out test set

Main metrics:

* FDM-Pseudo / FDM-GT performance ratio
* FDM-FilteredPseudo improvement over FDM-Pseudo
* FDM-Mix gap to FDM-GT

Primary RQ:

* RQ3

Example reporting:

```text
FDM-GT mouse Pearson: 0.60
FDM-Pseudo mouse Pearson: 0.48
Pseudo/GT ratio: 80%
```

---

### Eval C: Offline teacher-forced FDM evaluation

Purpose:

* evaluate next-action prediction under logged human trajectories.

Input:

* logged video
* GT or pseudo previous actions, depending on model

Metrics:

* next-action NLL
* top-1 / top-k action accuracy
* mouse movement Pearson correlation X/Y
* mouse movement scale ratio X/Y
* mouse dequantized L1/L2 error
* keyboard per-key accuracy
* keyboard precision / recall / F1
* mouse button accuracy / F1
* click-position error
* no-op false positive rate
* no-op false negative rate
* action event edit distance
* long-context degradation curve

Report by:

* all clips micro-average
* game macro-average
* per-game
* per-category
* high-mouse-motion segments
* high-keyboard-activity segments
* menu/UI-heavy segments
* low-motion / idle segments
* held-out games

Primary RQs:

* RQ1
* RQ4
* RQ5

---

### Eval D: Free-running-on-logged-video

This is not online rollout.

Logged video remains fixed, but model-generated actions are fed back into the action-history input.

Procedure:

```text
for each held-out clip:
  condition on first N seconds of GT actions
  then repeatedly:
    input logged video frames
    input model’s own previous predicted actions
    predict next action
```

This does not test true closed-loop environment control, because video does not respond to the model’s actions.

But it tests:

* action-history compounding
* key-state drift
* no-op collapse
* mouse trajectory drift relative to logged human actions
* stability across different game types

Metrics:

* same action metrics as offline teacher-forced evaluation
* degradation over rollout horizon
* key/button state consistency
* mouse drift over time

Primary RQs:

* RQ4
* RQ5

---

### Eval E: Scale trend evaluation

Purpose:

* answer whether the FDM-1-style recipe scales on D2E.

Train on:

```text
1%
5%
10%
25%
50%
100%
```

Evaluate:

* IDM quality vs data scale
* FDM-GT vs data scale
* FDM-Pseudo vs data scale
* Pseudo/GT ratio vs data scale
* held-out-game performance vs data scale
* context length sensitivity vs data scale

Primary RQ:

* RQ5

---

# RQ-to-Experiment Matrix

## RQ1: Can the full pipeline be built over heterogeneous D2E?

Required experiments:

* VE-1 + IDM-3 + FDM-4
* VE-1 + IDM-3 + FDM-5
* full data pipeline
* packed action-token dataset
* pseudo-label generation pipeline

Success evidence:

* all models train without collapse
* action detokenization works
* MCAP/predicted action outputs are evaluable
* offline metrics beat no-op/action-only baselines

---

## RQ2: Can IDM pseudo-label diverse gameplay?

Required experiments:

* IDM-0 / IDM-1 / IDM-2 / IDM-3 comparison
* per-game and per-category IDM metrics
* VE-1 vs VE-2 vs VE-3 for IDM
* τ ablation
* K-slot ablation

Success evidence:

* IDM-NonCausal beats IDM-Causal
* IDM beats no-op/action-prior baseline
* pseudo-label quality is useful across multiple game types
* sparse keyboard/button performance is non-trivial

---

## RQ3: How close is pseudo-label FDM to GT-label FDM?

Required experiments:

* FDM-GT
* FDM-Pseudo
* FDM-FilteredPseudo
* FDM-Mix

Success evidence:

* FDM-Pseudo recovers meaningful fraction of FDM-GT
* confidence filtering improves or stabilizes FDM-Pseudo
* FDM-Mix closes the gap to FDM-GT

---

## RQ4: Does FDM generalize to held-out games?

Required experiments:

* held-out-game split
* FDM-ActionOnly vs FDM-GT vs FDM-Pseudo
* with vs without game ID ablation
* free-running-on-logged-video evaluation

Success evidence:

* FDM-GT beats action-only on held-out games
* FDM-Pseudo remains non-trivial on held-out games
* game ID improves in-distribution but is not required for non-trivial generalization
* performance is not dominated by a few high-resource games

---

## RQ5: Does FDM-1-style recipe scale on D2E?

Required experiments:

* data scale sweep
* context length sweep
* model size sweep, if compute allows
* VE adaptation sweep

Success evidence:

* IDM quality improves with more data
* FDM-GT improves with more data
* FDM-Pseudo improves as IDM quality improves
* longer context improves or stabilizes prediction in at least some categories
* scaling trend is visible in macro-game metrics, not only micro-average

---

# Training Plan

## Phase 0: Data pipeline and baselines

Deliverables:

* D2E / OWAMcap reader
* video frame sampler
* 50ms action binning
* action tokenizer
* action de-tokenizer
* MCAP writer for predicted actions
* evaluation wrapper around D2E-style metrics
* per-game and per-category split manifest
* data scale split manifest

Checks:

* video timestamps align with input events
* keyboard/mouse overlays match visualizer
* raw HID deltas reconstruct plausible mouse movement
* no-op distribution is logged
* event overflow rate is logged
* per-game hours and action distributions are logged

Baselines:

* no-op baseline
* zero-mouse baseline
* previous-action repeat baseline
* action-only transformer
* V-JEPA 2 frozen action probes

---

## Phase 1: Video Encoder ablations

Candidates:

* VE-0: frozen V-JEPA 2 + probes
* VE-1: frozen V-JEPA 2 + trainable temporal compressor
* VE-2: V-JEPA 2 LoRA/adapters or last-block finetuning
* VE-3: V-JEPA 2 masked video domain adaptation
* VE-4: end-to-end finetuned V-JEPA 2, optional

Evaluate:

* action probes
* IDM downstream performance
* FDM downstream performance
* throughput and memory
* per-game / per-category metrics

Promotion rule:

* only top VE candidates are used for expensive IDM/FDM sweeps.

---

## Phase 2: IDM ablations

Candidates:

* IDM-0: no-op / majority baseline
* IDM-1: action-prior baseline
* IDM-2: causal IDM
* IDM-3: non-causal IDM
* IDM-4: calibrated/filterable IDM
* IDM-5: per-game specialist IDM, optional

Ablations:

* VE candidate
* future offset τ
* context length
* K action slots
* mouse tokenization
* mask schedule
* iterative unmasking steps

Evaluate:

* offline IDM action reconstruction
* per-game / per-category metrics
* calibration
* pseudo-label distribution
* pseudo-label confidence vs correctness

Promotion rule:

* select best IDM candidate for pseudo-label generation based on validation pseudo-label usefulness proxy, not only token accuracy.

---

## Phase 3: Pseudo-label generation

Run selected IDM candidate(s) over pseudo-label subset.

Create:

* D_GT
* D_PSEUDO_ALL
* D_PSEUDO_FILTERED
* D_MIX

Validate:

* predicted MCAP can be evaluated
* pseudo-label confidence correlates with correctness
* pseudo-label distribution does not collapse to no-op
* pseudo-label quality is reported per game
* filtered labels improve validation-quality/coverage tradeoff

---

## Phase 4: FDM ablations

Candidates:

* FDM-0: no-op baseline
* FDM-1: previous-action repeat baseline
* FDM-2: ActionOnly transformer
* FDM-3: VideoOnly transformer
* FDM-4: FDM-GT
* FDM-5: FDM-Pseudo
* FDM-6: FDM-FilteredPseudo
* FDM-7: FDM-Mix
* FDM-8: FDM-GT with game ID, ablation
* FDM-9: FDM-Pseudo with game ID, ablation

Ablations:

* context length
* VE candidate
* GT vs pseudo labels
* filtered pseudo vs unfiltered pseudo
* GT/pseudo mixture ratio
* with vs without game ID
* model size
* action tokenization
* key/button state auxiliary target

Evaluate:

* offline teacher-forced metrics
* free-running-on-logged-video metrics
* per-game and per-category metrics
* held-out-game performance
* data scale trend

---

## Phase 5: Scale trend experiments

Run selected configurations on data scales:

```text
1%
5%
10%
25%
50%
100%
```

Minimum configurations:

* best IDM candidate
* FDM-GT
* FDM-Pseudo
* FDM-FilteredPseudo
* FDM-Mix, if compute allows

Report:

* IDM quality vs data
* FDM-GT quality vs data
* FDM-Pseudo quality vs data
* Pseudo/GT ratio vs data
* held-out-game metrics vs data
* per-game macro-average vs data

---

# Key Reproduction Decisions

## Data

* exact D2E revision
* exact list of games
* per-game categories
* train/val/test split
* held-out-game split
* data scale split
* whether to include menu/loading/inactive segments
* whether to use audio
* whether to use active window metadata
* whether to train one generalist model or also per-game specialists

## Tokenization

* 50ms vs 33ms timestep
* K action slots per bin
* mouse compound token vs separate dx/dy tokens
* mouse bin boundary formula
* scroll representation
* click-position grid size
* key/button state auxiliary targets
* event overflow handling

## Video Encoder

* V-JEPA 2 checkpoint
* frozen vs LoRA/adapters vs last-block finetune
* masked video domain adaptation objective
* number of video tokens per bin
* temporal compressor architecture
* whether to finetune encoder during IDM/FDM training
* feature caching vs on-the-fly feature extraction

## IDM

* non-causal attention window
* future offset τ
* masked diffusion schedule
* number of unmasking steps
* top-k schedule
* confidence filtering
* pseudo-label calibration
* all-games model vs per-game specialist model

## FDM

* context length
* action loss weights
* pseudo-label filtering
* GT/pseudo mixture ratio
* sampling strategy for free-running-on-logged-video
* whether to use game ID embedding
* whether to add action chunk prediction
* model size sweep

---

# Hardware Assumptions

Target hardware:

* H200 x 4 GPU cluster

Recommended training strategy:

* bf16
* FlashAttention
* activation checkpointing
* FSDP or DeepSpeed ZeRO
* cached V-JEPA 2 features for frozen-backbone experiments
* on-the-fly V-JEPA 2 only for finetuning experiments

Expected feasible scope:

* VE-0 / VE-1 / VE-2 ablations
* Tiny/Base IDM and FDM candidates
* 10s and 30s context sweeps
* data scale sweep with selected configurations
* full D2E offline evaluation

Expected expensive scope:

* full V-JEPA 2 finetuning for all ablations
* 60s context with many video tokens/bin
* large IDM/FDM across all candidates
* online rollout infrastructure

---

# Recommended Experiment Schedule

## Minimal RQ-covering run

```text
VE:
  VE-0, VE-1, VE-2

IDM:
  IDM-0, IDM-1, IDM-2, IDM-3, IDM-4

FDM:
  FDM-0, FDM-1, FDM-2, FDM-3, FDM-4, FDM-5, FDM-6, FDM-7

Splits:
  recording-level in-distribution
  held-out-game
  pseudo-label simulation

Evaluations:
  offline IDM
  pseudo-label usefulness
  teacher-forced FDM
  free-running-on-logged-video
  scale trend on selected configs
```

---

# Recommended Defaults

```text
Dataset:
  D2E-480p
  all games
  pinned dataset revision
  recording-level train/val/test split
  held-out-game split
  per-game macro reporting

Timebase:
  50ms bins
  20fps sampled visual stream

Video Encoder:
  pretrained V-JEPA 2
  VE-1 as main default:
    frozen V-JEPA 2 + trainable temporal compressor
  VE-2 as main adaptation ablation:
    V-JEPA 2 LoRA/adapters or last-block finetuning
  4 video tokens per bin
  hidden size 512

Action Tokenization:
  MOUSE_MOVE_BIN_<xbin>_<ybin>
  49 signed exponential bins per axis
  fixed K = 8 event slots per bin
  keyboard: KEY_DOWN/KEY_UP by key code
  mouse buttons: DOWN/UP tokens
  scroll: direction or binned tokens
  click position: auxiliary 32x18 grid head

IDM:
  bidirectional transformer
  d_model 512
  8 layers
  10s context
  future offset τ = 100ms
  8-step or 16-step iterative unmasking
  no game ID in main model
  game ID only as ablation

FDM:
  causal transformer
  d_model 512
  8 layers
  10s context
  next-bin action prediction
  train FDM-GT, FDM-Pseudo, FDM-FilteredPseudo, FDM-Mix

Evaluation:
  no online rollout
  D2E-style 50ms offline metrics
  pseudo-label usefulness
  teacher-forced next-action metrics
  free-running-on-logged-video metrics
  per-game macro-average
  held-out-game split
  data scale trend
```

---

# Expected Outcome

This PoC should not claim to reproduce FDM-1 scale.

It should evaluate whether FDM-1’s approach transfers to the full D2E game distribution, and which design choices make the recipe work or fail.

## Strong positive result

* V-JEPA 2 based video encoder provides useful compressed video tokens across many games.
* IDM pseudo-labels actions across many D2E games with useful accuracy.
* Non-causal IDM consistently beats causal IDM.
* FDM-Pseudo approaches FDM-GT.
* Confidence filtering improves or stabilizes pseudo-label training.
* All-games model outperforms action-only and no-op baselines.
* Held-out-game performance is non-trivial.
* Longer context or more data improves performance.

## Partial positive result

* V-JEPA 2 features work for some game categories but not all.
* IDM works well for mouse movement but struggles with sparse keyboard/button events.
* FDM-GT works, but FDM-Pseudo is limited by IDM noise.
* Performance is strong for some game types but weak for others.
* Game ID embeddings improve in-distribution results but hurt interpretation of generalization.
* Scale trend appears in micro-average but not macro-game average.

## Negative result

* V-JEPA 2 features do not preserve enough action-relevant detail for D2E.
* IDM pseudo-labels are too noisy for FDM training.
* FDM relies mostly on action autocorrelation rather than video.
* Model performance is dominated by a few large/easy games.
* Offline D2E prediction is insufficient to validate the FDM-1 recipe.

Even a partial or negative result is useful, because it identifies which part of the FDM-1 recipe fails under public D2E-scale data.
