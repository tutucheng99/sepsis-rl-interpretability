# 🚨 CRITICAL REPRODUCIBILITY ISSUES - MUST FIX BEFORE SUBMISSION

**Date:** November 11, 2025
**Status:** BLOCKING - Submission not ready

---

## Executive Summary

A thorough code review has identified **6 critical reproducibility failures** that completely block independent verification of the manuscript's main claims. These issues affect:
- Dataset generation (cannot run)
- Algorithm comparison (3 of 6 algorithms missing)
- Action space specification (24 vs 25 mismatch)
- Survival metrics (incorrect calculation)
- Simulator dependency (not included)

**Recommendation:** DO NOT SUBMIT until all issues are resolved.

---

## Issue #1: Missing Simulator Dependency 🔴 BLOCKING

### Problem
The codebase depends on `gym-sepsis` but it is **not included** in the submission package.

### Evidence
```python
# code/src/envs/sepsis_wrapper.py:15-23
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'gym-sepsis'))  # ❌ gym-sepsis/ not in submission

try:
    from gym_sepsis.envs.sepsis_env import SepsisEnv
except ImportError:
    raise ImportError("Cannot import gym-sepsis...")  # ⚠️ Will always fail for reviewers
```

### Impact
- **Reviewers cannot run ANY experiments**
- All scripts fail with ImportError
- Zero reproducibility

### Current Status
- ✅ gym-sepsis exists in original project: `project_1/gym-sepsis/`
- ❌ gym-sepsis NOT in submission: `journal_submission/code/` (missing)

### Fix Required
**Option A (Recommended):** Include gym-sepsis as git submodule
```bash
cd journal_submission
git submodule add https://github.com/akiani/gym-sepsis.git gym-sepsis
git submodule update --init --recursive
```

**Option B:** Document installation in README
```markdown
## Installation
1. Clone gym-sepsis:
   git clone https://github.com/akiani/gym-sepsis.git
   cd gym-sepsis && pip install -e .
```

---

## Issue #2: Action Space Mismatch (24 vs 25) 🔴 CRITICAL

### Problem
**Manuscript claims 25 actions** (5×5 grid), but **code only uses 0-23** (24 actions).

### Evidence

**Manuscript (sections/04_methods.tex:18):**
```latex
Action Space. A discrete $5 \times 5$ grid over IV fluid and vasopressor
dosage bins (action $ = 5 \times \text{IV\_bin} + \text{VP\_bin}$),
yielding 25 actions.
```

**Code (src/data/collect_data.py:50):**
```python
def heuristic_policy(state):
    # ... clinical rules ...
    action = min(5 * iv_bin + vp_bin, 23)  # ❌ CAPPED AT 23! (0-23 = 24 actions)
    return action
```

**Baseline script (scripts/01_baseline_evaluation.py:28):**
```python
def random_policy(state):
    return np.random.randint(0, 24)  # ❌ Only 24 actions (0-23)
```

### Impact
- **All reported results are for 24-action space, not 25**
- Action (4, 4) = 24 is never sampled by any policy
- Figures/tables report incorrect action space size
- Invalidates reproducibility: reviewers get different results if they implement 25 actions

### Fix Required

**Option A:** Fix code to use 25 actions
```python
# Remove the min(..., 23) cap
action = 5 * iv_bin + vp_bin  # Range: 0-24 (25 actions)
```

**Option B:** Fix manuscript to say 24 actions
```latex
Action Space. A discrete grid of 24 actions over IV fluid and vasopressor
dosage bins, indexed 0-23.
```

**Recommendation:** Option B (fix manuscript) is safer since all experiments already used 24 actions.

---

## Issue #3: Dataset Size Mismatch 🔴 CRITICAL

### Problem
**Manuscript claims 10,000 episodes** (~100K transitions), but **code targets 2,100 episodes**.

### Evidence

**Manuscript (sections/04_methods.tex:24):**
```latex
We generated an offline dataset of 10,000 episodes (~100K transitions)
using a heuristic policy... Data partitioning: 9,000 train, 500 validation,
500 test episodes.
```

**Code (src/data/collect_data.py:1-7, 238-245):**
```python
"""
Data Collection Script for Sepsis RL Project

Collects offline dataset using heuristic policy
Target: 20,000 transitions (~2,100 episodes)  # ❌ Only 2,100 episodes!
Estimated time: ~1 hour
"""

# Line 238-245 (estimated):
n_target_transitions = 20_000
avg_episode_length = 9.5
n_episodes = int(n_target_transitions / avg_episode_length)  # ≈ 2,105 episodes
```

**No train/val/test split in code:**
```python
# ❌ Code does NOT create 9000/500/500 splits anywhere
# Only collects ~2,100 episodes and saves as one pickle file
```

### Impact
- **Reviewers cannot reproduce Tables 1-2** with the described dataset
- Results are based on 2,100 episodes, not 10,000
- No validation/test splits exist in code

### Fix Required

**Option A:** Rerun experiments with 10K episodes
```python
n_target_transitions = 100_000  # 10K episodes × 10 steps/episode
n_episodes = 10_000
```

**Option B:** Update manuscript to match code
```latex
We generated an offline dataset of 2,100 episodes (~20K transitions)
using a heuristic policy...
```

**Recommendation:** Option B (update manuscript) unless you can afford to rerun all experiments.

---

## Issue #4: Missing Online RL Implementations 🔴 BLOCKING

### Problem
**Manuscript evaluates 6 algorithms**, but **code only contains 3**.

### Evidence

**Manuscript (sections/05_results.tex:10-12):**
```latex
Table \ref{tab:overall-performance} and Figure \ref{fig:algorithm-comparison}
show results for 8 policies. DDQN-Attention achieves highest survival (95.4%),
with all methods in narrow range (94.0--95.4%).
```

**Table 1 claims:**
- ✅ BC (implemented: `scripts/02_train_bc.py`)
- ✅ CQL (implemented: `scripts/03_train_cql.py`)
- ✅ DQN (implemented: `scripts/04_train_dqn.py`)
- ❌ **DDQN-Attention** (missing)
- ❌ **DDQN-Residual** (missing)
- ❌ **SAC** (missing)

**Code search results:**
```bash
$ find scripts -name "*.py" | grep -E "(attention|residual|sac)"
# (no results)
```

**Visualization code (scripts/06_visualization.py:39-52):**
```python
# Only loads 3 algorithms:
'bc': load_d3rlpy_policy(models_dir / 'bc_simple_reward.d3'),
'cql': load_d3rlpy_policy(models_dir / 'cql_simple_reward.d3'),
'dqn': load_d3rlpy_policy(models_dir / 'dqn_simple_reward.zip')
# ❌ No DDQN-Attention, DDQN-Residual, or SAC
```

### Impact
- **Main claims are not reproducible**
- Figure 2 (leg_6model_comparison.png) cannot be regenerated
- Table 1 cannot be verified
- "600-fold interpretability difference" claim relies on these missing algorithms

### Fix Required

**Option A:** Add missing implementations
1. Implement DDQN-Attention (scripts/05_train_ddqn_attention.py)
2. Implement DDQN-Residual (scripts/05_train_ddqn_residual.py)
3. Implement SAC (scripts/05_train_sac.py)
4. Add LEG analysis for all 6 algorithms

**Option B:** Remove claims about missing algorithms
- Trim manuscript to only BC, CQL, DQN
- Remove Figure 2 (6-model comparison)
- Update abstract/discussion

**Recommendation:** If coauthor Yalun Ding implemented these, **obtain the code immediately**. Otherwise, must use Option B.

---

## Issue #5: Incorrect Survival Calculation 🟡 MODERATE

### Problem
Survival is inferred from **cumulative reward > 0**, which is incorrect for shaped rewards.

### Evidence

**Code (src/evaluation/metrics.py:66-72):**
```python
def evaluate_policy(env, policy_fn, ...):
    # ...
    while not done:
        # ... collect rewards ...
        episode_return += reward  # Accumulates ALL rewards (terminal + shaped)

    # Determine survival: positive terminal reward indicates survival
    survived = episode_return > 0  # ❌ WRONG with shaped rewards!
```

### Problem
With **shaped rewards** (code/src/envs/reward_functions.py:52-140):
```python
def paper_reward(state, next_state, action, done):
    # Shaped intermediate rewards:
    reward = (
        -0.05 * lactate_change +
        0.10 * bp_improvement +
        -0.01 * action_magnitude
    )
    if done:
        reward += 15 if survived else -15  # Terminal only
    return reward
```

If intermediate rewards are positive enough, `episode_return > 0` even if patient died (terminal = -15).

### Impact
- **False positive survivals** when using shaped rewards
- Biased SOFA-stratified metrics
- Affects reward_function_comparison.png accuracy

### Fix Required

**Track terminal outcome directly:**
```python
def evaluate_policy(env, policy_fn, ...):
    # ...
    survived = info.get('survived', episode_return > 0)  # Use env signal
    # OR: Parse final reward
    survived = (final_reward > 10)  # If terminal reward is ±15
```

---

## Issue #6: Corrupted env_wrapper.py 🟡 MODERATE

### Problem
Reviewer reports `code/src/utils/env_wrapper.py:1-138` is "heavily corrupted (garbled characters)".

### Evidence
**My inspection:** File appears intact with Chinese comments, but encoding may be broken in submission ZIP.

**Actual content (lines 1-10):**
```python
"""
环境兼容性包装器
解决 gym-sepsis 使用旧 gym 的问题
"""
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='gym')
import numpy as np
# ...
```

### Impact
- File may not be readable by reviewers
- Import errors if encoding is broken
- Chinese comments may confuse international reviewers

### Fix Required
1. **Re-save with UTF-8 encoding:**
   ```bash
   iconv -f GB2312 -t UTF-8 env_wrapper.py > env_wrapper_utf8.py
   ```

2. **Translate comments to English:**
   ```python
   """
   Environment compatibility wrapper
   Solves gym-sepsis dependency on legacy gym
   """
   ```

3. **Verify in ZIP:**
   ```bash
   unzip -p submission.zip code/src/utils/env_wrapper.py | head -20
   ```

---

## Verification Checklist Before Resubmission

### Critical (MUST FIX):
- [ ] Include gym-sepsis or document installation (Issue #1)
- [ ] Fix 24 vs 25 action space mismatch (Issue #2)
- [ ] Fix 2.1K vs 10K episode dataset claim (Issue #3)
- [ ] Add missing online RL code OR remove claims (Issue #4)

### Important (SHOULD FIX):
- [ ] Fix survival calculation for shaped rewards (Issue #5)
- [ ] Fix env_wrapper.py encoding/comments (Issue #6)

### Documentation:
- [ ] Update README with exact reproduction steps
- [ ] Add requirements.txt with exact versions
- [ ] Include dataset generation script with correct parameters
- [ ] Verify all file paths are relative, not absolute

---

## Recommended Action Plan

### Immediate (Today):
1. **Check if coauthor Yalun Ding has online RL code**
   - If yes: Obtain and integrate immediately
   - If no: Prepare to trim manuscript claims

2. **Decide on action space fix:**
   - Easiest: Update manuscript to say 24 actions (30 min)
   - Harder: Rerun all experiments with 25 actions (1 week)

3. **Decide on dataset size:**
   - Easiest: Update manuscript to say 2,100 episodes (30 min)
   - Harder: Regenerate 10K episodes dataset (1 day)

### Short-term (This week):
4. **Add gym-sepsis to submission:**
   - Document installation OR include as submodule

5. **Fix survival calculation:**
   - Modify metrics.py to track terminal outcome directly

6. **Fix encoding issues:**
   - Convert Chinese comments to English
   - Test ZIP extraction on Windows/Mac/Linux

### Before resubmission:
7. **Full reproducibility test:**
   - Fresh virtualenv
   - Install from scratch
   - Run all scripts
   - Verify all figures regenerate

---

## Estimated Time to Fix

**Minimal fixes (manuscript only):**
- Update action space description: 30 min
- Update dataset size: 30 min
- Add installation docs: 1 hour
- **Total: 2 hours**

**Full code fixes:**
- Implement 3 missing algorithms: 1-2 weeks
- Regenerate datasets: 1-2 days
- Full testing: 2-3 days
- **Total: 2-3 weeks**

---

## Contact for Resolution

If original online RL code exists (Yalun Ding's contribution), obtain it ASAP. Otherwise, must choose between:
1. **Extended timeline:** Implement missing code (2-3 weeks)
2. **Reduced scope:** Remove claims about online RL, focus on BC/CQL/DQN only

**Recommendation:** For fastest path to publication, update manuscript to match existing code (Option 2 for Issues #2-4).

---

**Status:** ⚠️ SUBMISSION BLOCKED until critical issues resolved
**Priority:** 🔴 HIGH - Affects core reproducibility claims
**Owner:** Corresponding author must decide on fix strategy
