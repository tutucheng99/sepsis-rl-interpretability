# Clinical Safety Guidelines for Sepsis Treatment

## Overview

This document specifies clinical safety constraints for the sepsis RL agent. These constraints are based on established clinical guidelines, expert knowledge, and medical literature.

**Version**: 1.0
**Last Updated**: 2025-11-11
**Status**: Draft (requires clinical validation)

---

## L1: Semantic Safety Constraints (Hard Rules)

These are **absolute constraints** that must NEVER be violated. They represent hard clinical contraindications and safety boundaries.

### 1. Vasopressor Administration Rules

#### 1.1 Minimum Blood Pressure Requirements

**Rule**: Vasopressors should not be withheld in severe hypotension.

```
IF MAP < 50 mmHg
THEN vasopressor_level >= 2  (at least low-medium dose)
```

**Rationale**: Severe hypotension (MAP < 50) indicates shock and requires immediate vasopressor support. Withholding vasopressors in this state is life-threatening.

**Source**: Surviving Sepsis Campaign Guidelines (2021)

#### 1.2 Maximum Vasopressor Dosage

**Rule**: Norepinephrine dosage must not exceed clinical safety limits.

```
norepinephrine_dose <= 0.5 mcg/kg/min
```

**Rationale**: Doses above 0.5 mcg/kg/min significantly increase risk of limb/organ ischemia without proven benefit.

**Source**: Clinical consensus, Surviving Sepsis Campaign

#### 1.3 Vasopressor Without Fluid Contraindication

**Rule**: High-dose vasopressors without adequate fluid resuscitation is contraindicated.

```
IF vasopressor_level == 4 (high dose)
AND fluid_level == 0 (no fluids)
AND cumulative_fluids < 30 mL/kg
THEN BLOCK this action
```

**Rationale**: "Dry squeeze" (vasopressors without volume) can worsen tissue perfusion and is not supported by guidelines.

**Source**: Surviving Sepsis Campaign - initial resuscitation bundle

---

### 2. Fluid Administration Rules

#### 2.1 Renal Dysfunction Limits

**Rule**: Limit fluid administration in severe renal dysfunction.

```
IF creatinine > 3.0 mg/dL OR oliguria (urine output < 0.5 mL/kg/hr)
THEN fluid_level <= 2  (no high-volume fluid boluses)
```

**Rationale**: Severe AKI with oliguria indicates inability to excrete fluids. Aggressive fluid administration can cause pulmonary edema and worsen outcomes.

**Source**: KDIGO AKI Guidelines, FEAST trial

#### 2.2 Fluid Overload Risk

**Rule**: Do not exceed maximum safe fluid bolus.

```
fluid_per_step <= 2000 mL
```

**Rationale**: Single boluses > 2L significantly increase risk of pulmonary edema and fluid overload.

**Source**: Clinical consensus

#### 2.3 Cumulative Fluid Limit

**Rule**: Monitor and limit cumulative positive fluid balance.

```
IF cumulative_fluid_balance > 10 L in 72 hours
THEN fluid_level == 0  (no further fluids unless diuresis planned)
```

**Rationale**: Positive fluid balance > 10L associated with increased mortality in septic shock.

**Source**: FACTT trial, recent meta-analyses

---

### 3. Combined Treatment Rules

#### 3.1 Hypotension with Adequate Fluids

**Rule**: If hypotensive despite adequate fluids, vasopressors are required.

```
IF MAP < 65 mmHg
AND cumulative_fluids >= 30 mL/kg
AND vasopressor_level == 0
THEN BLOCK "no vasopressor" action
```

**Rationale**: Persistent hypotension after adequate fluid resuscitation defines vasopressor-dependent shock. Delaying vasopressors worsens outcomes.

**Source**: Surviving Sepsis Campaign

---

### 4. Patient-Specific Contraindications

#### 4.1 Known Allergies

```
IF patient has documented allergy to agent
THEN BLOCK that agent
```

#### 4.2 Pre-existing Conditions

**Heart Failure**:
```
IF known systolic heart failure (EF < 40%)
THEN fluid_level <= 2  (cautious fluid administration)
```

**Arrhythmias**:
```
IF active arrhythmia AND on anti-arrhythmic
THEN careful with vasopressor dose escalation
```

---

## L2: Cognitive Safety Guidelines (Soft Rules / Warnings)

These are **warning conditions** that should trigger increased scrutiny or conservative behavior, but are not absolute contraindications.

### 1. Out-of-Distribution (OOD) States

**Guideline**: Exercise extreme caution (high conservatism) in states that are:
- Rarely seen in training data
- Far from typical clinical patterns
- High uncertainty in outcome prediction

**Action**:
- Increase α (conservatism) in CQL
- Prefer behavioral policy in these regions
- Log for human review

### 2. High Confounding Uncertainty

**Guideline**: When confounding uncertainty U_del(s,a) is high:
- Outcomes may not be causally attributable to treatment
- Selection bias may be strong
- Prefer actions with lower uncertainty

**Action**:
- Apply pessimistic Q-value adjustment
- Consider fallback to BC in these regions

### 3. Extreme Physiological States

**Warning conditions**:
- SOFA score >= 15 (high mortality risk)
- Multiple organ failure (≥3 failing organs)
- Refractory shock (high-dose vasopressors + MAP < 65)

**Action**:
- Flag for palliative care consideration
- Conservative treatment approach
- Human review recommended

### 4. Rapid Deterioration

**Warning**: Rapid increase in SOFA score (Δ SOFA > 2 in 4 hours)

**Action**:
- Increase monitoring frequency
- Consider human escalation
- Avoid aggressive interventions that might destabilize further

---

## Implementation Notes

### For Developers

1. **L1 Rules**: Implement as hard constraints in `phase_1_safety_layers/l1_semantic_safety.py`
2. **L2 Guidelines**: Implement as soft gates in `phase_1_safety_layers/l2_cognitive_safety.py`
3. **Testing**: Every L1 rule must have unit tests confirming it blocks violating actions
4. **Documentation**: All rules must cite clinical sources

### For Clinicians

1. **Review Required**: These guidelines are drafts and require clinical validation
2. **Local Protocols**: Adapt to local hospital protocols and patient populations
3. **Updates**: Clinical guidelines evolve; this document should be reviewed annually
4. **Exceptions**: In rare cases, violations may be clinically justified (document thoroughly)

---

## Validation Checklist

Before deployment, verify:

- [ ] All L1 rules reviewed by ICU clinicians
- [ ] Rules align with Surviving Sepsis Campaign guidelines
- [ ] Hospital-specific protocols incorporated
- [ ] Edge cases considered and documented
- [ ] Legal review completed
- [ ] Ethics review completed

---

## References

1. **Surviving Sepsis Campaign**: International Guidelines for Management of Sepsis and Septic Shock (2021)
2. **KDIGO AKI Guidelines**: Kidney Disease: Improving Global Outcomes - Acute Kidney Injury (2012)
3. **FACTT Trial**: Fluids and Catheters Treatment Trial (NEJM 2006)
4. **FEAST Trial**: Fluid Expansion As Supportive Therapy (NEJM 2011)

---

## Disclaimer

**These guidelines are for research purposes only and have not been validated for clinical use.**

Any clinical application requires:
- Formal clinical validation studies
- Regulatory approval (FDA, EMA, etc.)
- Extensive prospective testing
- Continuous human oversight
- Clear liability framework

**DO NOT USE for actual patient care without proper authorization and safeguards.**

---

**Version History**:
- v1.0 (2025-11-11): Initial draft based on Surviving Sepsis Campaign 2021
