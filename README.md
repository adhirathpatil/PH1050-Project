# PH1050 Final Group Project, Problem 9: Atoms and the Periodic Table

PH1050 Foundation of Computational Physics (IIT Madras, Dr. Ayana Ghosh).

> **The question:** How do atoms stick together, and do the noble gases really behave alike?

This repository answers that question in three steps. We (1) build and *verify* a physics model of how atoms attract and repel, (2) test one of its predictions against *real data*, and (3) ask whether a *pattern-only* (machine-learning) model can predict boiling points from the periodic table, including for elements it has never seen.

The same idea runs through all three parts: **never trust a number until something independent has agreed with it.** Physics gives us exact answers to check against; real data tells us whether the model matches the world; a baseline and a held-out test tell us whether a learned model has actually learned anything.

---

## Contents

1. [Project at a glance](#1-project-at-a-glance)
2. [Repository layout](#2-repository-layout)
3. [Setup and how to run](#3-setup-and-how-to-run)
4. [Part 1: The physics model](#4-part-1-the-physics-model)
5. [Part 2: Data and statistics](#5-part-2-data-and-statistics)
6. [Part 3: The data-driven model](#6-part-3-the-data-driven-model)
7. [Two implementations of Part 3](#7-two-implementations-of-part-3)
8. [Assumptions and design decisions](#8-assumptions-and-design-decisions)
9. [Known issues in the code](#9-known-issues-in-the-code)
10. [Limitations](#10-limitations)
11. [Reproducibility](#11-reproducibility)
12. [References](#12-references)

---

## 1. Project at a glance

| Part | What we did | Headline result |
|---|---|---|
| **1. Physics model** | Lennard-Jones potential; minimum by bisection; force by central differences; lowest-energy clusters of 2 to 7 atoms | Minimum at $r^*=2^{1/6}$ with $V=-1$ (12 digits); force error $\propto h^2$ (measured slope 1.98); all known cluster energies reproduced |
| **2. Data and statistics** | Cleaned a 118-element table; tested whether noble gases share a melting/boiling ratio | Mean ratio 0.953 (Ne to Xe); neon is the (weak) outlier |
| **3. Data-driven model** | Predicted boiling point from five element properties with four models, tested by cross-validation and by extrapolation to unseen periods | Decision tree best in both tests: MAE 658 K (CV) and 830 K (extrapolation), against baselines of 1393 K and 1567 K |

---

## 2. Repository layout

```
.
├── minimum_energy.py          # Part 1: V(r), force, bisection, convergence study
├── grid_minimizer.py          # Part 1: cluster energy/gradient, BFGS from random starts
├── part2_stats.py             # Part 2: noble-gas Tmelt/Tboil ratios, prediction interval
├── clean_elelments.py         # Part 2: keep needed columns, count missing values
├── ElementData.csv            # 118 elements x 30 columns (see Section 5.1)
├── README.md / README1.md     # short readmes (README1 has run instructions for Part 3)
├── lj_potential.png, force_convergence.png,
│   energy_per_atom.png, noble_gas_ratio.png   # figures written by the Part 1/2 scripts
│
├── Scikit_learn_based/        # Part 3, scikit-learn implementation (main results)
│   ├── 5fcross_validation/    #   baseline.py, linear_regression.py, knn.py,
│   │                          #   decision_tree.py, data_model.py, knn_analysis.py, result.py
│   └── Extrapolation/         #   *_ex.py models, data_split.py, data_model.py, result_ex.py
│
└── Without_scikit_learn/      # Part 3, from-scratch NumPy implementation (cross-check)
    ├── 5fold_cross validation/#   baseline.py, knn.py, linear_regressor.py,
    │                          #   cross_validation.py, data_model.py, main.py, result.py
    └── extrapolation/         #   data_split.py, data_model.py, main.py, result.py
```

Each module does one job. Plotting and "run everything" code sits under `if __name__ == "__main__":` so functions can be imported (for tests or other scripts) without opening a plot.

(`commit_test.py` and `gitpush_test.py` are scratch files from setting up Git and are not part of the analysis.)

---

## 3. Setup and how to run

**Requirements:** Python 3 (developed on 3.12), `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`.

```bash
pip install numpy scipy pandas matplotlib scikit-learn
```

**Part 1 and Part 2** (from the repository root):

```bash
python minimum_energy.py     # minimum, force check, convergence plot
python grid_minimizer.py     # cluster search N = 2..7, energy-per-atom plot
python part2_stats.py        # noble-gas ratios and prediction interval
```

**Part 3, scikit-learn version:**

```bash
cd Scikit_learn_based/5fcross_validation
python result.py                      # 5-fold cross-validation, all four models
python knn_analysis.py                # k = 1..20 sweep for KNN

cd ..                                 # now in Scikit_learn_based/
python -m Extrapolation.result_ex     # train periods 1-4, test periods 5-6
```

**Part 3, from-scratch version:**

```bash
cd "Without_scikit_learn/5fold cross validation"   # note the spaces
python main.py                        # baseline, KNN, linear regression
python knn_analysis.py                # k sweep (writes knn_cv_analysis.png)

cd ../extrapolation
python main.py
python knn_analysis.py                # writes knn_analysis.png
```

> The folder is named `5fold_cross validation` (underscore, then space) in the repository; use quotes in the shell.

---

## 4. Part 1: The physics model

### 4.1 The Lennard-Jones potential

**The problem it solves.** We want the energy of a group of atoms as a function of where they are. Quantum mechanics could in principle give this but is far too expensive for repeated experiments. Lennard-Jones proposed a cheap stand-in for neutral, closed-shell atoms such as the noble gases: only the *distance* $r$ between two atoms matters, and the energy has two competing pieces:

$$V(r)=4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12}-\left(\frac{\sigma}{r}\right)^{6}\right]$$

- The **$r^{-12}$ term** is a very steep repulsive wall: atoms whose electron clouds overlap push apart strongly.
- The **$-r^{-6}$ term** is a weak long-range attraction (neutral atoms have flickering, short-lived electric dipoles that attract each other; the effect fades as $r^{-6}$).
- The attraction wins at moderate distances and the wall wins up close, so there is a sweet spot. The exponent 6 has physical meaning; the exponent 12 does not: it is simply the square of the $r^{-6}$ term, cheap to compute and steep enough.

**Assumptions** (stated explicitly, because every later result depends on them):

1. Forces act between *pairs* of atoms and depend only on their distance (no three-body effects, no preferred directions). Reasonable for round closed-shell atoms; poor for metals or molecules with directional bonds.
2. Atoms are *classical* (no quantum effects). Reasonable for heavy atoms, weaker for light ones. This matters again in Part 2 (neon).
3. $\varepsilon=\sigma=1$. This costs nothing: $\sigma$ becomes the unit of length and $\varepsilon$ the unit of energy, so every noble gas is the *same* problem measured with different rulers.

### 4.2 The minimum, analytically

The slope of $V$ is zero at the bottom of the valley:

$$V'(r)=4\left(-12r^{-13}+6r^{-7}\right)=0\;\Rightarrow\;r^6=2\;\Rightarrow\;r^*=2^{1/6}\approx1.1225$$

Substituting back, $V(r^*)=4\left(\tfrac14-\tfrac12\right)=-1$: the pair is bound with depth exactly $\varepsilon$.

### 4.3 The minimum, numerically: bisection

`find_minimum(dV, a, b, tol=1e-12)` in `minimum_energy.py`.

The slope $V'$ is negative left of the minimum and positive right of it, so there must be a *sign change* in between. Bisection looks at the midpoint; if the slope is still negative the minimum lies to the right, otherwise to the left; it discards the other half and repeats. The interval halves every step, so about 40 steps take width 1 down to $10^{-12}$. It is slower than fancier methods but cannot fail while the sign change is bracketed, which is why we use it as the reference.

**Why we bisect $V'(r)=0$ and not $V(r)=-1$.** The problem sheet says to check "$V=-1$". But $V$ only *touches* $-1$ at the minimum and never crosses it, so there is no sign change to bracket and bisection cannot work. Bisecting the slope avoids this.

**Result.** Bracketing $[1,2]$ with the analytic derivative gives $r^*=1.122462048309$, identical to $2^{1/6}$ to all 12 printed digits, with $V(r^*)=-1.000000000000$. Using the central-difference derivative (next section) instead gives $1.122462049$, agreeing to about $10^{-9}$.

### 4.4 The force by central differences

The force is $F=-V'(r)$. Pretending we did not know the formula, we estimate the slope by looking a small distance $h$ to each side:

$$F_{\text{num}}(r)=-\frac{V(r+h)-V(r-h)}{2h}$$

**Why this is accurate.** Taylor-expanding $V(r\pm h)$ shows the first-order error terms cancel by symmetry, leaving

$$F_{\text{num}}-F=-\frac{h^2}{6}V'''(r)+\cdots\qquad(\text{truncation error}\propto h^2).$$

Shrinking $h$ tenfold should shrink this error a hundredfold (slope $+2$ on a log-log plot).

**Why $h$ cannot be shrunk forever.** Computers store numbers to about 16 digits (relative error $\varepsilon_m\approx1.1\times10^{-16}$). For tiny $h$, $V(r+h)$ and $V(r-h)$ are almost equal, so their difference throws away most of those digits, and dividing by the tiny $2h$ magnifies what is left. This round-off error scales as $\varepsilon_m|V|/h$ (slope $-1$). Two opposing errors imply a sweet spot, found by setting the derivative of $\tfrac{h^2}{6}|V'''|+\varepsilon_m|V|/h$ to zero:

$$h_{\text{opt}}\approx\left(\frac{3\,\varepsilon_m|V|}{|V'''|}\right)^{1/3}\approx2\times10^{-6}\quad(\text{at }r=1.5)$$

**Why powers of two.** Floating-point numbers are stored in binary. With $h=2^{-k}$, the points $r\pm h$ are represented *exactly*, so no extra rounding is introduced by forming them and the only round-off left is the one being analysed. The convergence study uses $h=2^{-k}$ for $k=2,\dots,26$ (`convergence_study`).

**Results** (`force_convergence.png`):

| Regime | Fitted slope | Expected |
|---|---|---|
| Truncation, $h>10^{-4}$ | $+1.98$ | $+2$ |
| Round-off, $h<10^{-7}$ | $-1.32$ | $\approx-1$ |
| Smallest error | $4.5\times10^{-12}$ at $h\approx1.9\times10^{-6}$ | theory: $\approx2\times10^{-6}$ |

The truncation branch is the cleanest straight line in the project. The round-off branch is noisy because round-off is effectively random, so $-1.32$ is only roughly $-1$. The bottom of the curve sits where the formula predicts. The two slopes come from `fit_slope` and `fit_roundoff_slope`, each fitted only on points well inside its regime.

**Tolerance check.** Before trusting $F_{\text{num}}$ over the range of interest we assert that its error is below a stated tolerance. At fixed $h$ the error is about $\tfrac{h^2}{6}|V'''|$, and $V'''$ is large at short range (about $-3.9\times10^4$ at $r=0.9$), so the tolerance must be chosen with that in mind. In the report: $h=10^{-5}$, absolute error, $0.9\le r\le3$, tolerance $10^{-6}$; the maximum absolute error is $6.49\times10^{-7}$ (at $r=0.9$), so the check passes. (See [Section 9](#9-known-issues-in-the-code) on keeping the code consistent with this.)

### 4.5 Lowest-energy clusters

`grid_minimizer.py`

**The question.** Given $N$ atoms, which arrangement has the lowest total energy? The energy is the sum over all pairs,

$$E=\sum_{i<j}V(r_{ij}),$$

a function of the $3N$ coordinates (the energy is unchanged by moving or rotating the whole cluster, so only $3N-6$ combinations are effectively free for $N\ge3$; the minimiser still works with all $3N$ and the extra directions are harmless flat directions).

**Why it is hard.** The energy landscape has many valleys. A minimiser started on a slope slides into the *nearest* valley, which is not necessarily the deepest. The remedy is blunt but reliable: start from many random arrangements and keep the lowest result. We follow the hint: **60 starts**, positions drawn uniformly from a box 2.4 wide, minimised with `scipy.optimize.minimize` (BFGS, a method that uses the slope to walk downhill). The random generator is seeded for reproducibility (`np.random.default_rng(seed)`, default `seed=0`).

**Engineering choices that matter:**

- **Exact gradient** (`gradient(x)`). We supply the analytic slope $\partial E/\partial\mathbf r_i=\sum_j V'(r_{ij})(\mathbf r_i-\mathbf r_j)/r_{ij}$ instead of letting SciPy estimate it. It is faster and more accurate; we checked it against finite differences (difference $\sim2\times10^{-6}$ against a gradient size of about 22.5).
- **Vectorised energy** (`energy(x)`). All pair distances are computed at once as an $N\times N$ table (each pair appears twice, hence the factor $\tfrac12$); no Python loops. Positions are stored as one flat array of length $3N$, as `minimize` requires, and reshaped to $(N,3)$ inside the function.
- **Rejecting overlapping starts** (`random_start(N, rng, min_sep=0.7)`). If two atoms start almost on top of each other, $r^{-12}$ is astronomically large and BFGS can break. We discard random starts with any pair closer than 0.7 (our choice, not part of the hint).
- **Acceptance by gradient norm.** A run is accepted when the gradient is essentially zero (norm $<10^{-4}$), *not* when SciPy's `success` flag is true: with a strict stopping rule BFGS often reports "precision loss" while sitting at a genuine minimum, and discarding those runs had silently made $N=6$ miss its known value.

**Results** (`best_cluster(N)` returns the best energy, positions and all run energies):

| $N$ | Shape found | $E$ found | $E$ known | Difference | Starts reaching it | Close pairs |
|---|---|---|---|---|---|---|
| 2 | dimer | −1.0000 | −1.000 | 0 | 100% | 1 |
| 3 | triangle | −3.0000 | −3.000 | 0 | 100% | 3 |
| 4 | tetrahedron | −6.0000 | −6.000 | 0 | 100% | 6 |
| 5 | triangular bipyramid | −9.1039 | −9.104 | 1.5e-4 | 100% | 9 |
| 6 | octahedron | −12.7121 | −12.712 | −6e-5 | **2% (1 of 60)** | 12 |
| 7 | pentagonal bipyramid | −16.5054 | −16.505 | −4e-4 | 20% | 16 |

("Close pairs" = atom pairs closer than 1.3, our threshold. Differences are below $10^{-3}$, the precision of the quoted known values.)

**What we learn.**

1. All five known energies are reproduced, including $N=3$ ($-3$) and $N=4$ ($-6$).
2. *The energy is almost exactly minus the number of close pairs.* Each close pair contributes about $-1$; the rest is a small correction. This is the intuition "atoms want to maximise the number of neighbours at the sweet-spot distance", and it explains the energy-per-atom plot (`energy_per_atom.png`): the energy per atom keeps falling because each extra atom can touch more neighbours, with diminishing returns.
3. The search is fragile for $N=6$: only 1 of 60 starts found the octahedron. With only 10 starts we would most likely have missed it and reported a wrong answer. This is why the number of starts and the *hit rate* should always be reported.

---

## 5. Part 2: Data and statistics

### 5.1 The data

`ElementData.csv`: 118 elements, 30 columns (atomic number, mass, period, group, melting point, boiling point, electronegativity, ...). It is a periodic-table CSV of the kind suggested in the problem statement; *add the exact source URL here*: `TODO`.

### 5.2 Cleaning

Real tables have holes, and each hole needs a decision. Only 72 of 118 elements have all seven properties used in this project at once.

| Column | Missing (of 118) | Why missing | What we did |
|---|---|---|---|
| Boiling point (the target) | 11 | superheavy period-7 elements, never measured | drop these rows → **107** remain |
| Group | 28 (24 of the 107) | lanthanides/actinides have no group | set to 3 (where they sit in the table) |
| Electronegativity | 18 (11 of the 107) | undefined for He, Ne, Ar; unknown for 8 superheavies | median of the **training** data |
| Melting point | 11 | helium has no solid at 1 atm; superheavies | excluded from ratio calculations |

**Why not simply drop incomplete rows?** It would cut 107 usable elements to 72 (a third of the data) and delete whole chemical families (He, Ne and Ar all lack electronegativity). Dropping the *target* is different: a row without the answer cannot teach or test anything.

**Why the median comes from the training data only.** Using the test part would let information from the exam leak into the study material, making every score look better than it deserves.

### 5.3 Do the noble gases really behave alike?

`part2_stats.py`

**What the model predicts.** In Lennard-Jones units, temperature is measured in units of $\varepsilon/k_B$ and length in units of $\sigma$. Re-expressed in these units, every noble gas has an *identical* potential, so all the physics (melting, boiling) is the same up to a temperature scale set by $\varepsilon$. A dimensionless ratio such as $T_\text{melt}/T_\text{boil}$ has no scale left, so the model predicts one value for all noble gases. This is the idea of *corresponding states*: the gases are the same story told in different units.

**Data** (helium excluded: no melting point at 1 atm):

| | $T_\text{melt}$ (K) | $T_\text{boil}$ (K) | Ratio |
|---|---|---|---|
| Ne | 24.56 | 27.07 | 0.907 |
| Ar | 83.80 | 87.30 | 0.960 |
| Kr | 115.79 | 119.93 | 0.965 |
| Xe | 161.40 | 165.03 | 0.978 |

Mean $=0.953$, sample standard deviation $=0.031$ (Ne to Xe): about 3% of the mean, tiny next to the six-fold range of boiling points (27 to 165 K). The model's headline claim holds well. A 95% confidence interval for the mean of the four is 0.903 to 1.002.

**Which element is the outlier, and is it really?** Neon sits lowest, but "looks different on a plot" is not a test. We ask: if Ne behaved like the others, where would a *new* noble-gas ratio fall? From Ar, Kr, Xe (mean 0.968, standard deviation 0.0093) a 95% *prediction interval* is

$$\bar x\pm t_{0.975,\,n-1}\,s\sqrt{1+\tfrac1n}=0.968\pm4.30\times0.0093\times\sqrt{1+\tfrac13}=(0.922,\;1.014),$$

with $n=3$ and therefore $n-1=2$ degrees of freedom. (The $\sqrt{1+1/n}$ accounts for both our uncertainty in the mean and the natural scatter of an individual gas.) Neon's $0.907$ falls outside, with $t=-5.7$ and two-sided $p\approx0.03$.

**Conclusion: neon is an outlier, but weakly established.** Only three reference points make the interval wide, and we chose neon *after* seeing it was the lowest, which flatters the $p$-value.

**Why neon?** A hypothesis, not something our data tests. Lennard-Jones is classical, but light atoms show noticeable quantum motion even at low temperature, which effectively makes them "larger" and less tightly bound than the classical model assumes. Neon is the lightest atom in the sample (helium is lighter still and is the extreme case: it does not even solidify at 1 atm). That would be assumption 2 of Section 4.1 failing first where it should.

---

## 6. Part 3: The data-driven model

### 6.1 Setup

**Task.** Predict an element's boiling point from five facts: **atomic number, atomic mass, period, group, electronegativity** (107 elements). The model gets no physics, only examples.

**Score.** Mean absolute error (MAE), $\tfrac1n\sum|y_i-\hat y_i|$, in kelvin: the average miss, in units everyone understands.

**Baseline.** Predict the training-set mean boiling point (about 2436 K) for every element. Any model that cannot beat this has learned nothing.

**Three models, in order of flexibility:**

| Model | In plain words |
|---|---|
| Linear regression | boiling point ≈ weighted sum of the five features; one global tilted surface; rigid |
| k-nearest neighbours (KNN) | find the $k$ most similar elements (features first rescaled to comparable size, so mass, which reaches ~300, does not drown out period, which stops at 7) and average their boiling points; local, no assumed shape |
| Decision tree | a sequence of yes/no questions ("is group ≥ 6?") that splits the elements into bins, predicting each bin's average; piecewise-constant, can follow irregular patterns |

**Why a straight line is expected to struggle.** Boiling point reflects bonding type (metallic, covalent network, weak dispersion forces), so it varies periodically but not monotonically down a period.

# Predicting Boiling Points of Elements

## Methodology

Two implementations were developed: one using Scikit-learn and another implementing the algorithms from scratch.

The models evaluated were:
- Baseline Regression
- K-Nearest Neighbours (KNN)
- Linear Regression
- Decision Tree Regression (Scikit-learn implementation)

Two evaluation approaches were used:
- **Five-fold Cross-Validation:** The dataset was divided into five folds. Each fold was used once for testing, while the remaining four were used for training.
- **Extrapolation:** Models were trained using elements from Periods 1–4 and tested on elements from Periods 5–6 to evaluate their performance on previously unseen periods.

Rows with missing boiling-point values were removed. Missing Group values were filled with 3, and missing Electronegativity values were imputed using the median calculated from the training data. Mean Absolute Error (MAE) was used as the evaluation metric.

## Results

### Five-fold Cross-Validation

| Model | Scikit-learn MAE | Scratch MAE |
|---|---|---|
| Baseline | 1392.95 | 1392.95 |
| KNN | 944.13 | 884.72 |
| Linear Regression | 1101.42 | 1106.22 |
| Decision Tree | 657.75 | — |

### Extrapolation

| Model | Scikit-learn MAE | Scratch MAE |
|---|---|---|
| Baseline | 1566.66 | 1533.85 |
| KNN | 1675.30 | 1811.08 |
| Linear Regression | 1486.73 | 2331.41 |
| Decision Tree | 830.39 | — |

**Note:** We could not implement the decision tree from scratch because it was too complicated.
**Observations **
In cross-validation, KNN and Linear Regression performed better than the baseline. However, their performance deteriorated during extrapolation, indicating difficulty predicting elements from periods not represented in the training data.
KNN showed an interesting change in behaviour with the number of neighbours ((k)). In cross-validation, increasing (k) generally increased MAE because including more distant and less similar elements could oversmooth predictions. During extrapolation, increasing (k) reduced MAE, possibly because averaging more neighbours reduced dependence on individual, potentially unrepresentative neighbours. This is specific to the dataset and does not mean KNN can reliably extrapolate beyond its training range.
The Scikit-learn Decision Tree achieved the lowest MAE among the evaluated models in both experiments. However, its performance should be interpreted alongside the possibility of overfitting.
Despite these improvements, the prediction errors remain substantial, suggesting that the selected features and models do not fully capture the complexity of elemental boiling points.


## 8. Assumptions and design decisions

| Where | Decision | Why |
|---|---|---|
| Part 1 | $\varepsilon=\sigma=1$ | units absorb them; all noble gases are the same problem |
| Part 1 | bisect $V'(r)=0$, not $V=-1$ | $V$ never crosses $-1$ |
| Part 1 | $h=2^{-k}$ | $r\pm h$ exactly representable in binary |
| Part 1 | analytic gradient for BFGS | faster, more accurate than finite differences |
| Part 1 | reject starts with a pair closer than 0.7 | $r^{-12}$ blows up and breaks the optimiser |
| Part 1 | accept a run by gradient norm $<10^{-4}$ | BFGS often reports "precision loss" at a true minimum |
| Part 1 | 60 random starts, fixed seed | follows the hint; reproducible |
| Part 2 | exclude He from the ratio | no solid at 1 atm |
| Part 2 | prediction interval from Ar, Kr, Xe, $n-1=2$ d.o.f. | tests whether a *new* gas fits the others |
| Part 3 | drop rows with no boiling point | cannot teach or test |
| Part 3 | group = 3 for lanthanides/actinides | convention (their position in the table) |
| Part 3 | median electronegativity from training data only | avoids leaking test information |
| Part 3 | standardise features with training statistics | KNN distances must not be dominated by atomic mass |
| Part 3 | report MAE in K, with a baseline | interpretable and gives a yardstick |
| Part 3 | split extrapolation by *period*, not at random | tests unseen regions of the table |

## 9. Limitations

- Only four noble gases enter the ratio test, and neon was singled out after looking.
- Part 3 has 107 elements, and the extrapolation result is one split of 50 elements with one random seed.
- $k=5$ was fixed rather than tuned (and is not the cross-validation optimum), and the tree depth was not limited, so overfitting is possible; the tree's large fold-to-fold variation is consistent with this.
- Hyper-parameters were not chosen by nested cross-validation.
- He, Ne and Ar (and eight superheavy period-7 elements, which enter only the cross-validation) receive the median electronegativity even though noble gases behave differently from typical elements.
- The data-driven model finds periodic structure (group); it does not discover new physics.

---

## 10. Reproducibility

- Cluster search: `np.random.default_rng(seed)` with `seed=0` in `best_cluster`.
- scikit-learn cross-validation: `KFold(5, shuffle=True, random_state=42)`; decision tree `random_state=42`.
- Extrapolation split is deterministic (by period), so it needs no seed.
- Re-running the scripts regenerates every number and figure quoted above.

---

## 11. References

1. J. E. Lennard-Jones, On the determination of molecular fields. II. From the equation of state of a gas, *Proc. R. Soc. A* **106**, 463 (1924). **
2. F. Pedregosa *et al.*, Scikit-learn: Machine Learning in Python, *J. Mach. Learn. Res.* **12**, 2825 (2011).
3. T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer (2009).
4. Element property table (`ElementData.csv`), obtained from an online Git repository.
