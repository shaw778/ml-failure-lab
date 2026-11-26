# Test scaler refit

## Question and prerequisites

Did test preprocessing preserve the scale and origin learned on the training set? Feature transformers such as `StandardScaler` compute summary statistics (sample mean $\mu$ and standard deviation $\sigma$) during fitting. When test observations are scaled by independently fitting a new transformer rather than applying the training-fitted one, the coordinate system shifts. You need basic familiarity with feature scaling ($z = (x - \mu) / \sigma$) to follow this example.

## Original synthetic input

The training fixture defines a single continuous feature with two observations:

$$X_{\text{train}} = [[0.0], [2.0]]$$

Fitting a `StandardScaler` on this training set computes:

$$\mu_{\text{train}} = 1.0, \quad \sigma_{\text{train}} = 1.0$$

The primary test set introduces higher-magnitude observations:

$$X_{\text{test}} = [[10.0], [12.0]]$$

In the model's training coordinate system, these test observations correspond to $z = (10.0 - 1.0) / 1.0 = 9.0$ and $z = (12.0 - 1.0) / 1.0 = 11.0$. A second shifted test fixture evaluates $X_{\text{shift}} = [[4.0], [6.0]]$, corresponding to $z = [[3.0], [5.0]]$.

## Misleading operation and correction

In the incorrect workflow, a new `StandardScaler` is fitted directly on the test data:

$$\text{test\_scaler} = \text{StandardScaler}().\text{fit}(X_{\text{test}})$$

This computes test statistics $\mu_{\text{test}} = 11.0$ and $\sigma_{\text{test}} = 1.0$, transforming $[[10.0], [12.0]]$ into:

$$[[-1.0], [1.0]]$$

Even though the transformed array has valid dimensions and numeric types, its coordinates are inverted relative to the training feature space. A model or threshold rule trained to recognize $z > 5.0$ as a high-value signal fails silently, outputting $[0, 0]$ instead of $[1, 1]$.

The correction reuses the pre-fitted transformer from training:

$$X_{\text{test, scaled}} = \text{training\_scaler}.\text{transform}(X_{\text{test}})$$

This transforms $[[10.0], [12.0]]$ into $[[9.0], [11.0]]$, strictly preserving the training coordinate system and producing the correct predictions $[1, 1]$.

## Run and verification

```text
python -m ml_failure_lab run test-scaler-refit
python -m ml_failure_lab verify test-scaler-refit --approach bad
python -m ml_failure_lab verify test-scaler-refit --approach fixed
```

The `bad` verification exits with code 1, reporting that test data was transformed with test-specific parameters ($\mu = 11.0, \sigma = 1.0$) rather than training parameters ($\mu = 1.0, \sigma = 1.0$). The `fixed` verification exits with code 0. Tests verify that transformed test values match manual ground-truth calculations under relative and absolute tolerances of $10^{-5}$ across multiple shifted fixtures.

## Limits and reference

Reusing training transformers ensures consistent feature semantics between training and inference. However, consistent scaling does not eliminate covariate shift or guarantee that an estimator will generalize accurately to out-of-distribution inputs.

Reference: [Common pitfalls and recommended practices in scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html).
