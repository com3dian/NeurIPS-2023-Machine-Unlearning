This folder contains the metric paper mentioned in the competition's homepage.

![metric](../pictures/metric.svg)

## Notes on the metric paper

| Notations | Meanings                  |
| --------- | ------------------------- |
| A         | training algorithm        |
| D         | datasets                  |
| R         | space of network's output |
| FPR       | Fasse positive rate       |
| FNR       | False negeative rate      |
| S         | Forgotten set             |

### 2. Background: Differential Privacy

**Definition: Example-level differential privacy (DP)**: A training algorithm A : D → R is (ε, δ) example-level DP if for all pairs of datasets D and D ′ from D that differ by addition or removal of any **single** training example and all output regions R ⊆ R:
$$
Pr[A(D) ∈ R] ≤ e^ε Pr[A(D' ) ∈ R] + δ.               \quad \quad (1)
$$
which means, 
$$
Pr[A(D) ∈ R] \rightarrow Pr[A(D ′ ) ∈ R] \quad \quad (2)
$$

Base on (1), a estimation of the $\epsilon$ at a fixed δ is 
$$
\hat \epsilon = max\{log(\frac{1 - \delta - \hat FPR}{\hat FNR}), \frac{1 - \delta - \hat FNR}{\hat FPR})\} \quad \quad (3)
$$
where $\hat FPR$ and $\hat FNR$ estimates of the true FPR and FNR under an instantiated membership inference attack* that can inspect the model trained with DP in a black- or white-box fashion and attempts to infer whether the model was trained on D or D′.

**Definition: Group-level Differential privacy**: A training algorithm A : D → R is (ε, δ, k) group-level DP if for all pairs of datasets D and D ′ from D that differ by addition or removal of up to k training examples and all output regions R ⊆ R:
$$
Pr[A(D) ∈ R] ≤ e^ε Pr[A(D') ∈ R] + δ.
$$


### 3. Defining Machine Unlearning
