# Loss Functions

## In Progress🚧

![Cover](cover.jpg)

*Note: the concept of loss functions is closely related to that of [Similarity Metrics](https://github.com/intelligent-username/Similarity-Metrics).*

Loss arises whenever performance falls short of expectation. In predictive applications, it is essential to assess the reliability of the estimator. A **loss function** provides a quantitative rule for measuring how far predictions diverge from observed outcomes. By minimizing the loss, we guide the learning process of a predictive system toward parameters that yield greater accuracy.

## The Functions

\*Note: from now on, $N$ indicates the number of samples, $y_i$ the true value, and $\hat{y}_i$ the predicted value. Any other notation will be defined as needed.

### 1. Mean Squared Error (MSE)

$$L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

Mean Squared Error (MSE) is among the most widely recognized loss functions. For a set of $N$ points, it calculates the *average* squared deviation of predictions from the actual values. This loss is standard in regression, statistics, and advanced machine learning. The squaring of differences prevents positive and negative errors from canceling out. Being both quadratic and differentiable, it is smooth and convex, with a single global minimum that represents the optimal model parameters. Performing gradient descent on MSE is straightforward.

**Application:** Primarily used in regression, forecasting, and reinforcement learning. When errors are roughly Gaussian, MSE is optimal as it corresponds to maximizing the likelihood under a normal error model.

---

### 2. Mean Absolute Error (MAE)

$$L = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

Similar to MSE, but without squaring the errors. As a consequence, MAE is often more accurate, (less sensitive to outliers), and provides a more intuitive average error. MAE is also closer to the median of the error, because absolute deviation weights all residuals equally and balances them symmetrically around a central point.

However, since the absolute value function is not differentiable at zero, optimization is harder, although micro-gradients can be employed.

**Application:** Suitable for cases where noise is irregular and outliers are more common.

---

### 3. Quantile Loss

$$
L_i =
\begin{cases}
\tau \cdot |y_i - \hat{y}_i|, & y_i \ge \hat{y}_i \\
(1-\tau) \cdot |y_i - \hat{y}_i|, & y_i < \hat{y}_i
\end{cases}
$$

or more concisely:

$$L_\tau = \sum_{i=1}^N \max\big(\tau(y_i - \hat{y}_i), (1-\tau)(\hat{y}_i - y_i)\big)$$

- Where $\tau \in (0, 1)$ is a parameter that determines the quantile being estimated. I.e. if $\tau = 0.9$, we are estimating the 90th percentile.

Quantile loss is a generalization of MAE , it can be tuned asymmetrically in order to give preference to over-or-under-predictions based on the domain-specific context. Note that, when $\tau = 0.5$, Quantile Loss is just the same as MAE.

This is called *quantile* loss because it specifically targets a certain range (i.e. quantile) of predictions by adjusting the penalty. Once again, if we set $\tau = 0.9$, the loss function will penalize under-predictions (where $\hat{y}_i < y_i$) more heavily than over-predictions. In other words, we would be making predictions in the "over-estimation" interval. This is especially useful in Gradient Boosting, where component-based guesses are adjusted iteratively based on the range of previous residuals.

Another important application of this is quantile regression, wherein we estimate a range rather than direct value or mean.

---

### 4. Huber Loss

$L_\delta =
\begin{cases}
\tfrac{1}{2}(y_i - \hat{y}_i)^2 & \text{if } |y_i - \hat{y}_i| \le \delta \ \\
\delta|y_i - \hat{y}_i| - \tfrac{1}{2}\delta^2 & \text{otherwise}
\end{cases}$

with $\delta > 0$.

MSE and MAE are both useful. However, they each have their own shortcomings: MSE is sensitive to outliers and MAE is not differentiable everywhere. Huber Loss solves these problems.

For small errors, i.e. $|y_i - \hat{y}_i| \le \delta$, Huber Loss behaves like MSE, being quadratic and smooth. For larger errors, it switches to a linear form like MAE, reducing the influence of outliers. The (hyper)parameter $\delta$ controls the threshold between these two regimes.

**Application:** Stronger curve fitting, sensor calibration, and financial modelling.

---

### 5. Log Loss

$$L = -\frac{1}{N} \sum_{i=1}^N \left[y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)\right]$$

with

- $y_i \in \{0, 1\}$ being the true binary label, and
- $\hat{y}_i \in (0, 1)$ being the predicted probability of the positive class.
- $\log_n$ is assumed to have the base $e$, but bases 2 or 10 can also be used, changing the scale but not the relative ordering of losses.

Also called Binary Cross-Entropy, log Loss is derived from information theory as the negative log-likelihood of a Bernoulli distribution. It measures how well the predicted probabilities align with the actual outcomes. Confident but wrong predictions receive a higher penalty than uncertain ones.

**Application:** Log loss is absolutely essential to logistic regression, binary classification, and probabilistic forecasting.

---

### 6. Categorical Cross-Entropy

Categorical Cross-Entropy generalizes Log Loss to multiple classes:

$$L = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(\hat{y}_{i,c})$$

with

- $y_{i,c} \in {0,1}$ indicates whether sample $i$ belongs to class $c$ (i.e. exactly one entry is 1, the rest are 0).
- $\hat{y}_{i,c} \in (0, 1)$ being the *predicted* probability for sample $i$ to belong to class $c$.

Now, since $$\sum_{c=1}^C y_{i,c} = 1$$ for each sample $i$, we can simplify to:

$$
L = -\frac{1}{N} \sum_{i=1}^N \log(\hat{y}_{i,k^\ast})
$$

This time, unlike binary cross-entropy, we take the negative log-likelihood of a categorical distribution. Once again, $log_n$ is assumed to have the base $e$, and more confident wrong predictions are punished more heavily.

**Application:** Used in multi-class (discrete) classification.

---

### 7. Focal Loss

With other cross-entropy losses, we may have a disproportionate number of examples that are 'easy' to predict, and these may dominate the loss values. Focal Loss modifies standard cross-entropy by adding a modulating factor.

$$L = -\frac{1}{N} \sum_{i=1}^N \big[\alpha (1 - \hat{y}_i)^\gamma y_i \log(\hat{y}_i) + (1-\alpha) \hat{y}_i^\gamma (1-y_i) \log(1-\hat{y}_i)\big]$$

With

- $\alpha \in [0,1]$ balancing the importance of positive/negative examples.
- $\gamma \ge 0$ focusing learning on hard examples (by reducing the loss contribution from easy ones). The higher $\gamma$ is, the more emphasis is placed on hard examples.
- $\hat{y}_i$ is the predicted probability for the true class.

When $\gamma = 0$, Focal Loss reduces to standard cross-entropy.

Notice that, with this added term, with an 'easy' class (i.e. a high confidence prediction), $\hat{y}_i$ will be close to 1, and the modulating factor $(1-\hat{y}_i) ^ \gamma$, will approach zero. Conversely, poorly predicted samples (small $\hat{y}_i$) yield a factor near one, leaving their loss unaffected and thus emphasized during training.

**Application:** Essential for enforcing higher accuracy on difficult examples, specifically in imbalanced classification problems, like medical diagnosis.

---

### 8. Hinge Loss

Support Vector Machines help classification models generalize better by maximizing the margin between classes. It is mainly trained using Hinge Loss.

$$L = \frac{1}{N} \sum_{i=1}^N \max(0, 1 - y_i \hat{y}_i)$$

Where:

- $(y_i \hat{y}_i)$ is the 'margin'.
- If $(y_i \hat{y}_i \ge 1)$, the sample is classified correctly with a safe margin, and loss = 0.
- If $(y_i \hat{y}_i < 1)$, either the sample is misclassified or classified with insufficient confidence, and loss increases linearly.

&nbsp; We take the maximum of zero and the margin-based term to ensure that only misclassified or low-confidence samples contribute to the loss. It penalizes points that are close to or on the wrong side of the decision boundary. Unlike cross-entropy, it does not push probabilities toward 1, but pushes the decision boundary so that classes are separated by the largest possible margin. This helps improve robustness against noise and overfitting. Minimizing Hinge Loss naturally leads to maximizing this margin, which improves generalization.

---

### 9. Centre Loss

$$L = \tfrac{1}{2} \sum_{i=1}^N | x_i - c_{y_i} |_2^2$$

Where:

- $x_i$ is the feature representation of the $i$-th sample.
- $c_{y_i}$ is the centre (average embedding) of the class to which the $i$‑th sample belongs.
- $|\cdot|_2$ denotes the Euclidean distance.

Center Loss is used to encourage a model to make features of the same class cluster together in feature space. When working with deep learning tasks, input features are often embedded as vectors and used to make predictions. Often times, classification loss is used, but, when we need specificity (such as recoginizing a specific face, not just a 'human face'), we need to ensure that features of the same class are close together.

The loss measures how far each feature vector is from its class center and penalizes larger distances. Minimizing Center Loss forces samples from the same class to cluster tightly around their center, reducing intra‑class variance.
This is often combined with a standard classification loss from earlier to ensure both inter-class separability and intra-class compactness.

**Application:** Crucial in deep feature learning tasks such as image retrieval.

---

### 10. Negative Log Likelihood (NLL)

$L = -\frac{1}{N} \sum_{i=1}^N \log P(y_i \mid x_i; \theta)$

$P(y_i \mid x_i; \theta)$ is the predicted probability of the true label under model parameters $\theta$. This is the general log-likelihood loss, of which binary and categorical cross-entropy are special cases.

**Application:** Standard in probabilistic classification and likelihood-based inference.

---

### 11. Cosine Similarity Loss

Cosine similarity between vectors $u$ and $v$ is:

$\cos(u, v) = \frac{u \cdot v}{|u||v|}$

The associated loss (Cosine Embedding Loss) is:

$L = 1 - \frac{y \cdot \hat{y}}{|y||\hat{y}|}$

This loss emphasizes angular alignment, penalizing directional deviation more than magnitude differences.

**Application:** Widely used in embedding spaces, retrieval systems, and representation learning.

---

### 12. Jaccard Loss (IoU Loss)

The Jaccard Index, or Intersection over Union (IoU), for sets $A$ and $B$ is:

$J(A,B) = \frac{|A \cap B|}{|A \cup B|}$

As a loss:

$L = 1 - J(A,B)$

This measures set overlap, penalizing mismatch between prediction and ground truth.

**Application:** Common in segmentation and detection tasks requiring spatial overlap accuracy.

---

### **Which Loss Functions to Know and Use**

- **Regression Tasks:** MSE, MAE, Quantile, Huber
- **Binary Classification:** Log Loss (Binary Cross-Entropy)
- **Multi-Class Classification:** Categorical Cross-Entropy

## Installation & Setup

1. Create and activate a Python environment (3.9+ recommended).
2. Install core dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

   Or use conda:

   ```powershell
   conda env create -f env.yml
   conda activate loss-functions
   ```

3. Run tests to verify implementations:

   ```powershell
   pytest tests/
   ```

4. Open any notebook inside `demos/` to begin prototyping and documenting examples.

Feel free to replace or expand upon these steps as the draft matures.

This repository is under the [MIT License](LICENSE).
