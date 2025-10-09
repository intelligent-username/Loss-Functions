# Loss Functions

![Cover](cover.jpg)

*Note: the concept of loss functions is closely related to that of [Similarity Metrics](https://github.com/intelligent-username/Similarity-Metrics).*

Loss arises whenever performance falls short of expectation. In predicive applications, it is essential to assess the reliability of the estimator. A **loss function** provides a quantitative rule for measuring how far predictions diverge from observed outcomes. By minimizing the loss, we guide the learning process of a predictive system toward parameters that yield greater accuracy.

- [Loss Functions](#loss-functions)
  - [The Functions](#the-functions)
    - [1. Mean Squared Error (MSE)](#1-mean-squared-error-mse)
    - [2. Mean Absolute Error (MAE)](#2-mean-absolute-error-mae)
    - [3. Quantile Loss](#3-quantile-loss)
    - [4. Huber Loss](#4-huber-loss)
    - [5. Log Loss](#5-log-loss)
    - [6. Categorical Cross-Entropy](#6-categorical-cross-entropy)
    - [7. Negative Log Likelihood (NLL)](#7-negative-log-likelihood-nll)
    - [8. Focal Loss](#8-focal-loss)
    - [9. Centre Loss](#9-centre-loss)
    - [10. Hinge Loss](#10-hinge-loss)
    - [11. Cosine Similarity Loss](#11-cosine-similarity-loss)
    - [12. Jaccard Loss](#12-jaccard-loss)
  - [Use Cases](#use-cases)
  - [Installation \& Setup](#installation--setup)
  - [License](#license)

## The Functions

\*Note: from now on:

- $N$ is the number of samples
- $y_i$ is the true value, and
- $\hat{y}_i$ (or anything else with a h$\hat{a}$t) is the predicted value.
- $\log_n$ is assumed to have the base $e$, but bases 2 or 10 can also be used, changing the scale but not the relative ordering of losses.

- 'Robustness' refers to resilience against outliers.

Any other notation will be defined as needed.

---

### 1. Mean Squared Error (MSE)

$$L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

Mean Squared Error (MSE) is among the most widely recognized loss functions. For a set of $N$ points, it calculates the *average* squared deviation of predictions from the actual values. This loss is standard in regression, statistics, and advanced machine learning. The squaring of differences prevents positive and negative errors from canceling out. Being both quadratic and differentiable, it is smooth and convex, with a single global minimum that represents the optimal model parameters. Performing gradient descent on MSE is straightforward.

**Application:** Primarily used in regression, forecasting, and reinforcement learning. When errors are roughly Gaussian, MSE is optimal as it corresponds to maximizing the likelihood under a normal error model.

---

### 2. Mean Absolute Error (MAE)

$$L = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

Similar to MSE, but without squaring the errors. As a consequence, MAE is often more robust, and provides a more intuitive average error. MAE is also closer to the median of the error, because absolute deviation weights all residuals equally and balances them symmetrically around a central point.

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

for individual losses
The total loss is:

$$L_\tau = \sum_{i=1}^N \max\big(\tau(y_i - \hat{y}_i), (1-\tau)(\hat{y}_i - y_i)\big)$$

- Where $\tau \in (0, 1)$ is a parameter that determines the quantile being estimated. I.e. if $\tau = 0.9$, we are estimating the 90th percentile.

Quantile loss is a generalization of MAE, it can be tuned asymmetrically in order to give preference to over-or-under-predictions based on the domain-specific context. Note that, when $\tau = 0.5$, Quantile Loss is just the same as MAE.

This is called *quantile* loss because it specifically targets a certain range (i.e. quantile) of predictions by adjusting the penalty. Once again, if we set $\tau = 0.9$, the loss function will penalize under-predictions (where $\hat{y}_i < y_i$) more heavily than over-predictions. In other words, predictions are biased towards overestimation. Decreasing $\tau$ increases the penalty for over-predictions, biasing them towards underestimation. This is especially useful in Gradient Boosting, where component-based guesses are adjusted iteratively based on the range of previous residuals.

Another important application of this is quantile regression, wherein we estimate a range rather than direct value or mean.

---

### 4. Huber Loss

$$
L_\delta =
\begin{cases}
\tfrac{1}{2}(y_i - \hat{y}_i)^2 & \text{if } |y_i - \hat{y}_i| \le \delta \ \\
\delta|y_i - \hat{y}_i| - \tfrac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

with $\delta > 0$.

MSE and MAE are both useful. However, they each have their own shortcomings: MSE is sensitive to outliers and MAE is not differentiable everywhere. Huber Loss solves these problems.

For small errors, i.e. $|y_i - \hat{y}_i| \le \delta$, Huber Loss behaves like MSE, being quadratic and smooth. For larger errors, it switches to a linear form like MAE, reducing the influence of outliers. The (hyper)parameter $\delta$ controls the threshold between these two regimes.

**Application:** Stronger curve fitting, sensor calibration, and financial modelling.

---

### 5. Log Loss

$$L = -\frac{1}{N} \sum_{i=1}^N \left[y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)\right]$$

with

- $y_i \in$  {$0, 1$} being the true binary label, and
- $\hat{y}_i \in (0, 1)$ being the predicted probability of the positive class.

Also called Binary Cross-Entropy, log Loss is derived from information theory as the negative log-likelihood of a Bernoulli distribution. It measures how well the predicted probabilities align with the actual outcomes. Confident but wrong predictions receive a higher penalty than uncertain ones.

**Application:** Log loss is absolutely essential to logistic regression, binary classification, and probabilistic forecasting.

---

### 6. Categorical Cross-Entropy

Categorical Cross-Entropy generalizes Log Loss to multiple classes:

$$L = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(\hat{y}_{i,c})$$

with

- $y_{i,c} \in \{0,1\}$ indicates whether sample $i$ belongs to class $c$ (i.e. exactly one entry equals $1$, the rest are $0$).
- $\hat{y}_{i,c} \in (0, 1)$ being the *predicted* probability for sample $i$ belonging to class $c$.

Now, since $$\sum_{c=1}^C y_{i,c} = 1$$ for each sample $i$, we can simplify to:

$$
L = -\frac{1}{N} \sum_{i=1}^N \log(\hat{y}_{i,k^\ast})
$$

This time, unlike binary cross-entropy, we take the negative log-likelihood of a categorical distribution.

**Application:** Used in multi-class (discrete) classification.

---

### 7. Negative Log Likelihood (NLL)

NLL is the general log-likelihood loss. Binary and categorical cross-entropy are special cases of it.

$$L = -\frac{1}{N} \sum_{i=1}^N \log P(y_i \mid x_i; \theta)$$

Where:

- $P(y_i \mid x_i; \theta)$ is the predicted probability of the true label under model parameters $\theta$.
- Since probabilities lie in $(0, 1)$, they have negative logs. The minus sign in front negates this to make the loss a positive value.

Negative Log-Likelihood measures how confident the model is in its predictions, and punishes it for being confidently wrong. Correct classes should be predicted with a probability of $1$, and all others should have a $0$. The negative log-likelihood captures how far off this ideal scenario is by taking the logarithm of the probability assigned to the correct class, and negating it. The higher this probability, the better the model. Then, we divide by $N$ to find the average loss. Thus, maximizing the likelihood (accuracy) means minimizing the negative log-likelihood.

As a result, **NLL** encourages models to predict the right class with calibrated confidence. For example, predicted $0.95$ for the true class would give a loss of only $-log(0.95)$, which is relatively small, whereas predicting $0.05$ would give a loss that is tremendously large. In Bayesian terms, it measures how well the predicted distribution matches the empirical distribution. Models minimizing NLL become increasingly reliable in their probabilistic interpretation — their predicted probabilities begin to reflect actual likelihoods observed in data.

This loss function is useful in more general classification tasks.

---

### 8. Focal Loss

With other cross-entropy losses, we may have a high proportion of examples that are 'easy' to predict, which may dominate the loss values. Focal Loss modifies standard cross-entropy by adding a modulating factor, $(1-\hat{y}_i)^\gamma$.

$$
L = -\frac{1}{N} \sum_{i=1}^N \big[\alpha (1 - \hat{y}_i)^\gamma y_i \log(\hat{y}_i) + (1-\alpha) \hat{y}_i^\gamma (1-y_i) \log(1-\hat{y}_i)\big]
$$

With

- $\alpha \in [0,1]$ balancing the importance of positive/negative examples.
- $\gamma \ge 0$ focusing learning on hard examples (by reducing the loss contribution from easy ones). The higher $\gamma$ is, the more emphasis is placed on hard examples.
- $\hat{y}_i$ is the predicted probability for the true class.

When $\gamma = 0$, Focal Loss reduces to standard cross-entropy.

Notice that, with this added term, with an 'easy' class (i.e. a high confidence prediction), $\hat{y}_i$ will be close to 1, and the modulating factor $(1-\hat{y}_i) ^ \gamma$, will approach zero. Conversely, poorly predicted samples (small $\hat{y}_i$) yield a factor near one, leaving their loss unaffected and thus emphasized during training.

Focal loss, in essence, is used for enforcing higher accuracy on difficult examples. For example, on medical diagnoses.

---

### 9. Centre Loss

   $$L = \tfrac{1}{2} \sum_{i=1}^N | x_i - c_{y_i} |_2^2$$

Where:

- $x_i$ is the feature representation of the $i$-th sample.
- $c_{y_i}$ is the centre (average embedding) of the class to which the $i$‑th sample belongs.
- $|\cdot|_2$ denotes the Euclidean distance.

Centre Loss is used to encourage a model to make features of the same class cluster together in feature space. When working with deep learning tasks, input features are often embedded as vectors and used to make predictions. Oftentimes, classification loss suffices, but when we need *specificity* (such as recognizing a specific face, not just a 'human face'), we need to ensure that features of the same class are close together.

The loss measures how far each feature vector is from its class centre and penalizes larger distances. Minimizing Centre Loss forces samples from the same class to cluster tightly around their centre, reducing intra‑class variance.
This is often combined with a standard classification loss from earlier to ensure both inter-class separability and intra-class compactness.

**Application:** Crucial in deep feature learning tasks such as image retrieval.

---

### 10. Hinge Loss

Support Vector Machines help classification models generalize better by maximizing the margin between classes. It is mainly trained using Hinge Loss.

$$L = \frac{1}{N} \sum_{i=1}^N \max(0, 1 - y_i \hat{y}_i)$$

Where:

- $y_i \in \{-1, 1\}$.
- $\hat{y}_i \in (0, 1)$.
- $(y_i \hat{y}_i)$ is the 'margin'.
- If $(y_i \hat{y}_i \ge 1)$, the sample is classified correctly with a safe margin, and loss = 0.
- If $(y_i \hat{y}_i < 1)$, either the sample is misclassified or classified with insufficient confidence, and loss increases linearly.

&nbsp; We take the maximum of zero and the margin-based term to ensure that only misclassified or low-confidence samples contribute to the loss. It penalizes points that are close to or on the wrong side of the decision boundary. Unlike cross-entropy, it does not push probabilities toward 1, but pushes the decision boundary so that classes are separated by the largest possible margin. This helps improve robustness against noise and overfitting. Minimizing Hinge Loss naturally leads to maximizing this margin, which improves generalization.

---

### 11. Cosine Similarity Loss

Often times, we represent features (for example images) numerically as vectors, and we want to find which two have the most similar direction. For this, we use the Cosine Similarity

$$\cos(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{|\vec{u}||\vec{v}|}$$

Where $\vec{u}$ and $\vec{v}$ are vector representations of our data. We want this similarity to be as high as possible so, in order to get a 'loss' to minimize, we create The Cosine Embedding Loss:

$$L = 1 - \frac{\vec{y} \cdot \vec{\hat{y}}}{|\vec{y}||\vec{\hat{y}}|}$$

Where

- $\vec{y}$ is the first vector (or, in 'loss terms', the 'actual' output)
- $\vec{\hat{y}}$ is the second vector (or, in 'loss terms', the 'predicted' output)
- $\vec{y} \cdot \vec{\hat{y}}$: the dot product measures directional alignment.
- $|\vec{y}||\vec{\hat{y}}|$: vector magnitudes ensure normalization

Notice that, when $\vec{y}$ = $\vec{\hat{y}}$, the cosine between them is 1, i.e. the prediction is $100$% accurate, so the loss is $0$. If they are perpendicular, Loss is 1, and if they point in opposite directions, loss is 2.

In this form of loss, we only care about angular similarity (directional overlap), and ignore magnitude.

**Application:** Widely used in embedding spaces (e.g. image recognition) and retrieval systems (e.g. k-nearest neighbours).

---

### 12. Jaccard Loss

Also called Intersection **Over Union Loss**.

Jaccard loss measures how much two sets overlap compared to their total combined area. In other words, it measures the overlap between predicted and true regions, how much they overlap divided by how much they cover together. For example, if the overlap is $100$%, then all predictions are correct.

**IoU** Score is measured by:

$$J(A,B) = \frac{|A \cap B|}{|A \cup B|}$$

Where $A$ is the set of true values $y_1, ..., y_N$, and $B$ is the set of predicted values $\hat{y}_1, ..., \hat{y}_N$.

And written as a loss as:

$$L_J = 1 - J(A,B) = 1 -  \frac{\sum_{i=1}^N (y_i\cdot\hat{y}_i)} {\sum_{i=1}^N(y_i + \hat{y}_i - y_i \cdot \hat{y}_i)}$$

Where:

$$y_i =
\begin{cases}
1, & \text{if  } i \text{ belongs to the true object}\\
0, & \text{otherwise}
\end{cases}
$$

and

$$
\hat{y}_i =
\begin{cases}
1, & \text{if predicted as object}\\
0, & \text{otherwise.}
\end{cases}
$$

For soft predictions, $\hat{y}_i$ may lie in $[0,1]$.

The numerator (dot product) counts overlapping predictions. The denominator takes the union of all positive predictions.
Here, we don't rescale, so the range for the loss function is $[0, 2]$.

Unlike MSE and cross-entropy, IoU considers global area overlap. That's why IoU directly measures geometric precision ($\frac{\text{True positives}}{\text{Total elements}}$).

IoU loss is used in image segmentation (i.e. finding which two pictures are of the same object) and other high accuracy overlap-based tasks.

---

## Use Cases

- For **Regression Tasks**, use MSE, MAE, Huber Loss, or Quantile Loss, depending on the context.
- For **Classification**, use Log Loss, Categorical Cross-Entropy, Focal Loss, or Negative Log-Likelihood depending on the context.
- For **Specialized Tasks**, use one of the others, like cosine similarity for embeddings, or Jaccard loss for segmentation, or make one of your own.

---

## Installation & Setup

1. Create and activate a local/virtual Python environment (or just download the dependencies straight to your system).

2. Install core dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

   Or use conda:

   ```powershell
   conda env create -f env.yml
   conda activate loss-functions
   ```

3. Run tests (if needed):

   ```powershell
   pytest tests/
   ```

4. Use any notebook inside `demos/` to begin prototyping the loss functions.

Feel free to replace or expand upon these demos.

---

## License

This repository is under the [MIT License](LICENSE).
