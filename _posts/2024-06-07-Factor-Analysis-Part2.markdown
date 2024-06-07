---
layout: post
title:  "Understanding Factor Analysis: Implementation and Geometric Interpretation (Part 2)"
date:   2024-06-07 10:27:48 -0600
tags: [factor-analysis, statistics, PCA]
---


Continuing from the last post, in this post we'll look into (one of) the solutions of factor analysis, and a nice interpretation of that solution. Specifically, we'll show that the EM-algorithm is alternating between weighted-ridge regression during the E-step and standard linear regression in the dual (tranpose space) during the M-step.

### The model

Formally, we can write the model like so 

$$

\begin{align*}
\underbrace{Z}_{p \times n} = \underbrace{\Lambda F}_{(p \times q) (q \times n)} + \epsilon,
\end{align*}

$$

with the following assumptions:

 - $F \sim \text{MVN}(0, \mathbf{I}_{q\times q})$
 - $\epsilon \sim N(0, \Psi)$, $\Psi$ a positive diagonal matrix
 - $F, \epsilon$ uncorrelated

### EM Algorithm

To implement factor analysis, a natural choice is to use the Expectation Maximization (EM) algorithm. I say "natural" because often when we use EM, we invent "fake" data that we can't see (the latent variable). In this case, we already have the latent variable baked into the model.

To do EM, we need to derive the E-step and the M-step. The E-step involves finding the expression

$$

\begin{align*}
Q(\Lambda, \Psi|\Lambda^{(t)}, \Psi^{(t)}) = \mathbb{E}_{F \,|Z, \Lambda^{(t)}, \Psi^{(t)}}[\ell_{Z, F}(\Lambda, \Psi)].
\end{align*}

$$

Where $\ell_{Z, F}(\Lambda, \Psi)$ is the joint log-likelihood. Why do we want this? The EM algorithm is a recursive procedure where you have two sets of unknowns: one set is the latent data (in our case, this is $F$), and the other set would be the parameters of interest ($\Lambda$ and $\Psi$). To perform EM we start by guessing some parameter values, assuming they are fixed, and then we solve for the most likely value of $F$ given those parameter guesses. Then we assume that this $F$ is fixed, and we use it to estimate (or "update") the parameters. Then we use those new parameters to estimate $F$, and so on and so forth until we converge.

To solve for these steps, we're going to take advantage of the properties of conditional normal distributions. If we have two multivariate normal distributions $X_a \sim N(\mu_a, \Sigma_a)$ and $X_b \sim N(\mu_b, \Sigma_b)$, (with arbitrary dimensions) their joint distribution is given by 

$$

\begin{aligned}
\begin{pmatrix}X_a \\ X_b\end{pmatrix}\end{aligned} \sim N\left(\begin{bmatrix}\mu_a\\ \mu_b \end{bmatrix}, \begin{bmatrix}\ \Sigma_a & \Sigma_{ab}\\ \Sigma_{ba} & \Sigma_b \end{bmatrix}\right),


$$

where $\Sigma_{ab} = \Sigma_{ba}^T$ is the covariance between $X_a$ and $X_b$. Then the following holds for the conditional distributions:

$$

\begin{aligned}
p(X_a|X_b = x_b) \sim N\left(\mu_a + \Sigma_{ab}\Sigma_b^{-1}(x_b - \mu_b), \Sigma_a - \Sigma_{ab}\Sigma_b^{-1}\Sigma_{ba}\right)
\end{aligned},

$$

and similarly for $p(X_b\| X_a = x_a)$. Then we have the joint as

$$

\begin{aligned}\begin{pmatrix}Z \\ F\end{pmatrix}\end{aligned} \sim N\left(\begin{bmatrix}0\\ 0 \end{bmatrix}, \begin{bmatrix}\ \Lambda\Lambda^T + \Psi & \Lambda\\ \Lambda^T & I_{q} \end{bmatrix}\right).

$$

The covariance being $\Lambda$ follows from

$$

\begin{aligned}
\text{Cov}(Z, F) &= \text{Cov}(\Lambda F + \epsilon, F)\\[10pt]
&= \Lambda \text{Cov}(F, F) + \text{Cov}(\epsilon, F)\\[10pt]
&=\Lambda I_q + 0 = \Lambda.
\end{aligned}

$$

Then using the conditional distributions from above, we have that 

$$

\begin{aligned} 
p(Z|F) &\sim N(0 + \Lambda I_q (F - 0), (\Lambda\Lambda^T + \Psi) - \Lambda\Lambda^T)\\[10pt]
&=N(\Lambda F, \Psi).\\[10pt]
p(F|Z) &\sim N(0 + \Lambda^T (\Lambda\Lambda^T + \Psi)^{-1}(Z - 0), I_q - \Lambda^T(\Lambda\Lambda^T + \Psi)^{-1}\Lambda)\\[10pt]
&=N(\Lambda^T(\Lambda\Lambda^T + \Psi)^{-1}Z, I_q - \Lambda^T(\Lambda\Lambda^T + \Psi)^{-1}\Lambda).
\end{aligned}

$$

There are nice interpretations that come out of this that we'll look into a bit later. 

For notational convenience we'll shorten $\mathbb{E}_{F \,\|Z, \Lambda^{(t)}, \Psi^{(t)}}[\cdot]$ to just $\mathbb{E}[\cdot]$, but keep in mind that they are all conditional. Now, using the general fact that $p(X, Y) = p(X\|Y)p(Y)$, we have

$$

\begin{aligned}
Q(\Lambda, \Psi|\Lambda^{(t)}, \Psi^{(t)}) &= \mathbb{E}[\ell_{Z, F}(\Lambda, \Psi)]\\[10pt]
&=\mathbb{E}[\log p(Z, F; \Lambda, \Psi)]\\[10pt]
&=\mathbb{E}[\log p(Z|F; \Lambda, \Psi) +\log p(F;\Lambda, \Psi)]\\[10pt]
&=\mathbb{E}\left[-\frac{np}{2}\log(2\pi) - \frac{n}{2}\log|\Psi| -\frac{1}{2}\text{tr}\left\{(Z-\Lambda F)^T\Psi^{-1}(Z-\Lambda F)\right\}  +\log p(F;\Lambda, \Psi)\right]\\[10pt]
&=\mathbb{E}\left[-\frac{np}{2}\log(2\pi) - \frac{n}{2}\log|\Psi| - \frac{1}{2}\text{tr}\left\{(Z-\Lambda F)^T\Psi^{-1}(Z-\Lambda F)\right\}  -\frac{nq}{2}\log(2\pi) - \frac{n}{2}\log|I_q| - \frac{1}{2}(FF^T)\right]\\[10pt]
\implies&\mathbb{E}\left[-\frac{n}{2}\log|\Psi| - \frac{1}{2}\text{tr}\left\{(Z-\Lambda F)^T\Psi^{-1}(Z-\Lambda F)\right\} \right]\\[10pt]
&=-\frac{n}{2}\log|\Psi| -\frac{1}{2}\mathbb{E}\left[ \text{tr}\left\{Z^T\Psi^{-1}Z - F^T\Lambda^T\Psi^{-1}Z -Z^T\Psi^{-1}\Lambda F +F^T\Lambda^T\Psi^{-1}\Lambda F\right\} \right]\\[10pt]
&=-\frac{n}{2}\log|\Psi| - \frac{1}{2} \text{tr}\{Z^T\Psi^{-1}Z\} + \frac{1}{2}\underbrace{\text{tr}\left\{\mathbb{E}[F^T]\Lambda^T\Psi^{-1}Z\right\}}_{:=A} + \frac{1}{2}\underbrace{\text{tr}\left\{Z^T\Psi^{-1}\Lambda\mathbb{E}[F]\right\}}_{:= B} -\frac{1}{2}\mathbb{E}\underbrace{\left[\text{tr}\{F^T\Lambda^T\Psi^{-1}\Lambda F\} \right]}_{:=C}.\\[10pt]
\end{aligned}

$$

In the 6th line we get rid of any terms that don't have to do with $\Psi$ or $\Lambda$. We can again use the circular-shift invariance property of the trace to find that $A = B$, and we can rearrange $C$ to get 

$$

\begin{align*}
\frac{1}{2}\left(-n\log|\Psi| - \text{tr}\left\{Z^T\Psi^{-1}Z\right\} + 2\,\text{tr}\left(Z^T\Psi^{-1}\Lambda\mathbb{E}[F]\right) -\text{tr}\left\{\Lambda^T\Psi^{-1}\Lambda\mathbb{E}[FF^T]\right\}\right).
\end{align*}

$$

Now we need to take the derivatives of this expression with respect to $\Lambda$ and $\Psi$ and set the equation equal to zero to solve for our updated parameters.  Starting with $\Lambda$ and using equations 101 and 117 of the [matrix cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf):


$$

\begin{align*}
\frac{\partial}{\partial \Lambda} Q(\Lambda, \Psi|\Lambda^{(t)}, \Psi^{(t)}) &= 2 \Psi^{-1}Z\mathbb{E}[F]^T - 2\Psi^{-1}\Lambda \mathbb{E}[FF^T] \stackrel{\text{set}}{=} 0\\[10pt]
\implies \Lambda^{(t+1)} &= Z \mathbb{E}[F]^T(\mathbb{E}[FF^T])^{-1}.\\[10pt]
\end{align*}

$$


The $\Psi$ case is much easier if we differentiate with respect to the inverse:

$$

\begin{align*}
\frac{\partial}{\partial \Psi^{-1}} Q(\Lambda, \Psi|\Lambda^{(t)}, \Psi^{(t)}) &= n\Psi - ZZ^T + 2Z\mathbb{E}[F]^T\Lambda^T - \Lambda\mathbb{E}[FF^T]\Lambda^T,
\end{align*}

$$

then substituting the $\Lambda$ estimate we solved for earlier into the right-most term:

$$

\begin{align*}
\frac{\partial}{\partial \Psi^{-1}} Q(\Lambda, \Psi|\Lambda^{(t)}, \Psi^{(t)}) &= n\Psi - ZZ^T + 2Z\mathbb{E}[F]^T\Lambda^T - Z\mathbb{E}[F]^T\Lambda^T\\[10pt]
&= n\Psi - ZZ^T + Z\mathbb{E}[F]^T\Lambda^T\\[10pt]
\implies \Psi^{(t+1)} &=\frac{1}{n}\text{diag}\left(ZZ^T - Z\mathbb{E}[F]^T\Lambda^T\right).
\end{align*}

$$

Since $\Psi$ is constrained to be diagonal, we force the off-diagonals of this expression to be zero.

### What is EM doing?

Looking back to the introduction, PCA had a very nice interpretation: find a subspace that loses as little "information" (variance) as possible, and project onto it. With our constraint that the errors be uncorrelated, we lose that nice interpretation; see the graphs below, where PCA is on the left, and factor analysis is on the right. We can at least see that factor analysis tends to push points towards the "center", but otherwise doesn't seem very interpretable.

<p align="center">
    <img width="350" src="/images/PCA_final_plot.png"> <img width="350" src="/images/FA_final_plot.png">
</p>



So how can we describe what's going on? 

Recall from before that due to the conjugacy of the normal distribution, we have 

$$

\begin{align*}
p(F|Z) \sim N(\Lambda^T(\Lambda\Lambda^T + \Psi)^{-1}Z, I_q - \Lambda^T(\Lambda\Lambda^T + \Psi)^{-1}\Lambda).
\end{align*}

$$

For notational convenience, let's denote 

$$

\begin{align*}
\Gamma := \Lambda^T(\Lambda\Lambda^T + \Psi)^{-1},
\end{align*}

$$

so that we can reduce this expression down to 

$$

\begin{align*}
p(F|Z) \sim N(\Gamma Z, I_q - \Gamma\Lambda).
\end{align*}

$$

Then from this we immediately see that 

$$

\begin{align*}
\mathbb{E}_{F|Z}[F] &= \Gamma Z\\
\mathbb{E}_{F|Z}[FF^T] &=\mathbb{E}_{F|Z}[F]\mathbb{E}_{F|Z}[F]^T + \text{Var}_{F|Z}[F]\\
&= \Gamma Z Z^T \Gamma^T + I_q - \Gamma\Lambda. 
\end{align*}

$$

Now let's go back and examine our E and M steps more closely. 

The EM algorithm is essentially solving a problem with two unknowns by guessing <span style="color:#0077bb">one unknown</span> and holding it fixed, and then using that to estimate the <span style="color:#44aa99">other unknown</span>, holding *that* fixed, and re-estimating the<span style="color:#0077bb"> first unknown</span>, back and forth until convergence. Let's start in the M-Step and assume that $F$ is known, while we want to estimate $\Lambda$ and $\Psi$. This amounts to solving this equation (colored parameters are assumed *known*):

$$

\begin{align*}
\textcolor{#cc6677}{Z} = \Lambda\textcolor{#0077bb}{F} + \epsilon,
\end{align*}

$$

which is equivalent to solving

$$

\begin{align*}
\textcolor{#cc6677}{Z}^T = \textcolor{#0077bb}{F}^T\Lambda^T + \epsilon^T.
\end{align*}

$$

Call $Z_i$ the $i^{th}$ column of $Z$ and call $Z^{i}$ the $i^{th}$ row of $Z$. Then $p(Z_i\|F) \sim N(\Lambda F_i, \Psi)$. But since we assume $\Psi$ is diagonal, and since we assume that each column $Z_i$ is iid, this implies that $p(Z^{i}\|F) \sim N(F^T\Lambda^{i}, \psi_i \mathbf{I}_n)$, where $\psi_i$ is the $i^{th}$ diagonal entry of $\Psi$. In other words, we end up with a standard regression problem, and we will naturally find that the solution for $\Lambda^T$ is the least squares solution, $\Lambda^T = ((F^T)^T F^T)^{-1}(F^T)^TZ^T$, or $\Lambda = ZF^T(FF^T)^{-1}$.

Now let's examine the E-step, which assumes that we know $\Lambda$ and $\Psi$, and would like to solve for $F$. In other words, we would like to solve

$$

\begin{align*}
\textcolor{#cc6677}{Z} = \textcolor{#44aa99}\Lambda F + \epsilon.
\end{align*}

$$

This *also* appears to be a standard regression problem, so we might expect the solution to this to be $F = (\Lambda^T\Lambda)^{-1}\Lambda^T Z$, but we would be mistaken. This is not a standard regression problem because $F$ is random, not fixed, and because the errors are *not* constant. Since $F$ is random and we assume that marginally $F \sim  N(0, \mathbf{I}_q)$, this is essentially Bayesian regression, as we've placed a prior on $F$. Since [Bayesian regression is effectively ridge regression](https://statisticaloddsandends.wordpress.com/2018/12/29/bayesian-interpretation-of-ridge-regression/), this should be easily solvable. Let's start by simplifying our problem and assuming that $\Psi$ has a constant diagonal, $\sigma^2$.

Then we would [expect that our solution](https://en.wikipedia.org/wiki/Ridge_regression) for $F$ should be 

$$

\begin{align*}
F = (\Lambda^T \Lambda + \sigma^2 \mathbf{I}_q)^{-1}\Lambda^T Z.
\end{align*}

$$

However, if we scroll up, we'd find that in fact 

$$

\begin{align*}
F = \Lambda^T(\Lambda\Lambda^T + \sigma^2\mathbf{I}_p)^{-1}Z.
\end{align*}

$$

So our $\Lambda$'s are all swapped around and we have a $p$-dimensional identity in the inverse instead of a $q$-dimensional identity. The first equation comes directly from the conditional distribution, and the second equation comes from a widely understood ridge regression result, so it doesn't seem possible for either of them to be incorrect, so what's happening? It turns out that these are equal. Using equation 158 in the Matrix Cookbook (the Woodbury identity) and letting $P^{-1} = \sigma^2\mathbf{I}_q$, $B = \Lambda$, and $R = \mathbf{I}_p$, we have that:

$$

\begin{align}F &= (\Lambda^T \Lambda + \sigma^2 \mathbf{I}_q)^{-1}\Lambda^T Z\\[10pt]
&=\frac{1}{\sigma^{2}}\mathbf{I}_q \Lambda^T(\frac{1}{\sigma^2}\Lambda\Lambda^T + \mathbf{I}_p)^{-1}Z\\[10pt]
&=\Lambda^T(\Lambda\Lambda^T + \sigma^2\mathbf{I}_p)^{-1}Z.

\end{align}

$$

But the issue is that this clearly only holds when $\Psi$ is constant diagonal. So what do we do in our more general case where $\Psi$ can have a non-constant diagonal? Same thing we would do in linear regression when we have heteroskedasticity: whiten our errors by using weighted least squares. The solution to weighted least squares is given by 

$$

\begin{align*}
\hat \beta_{WLS} = (X^TWX)^{-1}X^TWY,
\end{align*}

$$

where $W$ is our precision matrix, i.e., the inverse of the covariance matrix of the errors. Then if we added in ridge regression, we might expect the solution to look like 

$$

\begin{align*}
\hat \beta_{WLS\text{-Ridge}} = (X^TWX + \mathbf{I})^{-1}X^TWY.
\end{align*}

$$

Can we get our estimate of $F$ there? First we define $\tilde \Lambda = \Psi^{-1/2}\Lambda$. Then, 


$$

\begin{align}
F &= \Lambda^T(\Lambda\Lambda^T + \Psi)^{-1}Z\\[10pt]
&= \Lambda^T\Psi^{-1/2}\Psi^{1/2}(\Lambda\Lambda^T + \Psi)^{-1}\Psi^{1/2}\Psi^{-1/2}Z\\[10pt]
&=\Lambda^T\Psi^{-1/2}(\Psi^{-1/2}\Lambda\Lambda^T\Psi^{-1/2} + \mathbf{I}_p)^{-1}\Psi^{-1/2}Z \\[10pt]
&= \tilde \Lambda^T(\tilde \Lambda \tilde \Lambda^T + \mathbf{I}_p)^{-1}\Psi^{-1/2}Z\\[10pt]
&= (\tilde \Lambda^T \tilde \Lambda + \mathbf{I}_q)^{-1}\tilde \Lambda\Psi^{-1/2}Z \\[10pt]
&= (\Lambda^T \Psi^{-1}\Lambda + \mathbf{I}_q)^{-1}\Lambda\Psi^{-1}Z.
\end{align}

$$

Which matches what we "guessed" our solution might be for a weighted ridge regression problem.

Main point: The E-step is weighted ridge regression and the M-step is standard linear regression in the dual (transposed) space. The animation below shows the E-step on the left and the M-step on the right. In the first step, we guess a <span style="color:#44aa99">subspace</span> and ridge regress onto it to get the <span style="color:#0077bb">coordinates</span>. Then we move to M-step and perform standard linear regression of the <span style="color:#cc6677">scores</span> onto the <span style="color:#0077bb">coordinates</span> to get a new <span style="color:#44aa99">subspace</span> that we use in the E-step to update our <span style="color:#0077bb">coordinates</span>, and so on and so forth until convergence.

<img src="/images/FA_anim.gif">

We're cheating a *little* bit here with the visual; in the M-step we're watching one of two linear regressions (the other one would be History scores onto the factor), whereas in the E-step we're actually watching *all* of the ridge regressions happening simultaneously.

This is because our variable $Z$ is $2 \times 20$, so we have $20$ individual ridge regressions with $2$ points happening simultaneously in the E-step, so it wouldn't be very interesting to look at any one of those ridge regressions individually.

