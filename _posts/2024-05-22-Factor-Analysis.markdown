---
layout: post
title:  "Understanding Factor Analysis: Relationship to PCA (Part 1)"
date:   2024-05-24 10:27:48 -0600
tags: [factor-analysis, statistics, PCA]
---

## Intoduction

I spent a good chunk of time trying to understand factor analysis as fundamentally as possible. There are plenty of stack exchange posts, notes, and books that explain factor analysis and its relationship to PCA, but (I felt that) none motivated it from the ground up or explained the geometric intuitions particularly well.

I'm writing this as the article I *wished* I stumbled upon while learning. It is not a complete treatment of factor analysis, but is hopefully a good resource for anyone who thinks about these things like me (a psychopath).

Really, this first part is hardly about factor analysis. It goes over a classic factor analysis problem, aims to formulate the problem statement very generally, goes through what an intuitive solution might be, and then shows why that ultimately runs into an issue. The next post will be about its implementation and a nice geometric interpretation of that implementation.

## What is it?

Factor analysis is a dimensionality reduction technique that aims to relate a number $\color{#CC6677}{p}$ of <span style="color:#cc6677">observed variables</span> to a smaller number $\color{#0077bb}{q}$ of <span style="color:#0077bb">unobserved factors</span> through<span style="color:#44aa99"> factor loadings</span>. The problem is similar in nature to both linear regression and principal component analysis, but with a few subtle but important differences.

Factor Analysis was pioneered by Spearman in 1904 in his paper *'General Intelligence': Objectively Determined and Measured.* The gist of it (as much as you can get the gist of a 90 page paper) can be understood through an example with test scores. Highschool <span style="color:#cc6677">exam scores</span> are generally pretty highly correlated across subjects (students who do well in one area tend to do well in other areas), so we might posit that there's some sort of underlying <span style="color:#0077bb">"general intelligence"</span> (in Spearman's words) that can explain how these tests scores are related. We could set up a corresponding factor model for the $i^{th}$ student like so:

$$

\begin{align*}
    \textcolor{#CC6677}{\text{math}}_{i}&= \textcolor{#0077bb}{\text{IQ}}_i\cdot \textcolor{#44aa99}{w_{\text{math}}} + \epsilon_{\text{math}}\\
    \textcolor{#CC6677}{\text{english}}_{i}&= \textcolor{#0077bb}{\text{IQ}}_i\cdot \textcolor{#44aa99}{w_{\text{eng}}} + \epsilon_{\text{eng}}\\
    \textcolor{#CC6677}{\text{history}}_{i}&= \textcolor{#0077bb}{\text{IQ}}_i\cdot \textcolor{#44aa99}{w_{\text{hist}}} + \epsilon_{\text{hist}}\\
\end{align*}

$$

Here, the <span style="color:#cc6677">exam scores</span> for the three subjects would be our <span style="color:#cc6677">observed variables</span>, and the <span style="color:#0077bb">general intelligence</span> of the student (or <span style="color:#0077bb">IQ</span> for short) would be our <span style="color:#0077bb">unobserved factor</span>. The<span style="color:#44aa99"> weights</span>, $\color{#44aa99}{w}$, represent the <span style="color:#44aa99">factor loadings</span>, which tell us how strong the relationship between the factor and variable is. We also include possible errors, represented by $\epsilon$. Each subject could have a different level of variance in the error, but a **key** assumption of factor analysis is that the errors are *not* correlated across subjects. 

In general, we can have more than one factor. So suppose that we also notice that students who do well in one physical activity (say, basketball) tend to do well in others (say, football), and suppose we have some way to measure this. Then we might extend our model as such:

$$

\begin{align*}
    \textcolor{#CC6677}{\text{math}}_{i}&= \textcolor{#0077bb}{\text{IQ}}_i\cdot \textcolor{#44aa99}{w_{\text{math}}} + \epsilon_{\text{math}}\\
    \textcolor{#CC6677}{\text{english}}_i &= \textcolor{#0077bb}{\text{IQ}}_i \cdot \textcolor{#44aa99}{w_{\text{eng}}} + \epsilon_{\text{eng}}\\
    \textcolor{#CC6677}{\text{history}}_i&= \textcolor{#0077bb}{\text{IQ}}_i \cdot \textcolor{#44aa99}{w_{\text{hist}}} + \epsilon_{\text{hist}}\\
    \textcolor{#CC6677}{\text{basketball}}_i &= \textcolor{#0077bb}{\text{AQ}}_i \cdot \textcolor{#44aa99}{w_{\text{bb}}}+ \epsilon_{\text{bb}}\\
    \textcolor{#CC6677}{\text{football}}_i &= \textcolor{#0077bb}{\text{AQ}}_i \cdot \textcolor{#44aa99}{w_{\text{fb}}}+ \epsilon_{\text{fb}}\\
\end{align*}

$$

Where now we define a new factor for <span style="color:#0077bb">athletic ability</span> (called <span style="color:#0077bb">AQ</span>). We could also change the model such that each variable is related to *both* <span style="color:#0077bb">IQ</span> and <span style="color:#0077bb">AQ</span>, although the weights between <span style="color:#0077bb">AQ</span> and traditional <span style="color:#cc6677">exam scores</span> would likely be very small. 


## Model Motivation

To work up to the model above, let's see how we could arrive at the eventual solution by starting with a very general and simple problem.

**Problem statement**: I have a <span style="color:#cc6677">high dimensional object</span> that I think can be <span style="color:#0077bb">adequately represented</span> in a **given** <span style="font-style:italic; font-style:italic; font-style:italic; font-style:italic; font-style:italic; color:#44aa99">lower dimensional space</span>.
### Least Squares

A very familiar formulation of the problem above is least squares. Let's say that we have a <span style="color:#cc6677">vector</span> that lives in a $\color{#cc6677}p$ <span style="color:#cc6677">dimensional space</span>, but we think that it can be  represented in a $\color{#44aa99}{q}$ <span style="color:#44aa99">dimensional subspace</span>, and we would like to find the "<span style="color:#0077bb">closest</span>" <span style="color:#0077bb">coordinates</span> in that subspace. I.e., we might want a linear model of the form

$$

\begin{align*}
\underbrace{\textcolor{#cc6677}y}_{p \times 1} \approx \underbrace{\textcolor{#44aa99}{X}\textcolor{#0077bb}{\beta}}_{(p \times q) (q \times 1)}
\end{align*}

$$

with $q < p$. Note that there's no noise here (yet). We want to find a lower dimensional version of $y$ that is as near to $y$ as possible. If we set our objective function to be $\min \|\|y - X\beta\|\|_2^2$, then we arrive at the least squares solution $\hat y$, where we project $y$ into the column space of $X$ to get

$$

\begin{align}
\beta &= (X^TX)^{-1}X^Ty\\
y &= X\beta + r,
\end{align}

$$

where $r$ is our vector of residuals. A necessary consequence of this projection is that $r$ is orthogonal to the column space of $X$. 
### Extending the model

Now suppose we have a similar situation: 

$$
\begin{align*}
\underbrace{\textcolor{#cc6677}{Z}}_{p \times m} \approx \underbrace{\textcolor{#44aa99}{\Lambda} \textcolor{#0077bb}{F}}_{(p \times q)(q \times m)}
\end{align*}
$$

with $q < p$. The first difference is that now instead of saying we have a single $p-$dimensional point to project, we want to project $m$ different $p-$dimensional points.

However, suppose that we don't know $\textcolor{#44aa99}{\Lambda}$ *or* $\textcolor{#0077bb}{F}$. So not only do we not have <span style="color:#0077bb">coordinates</span>, we don't even know what the <span style="color:#44aa99">subspace</span> is. How could we possibly solve this problem?

Well we might first note that whatever <span style="color:#44aa99">subspace</span> we *do* end up choosing, we'll probably just project into it. That is, once the <span style="color:#44aa99">subspace</span> is found, we can find the <span style="color:#0077bb">coordinates</span> by using the least squares technique above. Then we know that $F$ must have the form $F = (\Lambda^T \Lambda)^{-1}\Lambda^TZ$, which simplifies the problem a bit for us. Now rewriting our equation above:

$$

\begin{align}
Z &\approx \Lambda F\\
&=\Lambda(\Lambda^T \Lambda)^{-1}\Lambda^TZ\\
&= P_{\Lambda}Z.
\end{align}

$$

In the last line we note that, for a full column rank (or "tall") matrix $\Lambda$, $\Lambda(\Lambda^T\Lambda)^{-1}\Lambda^T$ is the form of a projection matrix, $P_{\Lambda}$. So the problem becomes how to find the "best" projection of $Z$ possible. What do we mean by "best"?

Well, least squares hasn't let us down so far, so why not try to find the projection that minimizes the mean squared error? I.e., our objective function is:

$$

\begin{align*}\min||Z - P_{\Lambda}Z||_F^2,\end{align*}

$$

where the Frobenius norm is being used. Now we will note something interesting by expanding this expression. First, see that by properties of projection matrices, $P_{\Lambda} = P_{\Lambda}P_{\Lambda} = P_{\Lambda}^T$. Second, recall that the trace operator is circular-shift invariant, e.g., $\text{trace}(ABC) = \text{trace}(CAB) = \text{trace}(BCA)$. Then:


$$
\begin{align*}
\min||Z - P_{\Lambda}Z||_F^2 &= \min\text{trace}\left\{(Z - P_{\Lambda}Z)^T(Z - P_{\Lambda}Z)\right\}\\
&= \min\text{trace}\left\{(Z^T - Z^TP_{\Lambda}^T)(Z - P_{\Lambda}Z)\right\}\\
&= \min\text{trace}\left\{Z^TZ - Z^TP_{\Lambda}Z - Z^TP_{\Lambda}^TZ + Z^TP_{\Lambda}^TP_{\Lambda}Z\right\}\\
&= \min\text{trace}\left\{Z^TZ - Z^TP_{\Lambda}Z - Z^TP_{\Lambda}Z + Z^TP_{\Lambda}Z\right\}\\
&= \min\text{trace}\left\{Z^TZ - Z^TP_{\Lambda}^TP_{\Lambda}Z\right\}\\
&= \max \text{trace}\{P_{\Lambda}ZZ^TP_{\Lambda}^T\}\\
\end{align*}
$$

Note that $ZZ^T$ is proportional to our covariance matrix, and so $P_\Lambda ZZ^T P_{\Lambda}^T$ is proportional to the covariance matrix of our projected $Z$. The diagonals of a covariance matrix represent the variance, and so by minimizing the mean squared errors between our vectors and their projections, we are equivalently maximizing the variances in the projected space. This may be starting to sound familiar.

By the [Eckart-Young Theorem,](https://en.wikipedia.org/wiki/Low-rank_approximation) this minimization occurs when we take the eigenvalue decomposition, $ZZ^T = W \Omega W^T$, with $\Omega$ a diagonal matrix of the eigenvalues of $ZZ^T$ and $W$ an orthogonal matrix composed of the eigenvectors. Then the solution is that $P_{\Lambda}$ should be equal to $W_qW_q^T$ where the columns of $W_q$ are the first $q$ columns of $W$, ordered by the magnitudes of the associated eigenvalues.

Some may notice that all we've done is traditional PCA; this is essentially the exact formulation and solution to the problem.

## So is PCA just Factor Analysis?

No!

Let's go back to our exam score example to give an example of what's going on. First, let's limit ourselves to just (centered) Math and History scores. Let's say that we have twenty students, and we plot their <span style="color:#cc6677">exam scores</span> below:

<p align="center">
    <img width="400" src="/images/FA_base_plot.png">
</p>

Now we plot the estimated one-dimensional <span style="color:#44aa99">subspace</span>...

<p align="center">
    <img width="400" src="/images/PCA_subspace_plot.png">
</p>

And now we project the <span style="color:#cc6677">exam scores</span> to get their <span style="color:#0077bb">coordinates</span> in the <span style="color:#44aa99">subspace</span>...

<p align="center">
    <img width="400" src="/images/PCA_final_plot.png">
</p>

However, take a closer look at our <span style="color:#999933">residuals</span>. They have two components, one in the direction of the Math scores, and one in the direction of the History scores. Since the <span style="color:#999933">residuals</span> *must* be orthogonal to our <span style="color:#44aa99">subspace</span>, a consequence of this is that the Math residuals are clearly correlated with the History residuals, **which is a violation of the key assumption we stated at the very beginning.**

This the major difference between PCA and factor analysis; they have a very similar problem statement, but with factor analysis we enforce the constraint that our errors need to be uncorrelated. 

In other words, PCA tries to retain as much variance as possible (even if that means retaining noise!), while factor analysis tries to retain the *correlation structure* while filtering out the noise.

Imagine when your parents would tell you to clean your room as a kid. You might've gone about this by taking the time to organize the mess and put things back where they belong. Sometimes you might even intentionally leave some commonly used stuff out--why put it away if you're just going to take it back out in a few minutes anyway? That would be factor analysis. PCA would be throwing all of your shit into the closet. Your room might technically be cleaner, but sometimes it's *less* useful to be "clean" and unorganized than it is to be a little messier, but with more structure.

Unfortunately, maintaining this structure comes with a cost. While PCA has a nice analytic solution, by introducing our constraint we lose the ability to solve for the factor loadings "nicely", and we have to settle for a numerical method. So how do we go about doing that?

