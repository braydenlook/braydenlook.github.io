---
layout: post
title:  "It's All Logistic Regression? Looking into NBA Rating Methods and Prediction"
---

There are a lot of ways to rate teams in the NBA, but many of them seem unjustified or unexplained. I wanted to compile most of them into a single place and look into the assumptions each one makes and what's different about them. When it comes to the simpler rating methods, it turns out they're mechanically almost exactly the same, but with a few minor differences in assumptions.

### Preliminaries 

First, a few words on rating methods and why we use them. One of the simplest and most widely understood rating methods is seeding. For example, the Denver Nuggets were the 1-seed in the West this year, and the Milwaukee Bucks were the 1-seed in the East. That means they won the most games in their respective conferences. 

However, most fans understand that seeding is not always the best way to "rank" teams because its resolution is capped. For example, suppose the 1-seed has 67 wins and the 2-seed has 52 wins. Now suppose instead that the 1-seed has 58 wins and the 2-seed has 56 wins. The seeding can't "see" the more granular details. That's why we take a step down and typically think about wins or winning percentage instead of seeding.

However, there's resolution prolems here as well: let's say a team wins 55 games, but every time they win, they only win by a single point. Now let's say there's another team that won 55 games, but every time they won, they won by 20. We'd likely have reason to believe team 2 is stronger than team 1. Hence, we may want to take into account not just who's winning, but how much they're winning (or losing) by. 

In basketball, this is done via net rating. Net rating is a measure of how much you outscore your opponents on average every 100 possessions. For example, if your offensive rating (points scored per 100 possessions) is 114.7 and your defensive rating (point allowed per 100 possessions) is 112.1, then your net rating is 114.7-112.1 = +2.5.

For the same reasons that wins are typically a better measure of skill than seeding, net rating is typically a better measure of a team's ability than wins by way of predicting future success. With that out of the way, let's get into some of the methods.

### Pythagorean Wins

Pythagorean Wins is a way to estimate the winning percentage of a team against an "average" team. It's also used to estimate the number of wins a team is "expected" to have, given their net rating.

Pythagorean Win % is defined as such: 

$$Win \% = \frac{Ortg^{\alpha}}{Ortg^{\alpha} + Drtg^{\alpha}}$$

Where Ortg is the team's offensive rating, Drtg is the team's defensive rating, and $$\alpha$$ is an empirically derived parameter. Most notable about this formula, though, is it can be manipulated into a familiar form:

$$Win \% = \frac{Ortg^{\alpha}}{Ortg^{\alpha} + Drtg^{\alpha}} = \frac{1}{1 + \frac{Drtg}{Ortg}^{\alpha}} = \frac{1}{1 + e^{\alpha \ln{\frac{Drtg}{Ortg}}}} = \frac{1}{1 + e^{-\alpha (\ln{Ortg} - \ln{Drtg})}}$$

It'll be clear later why we want to put into into this form, but those who are familiar will recognize this function as the **logistic function**.

### ELO

Nic Dobson has an excellent article on how robust Elo is that you should read [here](https://nicidob.github.io/nba_elo/) There will be some overlap in this piece, but my aim is on the theoretical properties, not so much on the validation.

Elo is a rating method you've probably seen in Chess, or in competitive online games like League of Legends. It's an adaptive rating system that updates after each game, and has the additional characteristic (in its simplest form) of being "blind"; it doesn't see anything you do other than win or lose.

In short, we can represent the probability of team A beating team B as:

$$\frac{1}{1 + e^{\sigma (R_A - R_B)}}$$

Where $$\sigma$$ is a tuning parameter and $$R$$ represents the ELO rating of a team.

### Log5

Log5 is a formula made by the legendary Bill James to calculate the probability that team A beats team B, given each of their winning percentages so far. The formula for $$p_{a, b}$$, the probability team A beats team B, is given by:

$$p_{a, b} = \frac{p_a - p_ap_b}{p_a+p_b-2p_ap_b}$$

where $$p_a$$ and $$p_b$$ are team A and team B's winning percentages, respectively. The derivation and justification for this is not obvious from the formula or from Bill James' introduction of it, but the way I think about it that (eventually) made some sense to me is to reparametrize the equation by first imagining each team has a "rating", $$\gamma$$.

Now set

$$p_a = \frac{\gamma_a}{\gamma_a + 1/2}$$

such that

$$\gamma_a = 1/2\frac{p_a}{1-p_a}$$

Now if we plug this back into the formula given above, we can show that

$$p_{a, b} = \frac{1/2\gamma_a}{1/2\gamma_a + 1/2\gamma_b} = \frac{\gamma_a}{\gamma_a + \gamma_b}$$

You may see where this is going now. Let $$e^{\beta} = \gamma$$

$$p_{a, b} = \frac{e^{\beta_a}}{e^{\beta_a} + e^{\beta_b}} = \frac{1}{1 + e^{\beta_b-\beta_a}}$$

or,

$$p_{a, b} = \frac{1}{1 + e^{-(\beta_a-\beta_b)}}$$

Look at that, we're back at our familiar logistic function.

### Simple Rating System (SRS)

### Bradley Terry and Logistic Regression

This is where we try to tie most of these rating methods together. 

The Bradley-Terry Model is a way to predict the outcome of a head-to-head matchup based on other matchups that may or may not include both teams. The concept it much more abstract than just sports, but it happens to fit sports quite well, since we often have this indirect matchup data. 

For example, if the Lakers beat the Warriors 7 out of 10 matchups, we can conclude that the Lakers are *probably* a better team than the Warriors. But what if they've never played each other? What if we're told that the Warriors went 2-1 against the Suns, 1-1 with the Milwaukee Bucks, and 3-0 against the Clippers, while the Lakers went 0-3 against the Clippers, 3-0 against the Suns, and 2-0 against the Bucks? How do we untangle this?

Bradley-Terry starts by formulating that the probability that team A beats team B is given as 

$$p_{a, b} = \frac{\gamma_a}{\gamma_a + \gamma_b}$$

some proportion of what we'll call "strength points", the $$\gamma$$'s. So if team A has "strength" 1100 and team B has "strength" 500, then team A has a $$\frac{1100}{1600} = 0.6875$$ probability of beating team B. Then the goal is to solve for these strength numbers.

As we did in the Log5 section, we can let $$e^{\beta} = \gamma$$ to get 

$$p_{a, b} = \frac{1}{1 + e^{-(\beta_a-\beta_b)}}$$

The difference here is that before, $$\gamma$$ was a function of the team's winning percentage, $$\gamma_a = 1/2\frac{p_a}{1-p_a}$$. Now, we don't have an expression for $$\gamma$$ or $$\beta$$. So what do we do to solve?

For those who are statistically inclined, the following should make some sense. For others, this may seem like a leap, but we can solve for estimated $$\beta$$ values (and thus estimate probabilities) by using logistic regression.

### Tying it all together / TLDR

Let's put all of the formulas together and try to untangle what we've gone over:

**Pythagorean Win %**: $$p = \frac{1}{1 + e^{-\alpha (\ln{Ortg} - \ln{Drtg})}}$$ 

Tells you the expected winning percentage **against an average team**, given Ortg and Drtg. $$\alpha$$ is found empirically and is typically between 12-16.

**ELO**: $$p_{a, b} = \frac{1}{1 + e^{\sigma (R_A - R_B)}}$$

Tells you the expected winning percentage of team A against team B, given their current ELO ratings. Whoever wins gains ELO and whoever loses has their ELO reduced. $$\sigma$$ is a parameter similar to $$\alpha$$ that determines the variance involved, and is typically set to BLANK


**Log5**: $$p_{a, b} = \frac{1}{1 + e^{-(\beta_a-\beta_b)}}$$

Tells you the expected winning percentage of team A against team B, given their current winning percentages, where $$\gamma_a = 1/2\frac{p_a}{1-p_a}$$ and $$e^{\beta} = \gamma$$.

**Bradley-Terry** $$p_{a, b} = \frac{1}{1 + e^{-(\beta_a-\beta_b)}}$$