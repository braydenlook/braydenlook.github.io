---
layout: post
title:  "It's All Logistic Regression? Looking into NBA Rating Methods and Prediction"
---

There are a lot of ways to rate teams in the NBA, but many of them seem unjustified or unexplained. I wanted to compile most of them into a single place and look into the assumptions each one makes and what's different about them. When it comes to the simpler rating methods, it turns out they're mechanically almost exactly the same, but with a few minor differences in assumptions.

### Preliminaries 

First, a few words on rating methods and why we use them. One of the simplest and most widely understood rating methods is seeding. For example, the Denver Nuggets were the 1-seed in the West this year, and the Milwaukee Bucks were the 1-seed in the East. That means they won the most games in their respective conferences. 

Seeding is a simple and succinct way to rate teams, but runs into a few problems. One is that there's no nuance in the difference between seeds. If the 1-seed won 80 games and the 2 seed won 50 games, seeding sees no difference between this vs if the 1-seed won 55 games and the 2-seed won 54 games. 

The other problem is that when you split up seeds between conferences, they also can't "see" the difference between the strength of each conference. E.g., let's say the western 1-seed won 52 games and the eastern 2-seed won 60 games. We might think the 60-win team is stronger, but they're a lower seed. 

That's why we take a step down and typically think about wins or winning percentage instead of seeding. However, there's resolution prolems here as well: let's say a team wins 55 games, but every time they win, they only win by a single point. Now let's say there's another team that won 55 games, but every time they won, they won by 20. We'd likely have reason to believe team 2 is stronger than team 1. Hence, we may want to take into account not just who's winning, but how much they're winning (or losing) by. 

In basketball, this is done via net rating. Net rating is a measure of how much you outscore your opponents on average every 100 possessions. For example, if your offensive rating (points scored per 100 possessions) is 114.7 and your defensive rating (point allowed per 100 possessions) is 112.1, then your net rating is 114.7-112.1 = +2.5.

Net rating is typically a better measure of a team's ability than wins by way of predicting future success. With that out of the way, let's get into some of the methods.

### Pythagorean Wins

Pythagorean Wins is a way to estimate the winning percentage of a team against an "average" team. It's also used to estimate the number of wins a team is "expected" to have, given their net rating.

Pythagorean Win % is defined as such: 

$$Win \% = \frac{Ortg^{\alpha}}{Ortg^{\alpha} + Drtg^{\alpha}}$$

Where Ortg is the team's offensive rating, Drtg is the team's defensive rating, and $$\alpha$$ is an empirically derived parameter. Most notable about this formula, though, is it can be manipulated into a familiar form:

$$Win \% = \frac{Ortg^{\alpha}}{Ortg^{\alpha} + Drtg^{\alpha}} = \frac{1}{1 + \frac{Drtg}{Ortg}^{\alpha}} = \frac{1}{1 + e^{\alpha \ln{\frac{Drtg}{Ortg}}}} = \frac{1}{1 + e^{-\alpha (\ln{Ortg} - \ln{Drtg})}}$$

### ELO

Nic Dobson has an excellent article on how robust Elo is that you should read HERE. There will be some overlap in this piece, but my aim is on the theoretical properties.

Elo is a rating method you've probably seen in Chess, or in competitive online games like League of Legends. It's an adaptive rating system that updates after each game, and has the additional characteristic (in its simplest form) of being "blind"; it doesn't see anything you do other than win or lose.

### Log5

Log5 is a formula to calculate the probability that team A beats team B, given each of their winning percentages so far. The formula for $$p_{a, b}$$, the probability team A beats team B, is given by:

$$p_{a, b} = \frac{p_a - p_ap_b}{p_a+p_b-2p_ap_b}$$

where $$p_a$$ and $$p_b$$ are team A and team B's winning percentages, respectively. The derivation and justification for this is not obvious from the formula or from Bill James' introduction of it, but the most intuitive way to think of it is to reparametrize the equation by first imagining each team has a "rating", $$\beta$$.

Now set

$$p_a = \frac{\beta_a}{\beta_a + 1/2}$$

such that

$$\beta_a = 1/2\frac{p_a}{1-p_a}$$

Now if we plug this back into the formula given above, we can show that

$$p_{a, b} = \frac{1/2\beta_a}{1/2\beta_a + 1/2\beta_b} = \frac{\beta_a}{\beta_a + \beta_b}$$

You may see where this is going now. Let $$e^{\gamma} = \beta$$

$$p_{a, b} = \frac{e^{\gamma_a}}{e^{\gamma_a} + e^{\gamma_b}} = \frac{1}{1 + e^{\gamma_b-\gamma_a}}$$