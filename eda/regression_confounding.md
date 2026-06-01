# Regression for Potential Confounders

## Objective

Determine whether monetization model (free-to-play vs. paid) is associated with player retention after controlling for other factors.

## Response Variable

* Retention Ratio

## Predictors

* Free-to-Play Status
* Review Score
* Release Year
* Peak Concurrent Players

## Model

Retention Ratio ~ Free-to-Play Status + Review Score + Release Year + Peak Concurrent Players

## Rationale

Review score, release year, and peak concurrent players may influence retention independently of monetization model. Including these variables helps account for potential confounding effects.

## Interpretation

If free-to-play status remains significant after controlling for these variables, monetization model may have an independent effect on retention. Otherwise, differences in retention may be explained by other game characteristics.
