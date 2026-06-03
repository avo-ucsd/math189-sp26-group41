# Data Scraping
Week 9: May 25 - May 29

## Inferential Statistics
From the proposal: 
> For statistical testing, we plan to use Welch two-sample t-tests to compare mean retention ratios between free-to-play and paid games, Mann-Whitney U tests if the distributions are highly skewed, and confidence intervals or bootstrapping to estimate uncertainty in retention differences. We will also compare review scores near launch versus present-day review scores using paired tests where data is available.
>
> To make the analysis more rigorous, we will use regression models to control for possible confounding variables. For example, we may model long-term retention ratio as a function of free-to-play status, multiplayer status, release year, review score, and launch player count. We may also include interaction terms such as free-to-play × multiplayer to test whether free-to-play works better for online games than for single-player games.

### Tasks
1. Welch two-sample t-tests
   1. Retention ratio means between free-to-play and paid
2. Mann-Whitney U tests 
3. Confidence intervals
4. Bootstrapping for retention uncertainty
5. (IF WE HAVE TIME) Paired test for review scores at launch versus present day.
   1. For this one, we'd need to find a dataset or scrape data just FYI.
6. Regression for confounding
7. Long-term retention ratio as a function of multiple variables: free-to-play status, multiplayer status, release year, review score, etc.
   1. NOTE: review score probably is unavailable for our *current* data
   2. Interaction terms

| Task # | Assigned | Notes |
| ------ | -------- | ---- |
| 1      | Ana      | |
| 2      | Ana      | |
| 3      | Kangnan  | |
| 4      | Kangnan  | |
| 5      | Julian   | Please scrape historical review data and shove it in the `data/` folder |
| 6      | Tony   | |
| 7      | Ashley | |