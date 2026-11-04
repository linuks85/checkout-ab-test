# Checkout A/B Test Analysis

Statistical analysis of an A/B test comparing the current checkout flow (control) with a redesigned one (treatment).
The data is **synthetic** (generated with a seeded RNG, 1000 users per group), so the project demonstrates the
methodology, not a real business result.

## What it does

- Generates reproducible data: session duration (continuous) and conversion (binary).
- Checks assumptions: Shapiro-Wilk (normality), Levene (equal variances).
- Picks tests from the checks: **Welch's t-test** + **Mann-Whitney U** for duration, **chi-squared** (2x2) for conversion.
- Reports effect sizes and 95% confidence intervals (**Cohen's d**, difference CIs).
- Ends with a business recommendation based on statistical *and* practical significance.

## Results (seed 42)

| Metric | Control | Treatment | Test | Verdict |
|---|---|---|---|---|
| Session duration, min | 7.91 | 8.26 | Welch t=-2.55, p=0.0109 | significant, but small (Cohen's d = 0.114) |
| Conversion | 10.8% | 13.4% | chi2=2.94, p=0.0865 | not significant, CI includes 0 |

**Recommendation:** do not roll out on this data alone; extend the experiment to see whether the conversion effect is real.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python abtest.py      # full analysis output
pytest                # tests
```

## Stack

Python, NumPy, SciPy, pytest.

*Built as a learning project at Tomorrow School (AI Piscine).*
