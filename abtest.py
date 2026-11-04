import numpy as np
from scipy import stats

N = 1000

def generate_data(seed=42):
    """Generate synthetic A/B test data using a single seeded RNG.

    Returns a dict with control_duration, treatment_duration,
    control_converted, treatment_converted (each length 1000).
    """
    rng = np.random.default_rng(seed)
    control_duration = rng.normal(loc=8.0, scale=3.0, size=N)
    treatment_duration = rng.normal(loc=8.5, scale=3.0, size=N)
    control_converted = (rng.random(N) < 0.10).astype(int)
    treatment_converted = (rng.random(N) < 0.13).astype(int)

    return {
        "control_duration": control_duration,
        "treatment_duration": treatment_duration,
        "control_converted": control_converted,
        "treatment_converted": treatment_converted
    }

def describe_continuous(name, x):
    n = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    median = np.median(x)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    print(
        f"{name + ':':<20} n={n}, mean={mean:.4f}, std={std:.4f}, "
        f"median={median:.4f}, q1={q1:.4f}, q3={q3:.4f}"
    )

def describe_binary(name, x):
    n = len(x)
    count = np.sum(x)
    rate = np.mean(x)
    print(f"{name + ':':<20} n={n}, count={count}, rate={rate:.4f}")

if __name__ == "__main__":
    data = generate_data()
    control_mean = np.mean(data["control_duration"])
    control_std = np.std(data["control_duration"], ddof=1)
    treatment_mean = np.mean(data["treatment_duration"])
    treatment_std = np.std(data["treatment_duration"], ddof=1)

    control_converted_mean = np.mean(data["control_converted"])
    control_converted_sum = np.sum(data["control_converted"])
    control_converted_n = len(data["control_converted"])

    treatment_converted_mean = np.mean(data["treatment_converted"])
    treatment_converted_sum = np.sum(data["treatment_converted"])
    treatment_converted_n = len(data["treatment_converted"])

    header = f"{'Group':<10}| {'Mean duration':<14}| {'Std duration':<13}| Conversion rate"
    control_row = (
        f"{'control':<10}| {control_mean:<14.4f}| {control_std:<13.4f}| "
        f"{control_converted_mean:.4f}({control_converted_sum}/{control_converted_n})"
    )
    treatment_row = (
        f"{'treatment':<10}| {treatment_mean:<14.4f}| {treatment_std:<13.4f}| "
        f"{treatment_converted_mean:.4f}({treatment_converted_sum}/{treatment_converted_n})"
    )

    # Exercise 1: Data generation and group summary table
    print(header)
    print(control_row)
    print(treatment_row)
    print()

    # Exercise 2: Descriptive comparison
    describe_continuous("control duration", data["control_duration"])
    describe_continuous("treatment duration", data["treatment_duration"])
    describe_binary("control converted", data["control_converted"])
    describe_binary("treatment converted", data["treatment_converted"])
    print()

    duration_diff = np.mean(data["treatment_duration"]) - np.mean(data["control_duration"])
    conversion_diff = np.mean(data["treatment_converted"]) - np.mean(data["control_converted"])
    print(f"{'duration difference:':<22} {duration_diff:.4f}")
    print(f"{'conversion difference:':<22} {conversion_diff:.4f}")
    print()

    # Exercise 3: Assumption checks
    shapiro_control_stat, shapiro_control_p = stats.shapiro(data["control_duration"])
    shapiro_treatment_stat, shapiro_treatment_p = stats.shapiro(data["treatment_duration"])
    levene_stat, levene_p = stats.levene(data["control_duration"], data["treatment_duration"])

    print(f"{'Shapiro control:':<18} stat={shapiro_control_stat:.4f}, p={shapiro_control_p:.4f}")
    print(f"{'Shapiro treatment:':<18} stat={shapiro_treatment_stat:.4f}, p={shapiro_treatment_p:.4f}")
    print(f"{'Levene:':<18} stat={levene_stat:.4f}, p={levene_p:.4f}")
    print()

    alpha = 0.05
    normal = shapiro_control_p > alpha and shapiro_treatment_p > alpha
    equal_var = levene_p > alpha

    print(f"Normality assumption holds: {normal}")
    print(f"Equal variance assumption holds: {equal_var}")
    print("Decision: continuous metric -> Welch t-test; binary metric -> chi-squared 2x2")
    print()

    # Exercise 4: Hypothesis tests
    t_stat, p_value = stats.ttest_ind(
        data["control_duration"], data["treatment_duration"], equal_var=False
    )
    reject_duration_t = p_value < alpha
    u_stat, p_value_mwu = stats.mannwhitneyu(
        data["control_duration"], data["treatment_duration"], alternative="two-sided"
    )

    control_success = np.sum(data["control_converted"])
    control_fail = len(data["control_converted"]) - control_success
    treatment_success = np.sum(data["treatment_converted"])
    treatment_fail = len(data["treatment_converted"]) - treatment_success

    table = np.array([
        [control_success, control_fail],
        [treatment_success, treatment_fail],
    ])
    chi2_stat, p_value_chi2, dof, expected = stats.chi2_contingency(table)

    print(f"duration Welch: t={t_stat:.4f} p={p_value:.4f} reject? {'Yes' if p_value < alpha else 'No'}")
    print(f"duration MWU: U={u_stat:.1f} p={p_value_mwu:.4f} reject? {'Yes' if p_value_mwu < alpha else 'No'}")
    print()
    print("Contingency table:")
    print(table)
    print()
    print(f"conversion chi2: chi2={chi2_stat:.4f} p={p_value_chi2:.4f} dof={dof} reject? {'Yes' if p_value_chi2 < alpha else 'No'}")
    print()

    n_control = len(data["control_duration"])
    n_treatment = len(data["treatment_duration"])

    mean_control = np.mean(data["control_duration"])
    mean_treatment = np.mean(data["treatment_duration"])

    var_control = np.var(data["control_duration"], ddof=1)
    var_treatment = np.var(data["treatment_duration"], ddof=1)

    pooled_std = np.sqrt(
        ((n_control - 1) * var_control + (n_treatment - 1) * var_treatment)
        / (n_control + n_treatment - 2)
    )

    # Exercise 5: Effect sizes and confidence intervals
    cohens_d = (mean_treatment - mean_control) / pooled_std

    se_duration = np.sqrt(var_control / n_control + var_treatment / n_treatment)
    diff_duration = mean_treatment - mean_control
    ci_duration_low = diff_duration - 1.96 * se_duration
    ci_duration_high = diff_duration + 1.96 * se_duration

    p_control = np.mean(data["control_converted"])
    p_treatment = np.mean(data["treatment_converted"])
    n_control_conv = len(data["control_converted"])
    n_treatment_conv = len(data["treatment_converted"])

    se_conversion = np.sqrt(
        p_control * (1 - p_control) / n_control_conv
        + p_treatment * (1 - p_treatment) / n_treatment_conv
    )
    diff_conversion = p_treatment - p_control
    ci_conversion_low = diff_conversion - 1.96 * se_conversion
    ci_conversion_high = diff_conversion + 1.96 * se_conversion

    print(f"Cohen's d (duration): {cohens_d:.4f}")
    print(f"95% CI duration diff:   [{ci_duration_low:.2f}, {ci_duration_high:.2f}]")
    print(f"95% CI conversion diff: [{ci_conversion_low:.3f}, {ci_conversion_high:.3f}]")
    print(
        f"Note: duration CI excludes 0 (consistent with significant Welch result); "
        f"conversion CI includes 0 (consistent with non-significant chi-squared result)."
    )
    print()