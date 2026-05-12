import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("forest_evo_clean_data.csv")

df["player_cat"] = pd.Categorical(df["Player_ID"])
df["game_cat"] = pd.Categorical(df["Game_Type"])

player_idxs = df["player_cat"].cat.codes.values
game_idxs = df["game_cat"].cat.codes.values
player_names = df["player_cat"].cat.categories
game_names = df["game_cat"].cat.categories

y_obs = df["Rank"].values - 1

# Model
with pm.Model() as ranking_model:

    mu_p = pm.Normal("mu_p", mu=0, sigma=1)
    sigma_p = pm.HalfNormal("sigma_p", sigma=1)

    skill_raw = pm.Normal("skill_raw", mu=0, sigma=1, shape=len(player_names))
    skill = pm.Deterministic("skill", mu_p + skill_raw * sigma_p)
    game_effect = pm.Normal("game_effect", mu=0, sigma=1, shape=len(game_names))

    mu_y = -(skill[player_idxs] + game_effect[game_idxs])

    eps = pm.HalfNormal("eps", sigma=1)

    obs = pm.Normal("obs", mu=mu_y, sigma=eps, observed=y_obs)

    trace = pm.sample(10000, tune=2000, target_accept=0.99, random_seed=42)

# Save Summary Stats
summary_stats = az.summary(trace, var_names=["skill"])
summary_stats.index = player_names
summary_stats.to_csv("model_summary.csv")
print(summary_stats)

# Trace Plot
axes = az.plot_trace(
    trace,
    var_names=["skill"],
    coords={"skill_dim_0": list(range(len(player_names)))},
    compact=False,
)
num_plots = axes.shape[0]
for i in range(num_plots):
    name = player_names[i]
    axes[i, 0].set_title(f"Skill: {name} (Posterior)")
    axes[i, 1].set_title(f"Skill: {name} (Trace)")
fig = axes[0][0].get_figure()
fig.set_size_inches(12, len(player_names) * 2.5)
plt.subplots_adjust(hspace=0.8)
plt.savefig("trace_plot.png", dpi=300)
plt.show()
plt.close()

# Skill Plot
df_sorted = summary_stats.sort_values(by="mean", ascending=True)

fig, ax = plt.subplots(figsize=(12, 18))
y_pos = np.arange(len(df_sorted))
ax.hlines(
    y_pos,
    df_sorted["hdi_3%"],
    df_sorted["hdi_97%"],
    color="steelblue",
    alpha=0.6,
    linewidth=2,
    label="94% HDI",
)
ax.plot(
    df_sorted["mean"],
    y_pos,
    "o",
    color="white",
    markeredgecolor="steelblue",
    markersize=8,
    markeredgewidth=2,
)

x_ticks = np.arange(-5.5, 0.5, 0.5)
ax.set_xticks(x_ticks)

ax.set_yticks(y_pos)
ax.set_yticklabels(df_sorted.index, fontsize=13)

ax.set_ylim(-1, len(df_sorted))
ax.set_xlim(df_sorted["hdi_3%"].min() - 0.5, df_sorted["hdi_97%"].max() + 0.5)

ax.set_title(
    "Ranked Latent Skill Estimates (Top Performers at Top)", fontsize=20, pad=25
)
ax.set_xlabel("Skill Value (Higher is Better)", fontsize=15, labelpad=15)

ax.grid(axis="x", linestyle="--", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("skill_plot.png", dpi=300)
plt.show()
plt.close()

# DAG
pm.model_to_graphviz(ranking_model)

# Calculate Winning Probability
with ranking_model:
    ppc = pm.sample_posterior_predictive(trace)

post_skills = trace.posterior["skill"].values.reshape(-1, len(player_names))
wins = np.argmax(post_skills, axis=1)
win_counts = np.bincount(wins, minlength=len(player_names))
win_probs = win_counts / len(wins)

prob_df = pd.DataFrame(
    {"Player": player_names, "Win_Probability": win_probs}
).sort_values(by="Win_Probability", ascending=False)

prob_df.to_csv("win_probability.csv", index=False)
print(prob_df)

az.plot_ppc(ppc)
plt.savefig("ppc_plot.png")
plt.close()
