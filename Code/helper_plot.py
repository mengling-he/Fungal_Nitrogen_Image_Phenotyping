import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

def plot_trait_scatter(
    df,
    trait_col='area_pixels',
    resp_col='respiration',
    hue_col=None,
    add_regline=True,
    per_group_r2=False,
    figsize=(10, 6)
):
    """
    Plot scatter of trait vs respiration with R².

    Parameters:
    ----------
    df : DataFrame
    trait_col : str
        Trait column name (X)
    resp_col : str
        Response variable (Y), default 'respiration'
    hue_col : str or None
        Column for coloring (e.g., strain)
    add_regline : bool
        Whether to add regression line
    per_group_r2 : bool
        Whether to compute R² per group
    """

    # === data ===
    X = df[[trait_col]].values
    y = df[resp_col].values

    # === global model ===
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)

    # === plot ===
    plt.figure(figsize=figsize)

    sns.scatterplot(
        data=df,
        x=trait_col,
        y=resp_col,
        hue=hue_col,
        palette='Set1' if hue_col else None
    )

    # === regression line ===
    if add_regline:
        sns.regplot(
            data=df,
            x=trait_col,
            y=resp_col,
            scatter=False,
            color='black'
        )

    # === annotate R² ===
    plt.text(
        0.05, 0.95,
        f"$R^2$ = {r2:.3f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top'
    )

    # === group R² ===
    if per_group_r2 and hue_col:
        y_pos = 0.85
        for group, d in df.groupby(hue_col):
            if len(d) > 2:
                X_g = d[[trait_col]].values
                y_g = d[resp_col].values
                r2_g = LinearRegression().fit(X_g, y_g).score(X_g, y_g)

                plt.text(
                    0.05, y_pos,
                    f"{group}: R² = {r2_g:.2f}",
                    transform=plt.gca().transAxes,
                    fontsize=10
                )
                y_pos -= 0.08

    # === title ===
    plt.title(f'{resp_col} vs {trait_col}')
    plt.xlabel(trait_col)
    plt.ylabel(resp_col)

    if hue_col:
        plt.legend(title=hue_col)

    plt.tight_layout()
    plt.show()