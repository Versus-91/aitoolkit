import seaborn as sns


def draw_kde(df):
    kde = sns.kdeplot(data=df.loc[:, df.columns != 'Id'], fill=True, common_norm=False, palette="crest",
                      alpha=.5, linewidth=0)
    return kde.figure
