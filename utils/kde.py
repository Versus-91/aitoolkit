import seaborn as sns


def draw_kde(df, column_name):
    
    kde = sns.kdeplot(data=df[column_name], fill=True, common_norm=False, palette="crest",
                      alpha=.5, linewidth=0)
    return kde.figure
