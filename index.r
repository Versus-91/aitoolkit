# Load necessary library
library(ggplot2)

# Load the Iris dataset
data(iris)

# Perform PCA on the numeric columns (excluding the Species column)
iris_pca <- prcomp(iris[, -5], scale. = TRUE)

# Create a scree plot using base R
screeplot(iris_pca, type = "lines", main = "Scree Plot")

# Get the proportion of variance explained by each component
explained_variance <- iris_pca$sdev^2 / sum(iris_pca$sdev^2)

# Create a data frame for plotting
variance_df <- data.frame(
  Principal_Component = 1:length(explained_variance),
  Variance_Explained = explained_variance
)

# Plot using ggplot2
ggplot(variance_df, aes(x = Principal_Component, y = Variance_Explained)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_line() +
  geom_point() +
  xlab("Principal Component") +
  ylab("Proportion of Variance Explained") +
  ggtitle("Scree Plot") +
  theme_minimal()
