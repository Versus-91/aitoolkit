# Load the required packages
library(nnet)

# Load the Iris dataset
data(iris)

# Split data into train and test sets
set.seed(123)  # for reproducibility
train_indices <- sample(1:nrow(iris), 0.8 * nrow(iris))
train_data <- iris[train_indices, ]
test_data <- iris[-train_indices, ]

# Fit multinomial logistic regression model
multinom_model <- multinom(Species ~ ., data = train_data)

# Print summary of the model
summary(multinom_model)

# Confidence intervals for the coefficients
confint(multinom_model)
