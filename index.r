# Load the necessary library
library(nnet)

# Fit multinomial logistic regression model
model <- multinom(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = iris)

# Summary of the model
summary(model)
