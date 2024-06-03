# Install and load the glmnet package
library(glmnet)

# Load the Iris dataset
data(iris)

# Prepare the data
x <- as.matrix(iris[, 1:4])  # Features
y <- as.factor(iris$Species)  # Target variable

# Fit a multinomial logistic regression model with L2 regularization
model <- glmnet(x, y, family = "multinomial", alpha = 0)

# Display the model details
print(model)

# Cross-validation to find the best lambda
cv_model <- cv.glmnet(x, y, family = "multinomial", alpha = 0)
plot(cv_model)
best_lambda <- cv_model$lambda.min
print(best_lambda)

# Fit the model again with the best lambda
final_model <- glmnet(x, y, family = "multinomial", alpha = 0, lambda = best_lambda)

# Display the coefficients of the final model
coef(final_model)
