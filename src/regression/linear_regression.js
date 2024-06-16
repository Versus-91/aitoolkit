import Plotly from 'plotly.js-dist';


export default class LinearRegression {
    constructor(options) {
        this.options = options;
        this.model = null;

    }
    async train_test(x_train, y_train, x_test, y_test, labels, container_regularization, container_errors, container_coefs) {
        this.context = {
            X_train: x_train,
            y_train: y_train,
            y_test: y_test,
            X_test: x_test,
            regularization_type: this.options.regularization === "Lasso" ? 1 : 0,
            labels: labels
        };
        return import('https://webr.r-wasm.org/latest/webr.mjs').then(
            async ({ WebR }) => {
                const webR = new WebR({ interactive: false });
                await webR.init();
                await webR.installPackages(['jsonlite', 'ggplot2', 'plotly', 'plotly_json', 'tidyr', 'dplyr', 'ggrepel', 'glmnet', 'modelsummary'], true);
                await webR.objs.globalEnv.bind('xx', x_train);
                await webR.objs.globalEnv.bind('x_test', x_test);

                await webR.objs.globalEnv.bind('y', y_train);
                await webR.objs.globalEnv.bind('names', labels);
                await webR.objs.globalEnv.bind('is_lasso', this.context.regularization_type);


                const plotlyData = await webR.evalR(`
                    library(plotly)
                    library(ggplot2)
                    library(tidyr)
                    library(dplyr)
                    library(ggrepel)
                    library(modelsummary)
                    library(glmnet)
                    data(mtcars)  
                    # Select all columns except the first as predictors. 
                    x <- as.matrix(xx)  
                    colnames(x) <- names

                    lam = 10 ^ seq (-2,3, length =100)    
                    cvfit = cv.glmnet(x, y, alpha = is_lasso)
                    
                    betas = as.matrix(cvfit$glmnet.fit$beta)
                    lambdas = cvfit$lambda
                    names(lambdas) = colnames(betas)
                    
                    
                    p <- as.data.frame(betas) %>% 
                      tibble::rownames_to_column("variable") %>% 
                      pivot_longer(-variable) %>% 
                      mutate(lambda=lambdas[name]) %>% 
                    ggplot(aes(x=lambda,y=value,col=variable)) + 
                      geom_line() + 
                      geom_label_repel(data=~subset(.x,lambda==min(lambda)),
                                       aes(label=variable),nudge_x=-0.5) +
                      geom_vline(xintercept=c(cvfit$lambda.1se,cvfit$lambda.min),
                                linetype="dashed")+
                      scale_x_log10()
                    
                    df = with(cvfit,
                            data.frame(lambda = lambdas,MSE = cvm,MSEhi=cvup,MSElow=cvlo))

                    p2<-ggplot(df,aes(x=lambda,y=MSE)) + 
                    geom_point(col="#f05454") + 
                    scale_x_log10("lambda") + 
                    geom_errorbar(aes(ymin = MSElow,ymax=MSEhi),col="#30475e") + 
                    geom_vline(xintercept=c(cvfit$lambda.1se,cvfit$lambda.min),
                                linetype="dashed")+
                    theme_bw()

                    model <- lm(y ~ ., data = as.data.frame(x))
                    x <- as.matrix(x_test)  
                    colnames(x) <- names
                    predictions <- predict(model, newdata = as.data.frame(x))
                    # Get coefficients, p-values, and standard errors
                    coefs <- coef(model)
                    pvals <- summary(model)$coefficients[,4]
                    std_error <- summary(model)$coefficients[,2]
                    aic_value <- AIC(model)
                    bic_value <- BIC(model)
                    rsquared <- summary(model)$r.squared

                    best_lambda <- cvfit$lambda.min
                    x <- as.matrix(xx) 
                    colnames(x) <- names
                    # Get the coefficients for the best lambda
                    best_model <- glmnet(x, y, alpha =is_lasso, lambda = best_lambda)
                    coefficients <- as.matrix(coef(best_model))
                    
                    nonzero_coef <- coefficients[coefficients != 0]
                    print(coefficients)
                    
                    # Get the names of the non-zero features (excluding the intercept)
                    nonzero_features <- rownames(coefficients)[coefficients != 0 & rownames(coefficients) != "(Intercept)"]
                    print(nonzero_features)
                    # Subset the original data to only include non-zero features
                    X_reduced <- x[, nonzero_features]

                    # Fit a linear regression model using the non-zero features
                    linear_model_min <- lm(y ~ ., data = as.data.frame(X_reduced))
                    coefs_min <- coef(linear_model_min)
                    pvals_min <- summary(linear_model_min)$coefficients[,4]
                    std_error_min <- summary(linear_model_min)$coefficients[,2]
                    best_lambda <- cvfit$lambda.1se
                    best_model <- glmnet(x, y, alpha =is_lasso, lambda = best_lambda)
                    coefficients <- as.matrix(coef(best_model))
                    
                    nonzero_coef <- coefficients[coefficients != 0]
                    print(coefficients)
                    nonzero_features <- rownames(coefficients)[coefficients != 0 & rownames(coefficients) != "(Intercept)"]
                    print(nonzero_features)
                    X_reduced <- x[, nonzero_features]
                    linear_model_1se <- lm(y ~ ., data = as.data.frame(X_reduced))
                    coefs_1se <- coef(linear_model_1se)
                    pvals_1se <- summary(linear_model_1se)$coefficients[,4]
                    std_error_1se <- summary(linear_model_1se)$coefficients[,2]
                    models <- list(
                        "Simple OLS" = model,
                        "Lambda Min OLS" = linear_model_min,
                        "Lambda 1se OLS" = linear_model_1se
                        )
                    z <- modelplot(models =models,coef_omit = 'Interc')
                    
                    list(plotly_json(p, pretty = FALSE),plotly_json(p2, pretty = FALSE),coefs,pvals,std_error,predictions,aic_value,bic_value,rsquared,coefs_min,pvals_min,std_error_min,coefs_1se,pvals_1se,std_error_1se,plotly_json(z, pretty = FALSE))
                    `);
                let results = await plotlyData.toArray()
                let reg_plot = JSON.parse(await results[0].toString())
                reg_plot.layout.legend["orientation"] = 'h'
                reg_plot["legend"]
                Plotly.newPlot(container_regularization, reg_plot, {
                    showlegend: true,
                    legend: { "orientation": "h" }
                });
                Plotly.newPlot(container_errors, JSON.parse(await results[1].toString()), {});
                Plotly.newPlot(container_coefs, JSON.parse(await results[15].toString()), {});

                let summary = {
                    params: await results[2].toArray(),
                    bse: await results[4].toArray(),
                    pvalues: await results[3].toArray(),
                    predictions: await results[5].toArray(),
                    aic: await results[6].toNumber(),
                    bic: await results[7].toNumber(),
                    r2: await results[8].toNumber(),
                    best_fit_min: {
                        coefs: await results[9].toArray(),
                        bse: await results[11].toArray(),
                        pvalues: await results[10].toArray(),
                    },
                    best_fit_1se: {
                        coefs: await results[12].toArray(),
                        bse: await results[14].toArray(),
                        pvalues: await results[13].toArray(),
                    },
                };
                return summary;
            }
        );

    }
    predict(x_test) {
        const result = this.model.predict(x_test);
        return result
    }
}