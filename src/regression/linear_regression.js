import { asyncRun } from "../py-worker";


export default class LinearRegression {
    constructor(options) {
        this.options = options;
        this.model = null;

    }
    async train_test(x_train, y_train, x_test, y_test, labels, container_regularizayion, container_errors) {
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
                await webR.installPackages(['jsonlite', 'ggplot2', 'plotly', 'plotly_json', 'tidyr', 'dplyr', 'ggrepel', 'glmnet'], true);
                await webR.objs.globalEnv.bind('xx', x_train);
                await webR.objs.globalEnv.bind('x_test', x_test);

                await webR.objs.globalEnv.bind('y', y_train);
                await webR.objs.globalEnv.bind('names', labels);

                const plotlyData = await webR.evalR(`
                    library(plotly)
                    library(ggplot2)
                    library(tidyr)
                    library(dplyr)
                    library(ggrepel)
                    library(glmnet)
                    data(mtcars)  
                    # Select all columns except the first as predictors. 
                    x <- as.matrix(xx)  
                    colnames(x) <- names

                    
                    lam = 10 ^ seq (-2,3, length =100)    
                    cvfit = cv.glmnet(x, y, alpha = 0, lambda = lam)
                    
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
                      scale_x_log10()
                    
                    df = with(cvfit,
                            data.frame(lambda = lambdas,MSE = cvm,MSEhi=cvup,MSElow=cvlo))

                    p2<-ggplot(df,aes(x=lambda,y=MSE)) + 
                    geom_point(col="#f05454") + 
                    scale_x_log10("log(lambda)") + 
                    geom_errorbar(aes(ymin = MSElow,ymax=MSEhi),col="#30475e") + 
                    geom_vline(xintercept=c(cvfit$lambda.1se,cvfit$lambda.min),
                                linetype="dashed")+
                    theme_bw()

                    model <- lm(y ~ x)
                    x <- as.matrix(x_test)  
                    colnames(x) <- names
                    predictions <- predict(model, newdata = as.data.frame(x))
                    # Get coefficients, p-values, and standard errors
                    coefs <- coef(model)
                    pvals <- summary(model)$coefficients[,4]
                    std_error <- summary(model)$coefficients[,2]

                    list(plotly_json(p, pretty = FALSE),plotly_json(p2, pretty = FALSE),coefs,pvals,std_error,predictions)
                    `);
                let results = await plotlyData.toArray()
                console.log(results);
                Plotly.newPlot(container_regularizayion, JSON.parse(await results[0].toString()), {});
                Plotly.newPlot(container_errors, JSON.parse(await results[1].toString()), {});
                return {
                    params: await results[2].toArray(),
                    bse: await results[4].toArray(),
                    pvalues: await results[3].toArray(),
                    predictions: await results[5].toArray(),
                }
            }
        );

    }
    predict(x_test) {
        const result = this.model.predict(x_test);
        return result
    }
}