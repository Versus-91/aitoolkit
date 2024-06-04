import { asyncRun } from "../py-worker";


export default class LinearRegression {
    constructor(options) {
        this.options = options;
        this.model = null;

    }
    async train_test(x_train, y_train, x_test, y_test, labels) {
        this.context = {
            X_train: x_train,
            y_train: y_train,
            y_test: y_test,
            X_test: x_test,
            regularization_type: this.options.regularization === "Lasso" ? 1 : 0,
            labels: labels
        };
        const script = `
        import numpy as np
        import statsmodels.api as sm
        from js import X_train,y_train,y_test,X_test,labels,regularization_type
        import pandas as pd
        from sklearn.metrics import mean_squared_error

        df_test = pd.DataFrame(X_test,columns=labels)
        x_test = df_test.iloc[:,:]
        test = sm.add_constant(x_test, prepend = False)

        df_train = pd.DataFrame(X_train,columns=labels)
        x_train = df_train.iloc[:,:]
        train = sm.add_constant(x_train, prepend = False)



        # Fit OLS model
        model = sm.OLS(np.array(y_train), train)
        res = model.fit()
        preds = res.predict(test)

        coefficients = []
        alphas = np.logspace(-4, 1, 50)
        best_alpha = 0
        best_rmse = 10000
        for alpha in alphas:
            linear_reg = model.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=regularization_type, refit=True)
            coefficients.append(linear_reg.params.tolist())
            ypred = linear_reg.predict(test)
            rmse = mean_squared_error(y_test, ypred)
            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = alpha

        print(best_rmse)
        print(best_alpha)

        # Extract summary information
        summary_dict = {
            "params": res.params.tolist(),
            "bse": res.bse.tolist(),
            "preds": preds.tolist(),
            "tvalues": res.tvalues.tolist(),
            "pvalues": res.pvalues.tolist(),
            "rsquared": res.rsquared,
            "rsquared_adj": res.rsquared_adj,
            "fvalue": res.fvalue,
            "aic": res.aic,
            "bic": res.bic,
            "coefs": coefficients,
            "alphas" : alphas
        }
        
        summary_dict
        `;
        try {
            const { results, error } = await asyncRun(script, this.context);
            if (results) {
                return results;
            } else if (error) {
                console.log("pyodideWorker error: ", error);
            }
        } catch (e) {
            console.log(
                `Error in pyodideWorker at ${e.filename}, Line: ${e.lineno}, ${e.message}`,
            );
        }

    }
    predict(x_test) {
        const result = this.model.predict(x_test);
        return result
    }
}