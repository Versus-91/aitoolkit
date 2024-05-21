import { asyncRun } from "../py-worker";


export default class LinearRegression {
    constructor(options) {
        this.options = options;
        this.model = null;

    }
    async train_test(x_train, y_train, x_test, labels) {
        this.context = {
            X_train: x_train,
            y_train: y_train,
            X_test: x_test,
            alpha: this.options.alpha,
            l1: this.options.l1,
            labels: labels
        };
        const script = `
        import numpy as np
        import statsmodels.api as sm
        from js import X_train,y_train,X_test,labels,l1,alpha
        import pandas as pd

        df_test = pd.DataFrame(X_test,columns=labels)
        x_test = df_test.iloc[:,:]
        test = sm.add_constant(x_test, prepend = False)

        df_train = pd.DataFrame(X_train,columns=labels)
        x_train = df_train.iloc[:,:]
        train = sm.add_constant(x_train, prepend = False)



        # Fit OLS model
        model = sm.OLS(np.array(y_train), train)
        if alpha is 0 and l1 is 0:
            res = model.fit()
        else:
            res = model.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=l1, refit=True)
        preds = res.predict(test)

        coefficients = []
        alphas = np.logspace(-3, 3, 100)

        for alpha in alphas:
            linear_reg = model.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=l1, refit=True)
            coefficients.append(linear_reg.params.tolist())

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
            "f_pvalue": res.f_pvalue,
            "aic": res.aic,
            "bic": res.bic,
            "coefs": coefficients,
            "alphas" : np.log(alphas)
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