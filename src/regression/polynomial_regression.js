import { asyncRun } from "../py-worker";


export default class PolynomialRegression {
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
            labels: labels,
            degree: this.options.degree
        };
        const script = `
        import numpy as np
        import statsmodels.api as sm
        from js import X_train,y_train,X_test,labels,l1,alpha,degree
        import pandas as pd
        from sklearn.preprocessing import PolynomialFeatures

        polynomial_features= PolynomialFeatures(degree=degree)
        df_test = pd.DataFrame(X_test,columns=labels)
        
        df_train = pd.DataFrame(X_train,columns=labels)
        x_train = polynomial_features.fit_transform(df_train.iloc[:,:])

        x_test = polynomial_features.transform(df_test.iloc[:,:])


        # Fit OLS model
        model = sm.OLS(np.array(y_train), x_train)
        if alpha is 0 and l1 is 0:
            res = model.fit()
        else:
            res = model.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=l1, refit=True)
        preds = res.predict(x_test)
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
            "bic": res.bic
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