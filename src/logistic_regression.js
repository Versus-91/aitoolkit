
import { asyncRun } from "./py-worker";

export default class LogisticRegression {
    constructor() {
        this.context = {
        };
    }
    async train(x, y, x_test, column_names) {
        this.context = {
            X_train: x,
            y_train: y,
            X_test: x_test,
            column_names: column_names
        };
        const script = `
        import statsmodels.api as st
        import pandas as pd
        import numpy as np
        from js import X_train,y_train,X_test,column_names
        y = y_train
        df = pd.DataFrame(X_train,columns=column_names)
        x = df.iloc[:,:]
        x = st.add_constant(x, prepend = False)

        df_test = pd.DataFrame(X_test,columns=column_names)
        x_test = df_test.iloc[:,:]
        x_test = st.add_constant(x_test, prepend = False)

        mdl = st.MNLogit(y, x)
        mdl_fit = mdl.fit()
        summary = mdl_fit.summary()
        summary_table = summary.tables[1].data[0:]
        probabilities = np.array(mdl_fit.predict(x_test))
        
        (summary_table,probabilities,x_test.values)
    `;
        try {
            const { results, error } = await asyncRun(script, this.context);
            if (results) {
                console.log("pyodideWorker return results: ", results);
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

}