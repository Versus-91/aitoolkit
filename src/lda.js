import { asyncRun } from "./py-worker";
export default class DiscriminantAnalysis {
    constructor(options) {
        this.options = options
        this.context = {
            X_train: [0.8, 0.4, 1.2, 3.7, 2.6, 5.8],
            y_train: [0.8, 0.4, 1.2, 3.7, 2.6, 5.8],
            X_test: [0.8, 0.4, 1.2, 3.7, 2.6, 5.8],
        };
    }
    async train(x, y, x_test) {
        this.context = {
            lda_type: this.options.type,
            lda_type: this.options.prior,
            X_train: x,
            y_train: y,
            X_test: x_test,
        };
        const script = `
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from js import X_train,y_train,X_test,lda_type
        if lda_type == 0:
            da = LinearDiscriminantAnalysis()
        else:
            da = QuadraticDiscriminantAnalysis()
        da.fit(X_train, y_train)
        y_pred = da.predict(X_test)
        y_pred
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
