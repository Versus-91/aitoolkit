import { asyncRun } from "../py-worker";


export default class RandomForestRegressor {
    constructor(options) {
        this.options = options;
        this.model = null;

    }
    async train_test(x_train, y_train, x_test) {
        this.context = {
            X_train: x_train,
            y_train: y_train,
            X_test: x_test,
            rf_type: this.options.criteria,
            max_features: this.options.features,
            num_estimators: this.options.nEstimators <= 0 || !this.options.nEstimators ? 100 : this.options.nEstimators,
            max_depth: this.options.treeOptions.maxDepth <= 0 ? 5 : this.options.treeOptions.maxDepth
        };
        const script = `
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import accuracy_score
            from js import X_train,y_train,X_test,rf_type,max_features,num_estimators,max_depth
            clf = RandomForestRegressor(criterion=rf_type,max_features = max_features,n_estimators=num_estimators,max_depth = max_depth, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred
        `;
        try {
            const { results, error } = await asyncRun(script, this.context);
            if (results) {
                return Array.from(results);
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