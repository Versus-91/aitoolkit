import { asyncRun } from "./py-worker";
export default class TSNE {
    constructor() {
    }
    async train(x) {
        this.context = {
            x_train: x,

        };
        const script = `
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.manifold import TSNE
        from js import x_train       
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform( np.array(x_train))
        X_tsne
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
