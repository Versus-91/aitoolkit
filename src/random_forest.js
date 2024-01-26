

export default class RandomForest {
    constructor(options) {
        this.options = options;
        this.model = null;

    }
    train_test(x_train, y_train, x_test) {
        let worker = new Worker(
            new URL('./workers/randomforest', import.meta.url),
            { type: 'module' }
        );
        return new Promise((resolve, reject) => {
            worker.onmessage = (e) => {
                resolve(e.data)
            };
            worker.onerror = (error) => { throw error };
            worker.postMessage({ x: x_train, y: y_train, x_test: x_test, options: this.options });
        })

    }
    predict(x_test) {
        const result = this.model.predict(x_test);
        return result
    }
}