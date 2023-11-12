import { RandomForestClassifier as RFClassifier } from 'ml-random-forest';

export default class RandomForest {
    constructor(options) {
        this.model = new RFClassifier(options);
    }
    train(x_train, y_train) {
        this.model.train(x_train, y_train);
    }
    predict(x_test) {
        const result = this.model.predict(x_test);
        return result
    }
}