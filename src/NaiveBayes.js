import { GaussianNB } from 'scikitjs'
import {
    getClasses,
    getClassesAsNumber,
    getCrossValidationSets,
    getDataset,
    getDistinctClasses,
    getNumbers,
} from 'ml-dataset-iris';
export default class NaiveBayes {
    constructor(options) {
    }
    async train(x_train, y_train) {
        const numbers = getNumbers();
        const classes = getClassesAsNumber();
        const clf = new GaussianNB({ priors: [0.5, 0.5, 0.5] })
        await clf.fit(numbers, classes)
        console.log(clf.predict(numbers).dataSync())

    }
    predict(x_test) {
        const result = this.model.predict(x_test);
        return result
    }
}