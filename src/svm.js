import SVM from "libsvm-js/asm";
export default class SupportVectorMachine {
    constructor(options) {
        this.model = new SVM(options);
    }
    train(x_train, y_train) {
        this.model.train(x_train, y_train);
    }
    predict(x_test) {
        const result = this.model.predict(x_test);
        return result
    }
}