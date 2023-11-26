import SVM from "libsvm-js/asm";
export default class SupportVectorMachine {
    constructor(options) {
        this.model = new SVM({
            kernel: SVM.KERNEL_TYPES.RBF,
            type: SVM.SVM_TYPES.C_SVC,
            gamma: 0.25,
            cost: 1,
            quiet: true
        });
    }
    train(x_train, y_train) {
        this.model.train(x_train, y_train);
    }
    predict(x_test) {
        const result = this.model.predict(x_test);
        return result
    }
}