let IrisDataset = require('ml-dataset-iris');

require('ml-xgboost').then(XGBoost => {
    var booster = new XGBoost({
        booster: 'gbtree',
        objective: 'multi:softmax',
        max_depth: 5,
        eta: 0.1,
        min_child_weight: 1,
        subsample: 0.5,
        colsample_bytree: 1,
        silent: 1,
        iterations: 200
    });

    var trainingSet = IrisDataset.getNumbers();
    var predictions = IrisDataset.getClasses().map(
        (elem) => IrisDataset.getDistinctClasses().indexOf(elem)
    );

    booster.train(trainingSet, predictions);
    var predictions = booster.predict(trainingSet);
    console.log(predictions);
    // don't forget to free your model
    booster.free()
});