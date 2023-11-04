class KNN {


    train(x_train, y_train, k = 3) {
        var knn = new KNN(x_train, y_train, { k: 2 });
        var ans = knn.predict(test_dataset);
        console.log(ans);
    }
}