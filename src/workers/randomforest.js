import { RandomForestClassifier as RFClassifier } from 'ml-random-forest';


onmessage = (event) => {
    console.log(event.data.options);
    if (event.data.options.criteria === 'gini') {
        let model = new RFClassifier(event.data.options);
        model.train(event.data.x, event.data.y);
        let preds = model.predict(event.data.x_test)
        self.postMessage(preds)
    } else {

    }

}