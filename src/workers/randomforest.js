import { RandomForestClassifier as RFClassifier } from 'ml-random-forest';


onmessage = (event) => {

    let model = new RFClassifier(event.data.options);
    model.train(event.data.x, event.data.y);
    let preds = model.predict(event.data.x_test)
    self.postMessage(preds)


}