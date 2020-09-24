//import * as tf from '@tensorflow/tfjs';

async function load_model_and_predict(elemID, num){
    const model = await tf.loadLayersModel('model.json');
    const image = document.getElementById(elemID);
    const data = tf.browser.fromPixels(image, 1);
    const data2 = data.reshape([-1, 28]).expandDims();
    const whites = tf.fill([1, 28, 28], 255.0);
    const data3 = whites.sub(data2);
    const data4 = data3.div(255.0);
    const prediction = model.predict(data4);
    return [num, prediction];
}

function load(){
    for (var i = 0; i < 10; i++){
        const pred = load_model_and_predict('image'+i, i);
        pred.then((res) => {
//            console.log(i+res.toString());
            console.log(res[0]+' '+res[1].squeeze().toString());
        });
    }
}

