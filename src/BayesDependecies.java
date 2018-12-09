import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;

class BayesDependecies {

    static final String TRAINING_DATA_SET_FILENAME = "C:\\Program Files\\Weka-3-9\\data\\breast-cancer.arff";

    static final String TESTING_DATA_SET_FILENAME = "C:\\Program Files\\Weka-3-9\\data\\breast-cancer.arff";

    static final String PREDICTION_DATA_SET_FILENAME = "C:\\Program Files\\Weka-3-9\\data\\breast-cancer.arff";

    Instances getDataSet(String fileName) throws Exception {
        StringToWordVector filter = new StringToWordVector();
        int classIdx = 1;
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(fileName));
        Instances dataSet = loader.getDataSet();
        dataSet.setClassIndex(classIdx);
        filter.setInputFormat(dataSet);
        dataSet = Filter.useFilter(dataSet, filter);

        return dataSet;
    }

    void evaluateBayes(Instances predictingDataSet, Instances trainingDataSet, Classifier classifier, Evaluation evaluator, String info) {
        try {
            System.out.println(info);
            System.out.println(evaluator.toSummaryString());
            System.out.print(" the expression for the input data as per alogorithm is ");
            System.out.println(classifier);
            for (int i = 0; i < predictingDataSet.numInstances(); i++) {
                System.out.println(predictingDataSet.instance(i));
                double index = classifier.classifyInstance(predictingDataSet.instance(i));
                String className = trainingDataSet.attribute(0).value((int) index);
                System.out.println(className);
            }
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }

    }
}
