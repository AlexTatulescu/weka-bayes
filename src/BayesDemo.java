import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class BayesDemo {
    public static void main(String[] args) throws Exception {

        BayesDependecies bayesDependencies = new BayesDependecies();

        Instances trainingDataSet = bayesDependencies.getDataSet(BayesDependecies.TRAINING_DATA_SET_FILENAME);
        Instances predictingDataSet = bayesDependencies.getDataSet(BayesDependecies.PREDICTION_DATA_SET_FILENAME);

        Evaluation eval = new Evaluation(trainingDataSet);

        Classifier naiveBayesClassifier = new NaiveBayes();
        naiveBayesClassifier.buildClassifier(trainingDataSet);
        bayesDependencies.testClassifierOnDatasetsFromFolder(eval, naiveBayesClassifier);

        Classifier bayesNetClassifier = new BayesNet();
        bayesNetClassifier.buildClassifier(trainingDataSet);
        bayesDependencies.testClassifierOnDatasetsFromFolder(eval, bayesNetClassifier);

        bayesDependencies.evaluateBayes(predictingDataSet, trainingDataSet, naiveBayesClassifier, eval, "** Naive Bayes Evaluation with Datasets **");
        bayesDependencies.evaluateBayes(predictingDataSet, trainingDataSet, bayesNetClassifier, eval, "** BayesNet Evaluation with Datasets **");
    }
}