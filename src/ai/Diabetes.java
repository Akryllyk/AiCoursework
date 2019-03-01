package ai;


import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.meta.ClassificationViaRegression;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
/**
 *
 */
public class Diabetes {
    public static void main (String[] args) throws Exception{
        //load dataset
        DataSource source = new DataSource("/home/alex/AI stuff/diabetes.arff");
        DataSource source2 = new DataSource("/home/alex/AI stuff/diabetes_test.arff");
        Instances dataset = source.getDataSet();
        Instances dataset2 = source2.getDataSet();
        //set class index to the last attribute
        dataset.setClassIndex(dataset.numAttributes()-1);
        dataset2.setClassIndex(dataset2.numAttributes()-1);

        ClassificationViaRegression classifier = new ClassificationViaRegression();

        classifier.setClassifier(new LinearRegression());
        classifier.buildClassifier(dataset);

        Evaluation eval = new Evaluation(dataset2);

        for (int i = 0; i < dataset2.numInstances(); i++) {
            System.out.println("Instance " + (i+1) + ": " + dataset2.instance(i));

            if (eval.evaluateModelOnce(classifier, dataset2.instance(i)) == 0.0){
                System.out.println("Instance " + (i+1) + ": Tested Negative");
            } else{
                System.out.println("Instance " + (i+1) + ": Tested Positive");
            }
        }

    }
}
