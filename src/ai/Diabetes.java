package ai;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.SMO;
/**
 *
 */
public class Diabetes {
    public static void main (String[] args) throws Exception{
        //load dataset
        DataSource source = new DataSource("/home/alex/AI stuff/diabetes.arff");
        Instances dataset = source.getDataSet();
        //set class index to the last attribute
        dataset.setClassIndex(dataset.numAttributes()-1);
        //create and build the classifier!
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(dataset);
        //print out capabilities
        System.out.println(nb.getCapabilities().toString());

        SMO svm = new SMO();
        svm.buildClassifier(dataset);
        System.out.println(svm.getCapabilities().toString());

        String[] options = new String[4];
        options[0] = "-C"; options[1] = "0.11";
        options[2] = "-M"; options[3] = "3";

    }
}
