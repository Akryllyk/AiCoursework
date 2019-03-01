package ai;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 */
public class BankDetails {
    public static void main (String[] args) throws Exception{
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("/home/alex/AI stuff/bank-data.csv");
        Instances dataSet = source.getDataSet();

        //region
        Instances region = new Instances(dataSet, 0, dataSet.numInstances());
        int innerCity = region.numAttributes();
        for (int i = 0; i <innerCity ; i++) {
            if(!region.attribute(0).toString().equalsIgnoreCase(dataSet.attribute(3).toString())){
                region.deleteAttributeAt(0);
                innerCity = region.numAttributes();
                if(!region.attribute(1).toString().equalsIgnoreCase(dataSet.attribute(4).toString())){
                    region.deleteAttributeAt(i);
                    innerCity = region.numAttributes();
                }
            }
        }
        for (int i = 0; i < 7; i++) {
            region.deleteAttributeAt(2);
        }

        //age
        Instances age = new Instances (dataSet, 0, dataSet.numInstances());
        age.deleteAttributeAt(0);
        age.deleteAttributeAt(1);
        for (int i = 0; i < 7; i++) {
            age.deleteAttributeAt(3);
        }

        //marriage
        Instances marriage = new Instances(dataSet, 0, dataSet.numInstances());
        for (int i = 0; i < 5; i++) {
            marriage.deleteAttributeAt(0);
        }
        for (int i = 0; i < 4; i++) {
            marriage.deleteAttributeAt(1);
        }
        marriage.deleteAttributeAt(2);

        //region vs income
        SimpleKMeans regionModel = new SimpleKMeans();
        regionModel.setNumClusters(4);
        regionModel.buildClusterer(region);
        ClusterEvaluation regionClusterEval = new ClusterEvaluation();
        regionClusterEval.setClusterer(regionModel);
        regionClusterEval.evaluateClusterer(region);
        System.out.println(regionClusterEval.clusterResultsToString());

        //age vs region and income
        SimpleKMeans ageModel = new SimpleKMeans();
        ageModel.setNumClusters(10);
        ageModel.buildClusterer(age);
        ClusterEvaluation ageEvaluation = new ClusterEvaluation();
        ageEvaluation.setClusterer(ageModel);
        ageEvaluation.evaluateClusterer(age);
        System.out.println(ageEvaluation.clusterResultsToString());

        //marriage vs mortgage
        SimpleKMeans marriageMortgageModel = new SimpleKMeans();
        marriageMortgageModel.setNumClusters(10);
        marriageMortgageModel.buildClusterer(marriage);
        ClusterEvaluation marriageEval = new ClusterEvaluation();
        marriageEval.setClusterer(marriageMortgageModel);
        marriageEval.evaluateClusterer(marriage);
        System.out.println(marriageEval.clusterResultsToString());
    }
}
