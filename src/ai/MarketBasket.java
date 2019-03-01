package ai;

import weka.associations.FPGrowth;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 */
public class MarketBasket {
    public static void main(String[] args) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("/home/alex/AI stuff/marketbasket.arff");
        Instances instance = new Instances(source.getDataSet());

        System.out.println(instance.numInstances());
        FPGrowth model = new FPGrowth();
        model.setLowerBoundMinSupport(6.5);
        model.buildAssociations(instance);

        System.out.println(model);

    }
}

