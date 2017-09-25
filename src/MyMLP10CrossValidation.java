import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;


public class MyMLP10CrossValidation {

	double CrossValidationRun(Instances data, MultilayerPerceptron mlp, int seed, int folds){

		Random rand = new Random(seed);
		Instances randData = new Instances(data);
		randData.randomize(rand);
		if (randData.classAttribute().isNominal())
			randData.stratify(folds);

		Instances predictedData = null;
		
		try {
			Evaluation eval = new Evaluation(randData);

			//Perform 10-Cross Validation
			for (int n = 0; n < folds; n++) {
				Instances train = randData.trainCV(folds, n);
				Instances test = randData.testCV(folds, n);
				// the above code is used by the StratifiedRemoveFolds filter, the
				// code below by the Explorer/Experimenter:
				train = randData.trainCV(folds, n, rand);

				// build and evaluate classifier
				Classifier clsCopy = mlp;
				clsCopy.buildClassifier(train);
				eval.evaluateModel(clsCopy, test);

				// add predictions
				AddClassification filter = new AddClassification();
				filter.setClassifier(mlp);
				filter.setOutputClassification(true);
				filter.setOutputDistribution(true);
				filter.setOutputErrorFlag(true);
				filter.setInputFormat(train);
				Filter.useFilter(train, filter);  // trains the classifier
				Instances pred = Filter.useFilter(test, filter);  // perform predictions on test set
				if (predictedData == null)
					predictedData = new Instances(pred, 0);
				for (int j = 0; j < pred.numInstances(); j++)
					predictedData.add(pred.instance(j));
			}

			// output evaluation
			System.out.println();
			System.out.println("=== Setup ===");
			System.out.println("Classifier: " + mlp.getClass().getName() + " " +
					Utils.joinOptions((mlp).getOptions()));
			System.out.println("Dataset: " + data.relationName());
			System.out.println("Folds: " + folds);
			System.out.println("Seed: " + seed);
			System.out.println();
			System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));
			System.out.println("TPR = "+eval.truePositiveRate(1));
			System.out.println("TNR = "+eval.truePositiveRate(0));
			System.out.println("f-Measure = "+eval.fMeasure(1));
			double gMean= Math.sqrt(eval.truePositiveRate(1)* eval.truePositiveRate(0));
			System.out.println("g-Mean = "+ gMean);
			return gMean; 
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return -1;

	}
}
