import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;


public class ANNMax {
	
	
	private final static int MaxLayers=21, folds = 10, places =3;
	
	private static double roundAvoid(double value, int places) {
	    double scale = Math.pow(10, places);
	    return Math.round(value * scale) / scale;
	}
	
	public static void main(String[] args) throws Exception {
		System.setProperty( "file.encoding", "UTF-8" );
		Instances data=null;

		// loads data and set class index
		try {
			BufferedReader reader = new BufferedReader( new
					//FileReader("E:\\Maitrise Uqtr\\Sujet de recherche\\AnnConfig\\DATA\\Ant 17\\Binary"
					FileReader("E:\\Maitrise Uqtr\\Sujet de recherche\\AnnConfig\\DATA\\Ant 17\\Binary\\Duplicated"
					//FileReader("E:\\Maitrise Uqtr\\Sujet de recherche\\AnnConfig\\DATA\\Ant 15\\Binary\\Balanced"
							+ "\\Ant17BinaryDuplicatedD(d(LOC,Ce),d(RFC, Ca)).arff") );

			data = new Instances(reader);
			reader.close();
			data.setClassIndex(data.numAttributes() - 1);
			System.out.println(data.classAttribute());
		}

		catch ( IOException e ){
			e.printStackTrace();
		}
		
		double gMean;

		//*Naive Bayes
		weka.classifiers.bayes.NaiveBayes nv = new weka.classifiers.bayes.NaiveBayes();
		Evaluation nvEval =new Evaluation(data);
		nvEval.crossValidateModel(nv, data, folds, new Random(1));
		gMean= Math.sqrt(nvEval.truePositiveRate(1)* nvEval.truePositiveRate(0));
		System.out.println(roundAvoid(nvEval.weightedTruePositiveRate(), places)+
 				"\t"+ roundAvoid(nvEval.truePositiveRate(1), places)+
					"\t"+ roundAvoid(nvEval.truePositiveRate(0), places)+
						"\t"+ roundAvoid(gMean, places));
		//*/
		
		//*J48
		J48 j48= new J48();
		Evaluation jEval =new Evaluation(data);
		jEval.crossValidateModel(j48, data, folds, new Random(1));
		gMean= Math.sqrt(jEval.truePositiveRate(1)* jEval.truePositiveRate(0));
		
		System.out.println(roundAvoid(jEval.fMeasure(1), places)+
 				"\t"+ roundAvoid(jEval.truePositiveRate(1), places)+
					"\t"+ roundAvoid(jEval.truePositiveRate(0), places)+
						"\t"+ roundAvoid(gMean, places));
		//*/
		
		//*Random Forest
		RandomForest rf= new RandomForest();
		Evaluation rfEval =new Evaluation(data);
		rfEval.crossValidateModel(rf, data, folds, new Random(1));
		gMean= Math.sqrt(rfEval.truePositiveRate(1)* rfEval.truePositiveRate(0));
		
		System.out.println(roundAvoid(rfEval.weightedTruePositiveRate(), places)+
 				"\t"+ roundAvoid(rfEval.truePositiveRate(1), places)+
					"\t"+ roundAvoid(rfEval.truePositiveRate(0), places)+
						"\t"+ roundAvoid(gMean, places));
		//*/
		
		//*RLog
		Logistic rLog= new Logistic();
		Evaluation rLogEval =new Evaluation(data);
		rLogEval.crossValidateModel(rLog, data, folds, new Random(1));
		gMean= Math.sqrt(rLogEval.truePositiveRate(1)* rLogEval.truePositiveRate(0));
		
		System.out.println(roundAvoid(rLogEval.weightedTruePositiveRate(), places)+
 				"\t"+ roundAvoid(rLogEval.truePositiveRate(1), places)+
					"\t"+ roundAvoid(rLogEval.truePositiveRate(0), places)+
						"\t"+ roundAvoid(gMean, places));
		//*/
		
		//*SVM
		SMO svm= new SMO();
		Evaluation svmEval =new Evaluation(data);
		svmEval.crossValidateModel(svm, data, folds, new Random(1));
		gMean= Math.sqrt(svmEval.truePositiveRate(1)* svmEval.truePositiveRate(0));
		
		System.out.println(roundAvoid(svmEval.weightedTruePositiveRate(), places)+
 				"\t"+ roundAvoid(svmEval.truePositiveRate(1), places)+
					"\t"+ roundAvoid(svmEval.truePositiveRate(0), places)+
					"\t"+ roundAvoid(gMean, places));
		//*/
		
		//* MLP
		MultilayerPerceptron mlp= new MultilayerPerceptron();
		
		
		// Set Options
		mlp.setLearningRate(0.1);
		mlp.setMomentum(0.2);
		mlp.setTrainingTime(500);

		// other options
		//int seed  = 0; Todo use seed
		double MaxGMean=-1; 
		gMean =0;
		Evaluation best = null, eval= new Evaluation(data);
		int bestLayer1 =0, bestLayer2=0;
		String bestLayers="", layer="";
		
		//MyMLP10CrossValidation validation = new MyMLP10CrossValidation();
		
		//1st Layer Variation
		for (int l=1;l<MaxLayers; l++){
			layer=Integer.toString(l);
			//System.out.println("Hidden layers config : "+layer);
			mlp.setHiddenLayers(layer);
			eval.crossValidateModel(mlp, data, folds, new Random(1));
			//eval= validation.CrossValidationRun(data, mlp, seed, folds);
			gMean= Math.sqrt(eval.truePositiveRate(1)* eval.truePositiveRate(0));
			if(gMean>MaxGMean){
				bestLayer1 =l;
				bestLayers= layer;
				best = eval;
				MaxGMean= gMean;
			}
		}
		
		//*2nd Layer Variation
		for (int j=1;j<MaxLayers;j++){
			layer=Integer.toString(bestLayer1)+","+Integer.toString(j);
			//System.out.println("Hidden layers config : "+layer);
			mlp.setHiddenLayers(layer);
			eval.crossValidateModel(mlp, data, folds, new Random(1));
			//eval= validation.CrossValidationRun(data, mlp, seed, folds);
			gMean= Math.sqrt(eval.truePositiveRate(1)* eval.truePositiveRate(0));
			if(gMean>MaxGMean){
				bestLayer2=j;
				bestLayers= layer;
				best = eval;
				MaxGMean= gMean;
			}
		}
		//*/
		
		//* 3rd Layer variation
		if (bestLayer2>0){
			for (int k=1;k<MaxLayers;k++){
				layer=Integer.toString(bestLayer1)+","+Integer.toString(bestLayer2)+
						","+Integer.toString(k);
				//System.out.println("Hidden layers config : "+layer);
				mlp.setHiddenLayers(layer);
				eval.crossValidateModel(mlp, data, folds, new Random(1));
				//eval= validation.CrossValidationRun(data, mlp, seed, folds);
				gMean= Math.sqrt(eval.truePositiveRate(1)* eval.truePositiveRate(0));
				if(gMean>MaxGMean){
					bestLayers= layer;
					best = eval;
					MaxGMean= gMean;
				}
			}
		}
			
		//*/
		
		System.out.println(roundAvoid(best.weightedTruePositiveRate(), places)+
				 		"\t"+ roundAvoid(best.truePositiveRate(1), places)+
							"\t"+ roundAvoid(best.truePositiveRate(0), places)+
								"\t"+ roundAvoid(MaxGMean, places));
		System.out.println("Best hidden layers config is : "+bestLayers);
	}
	
}
