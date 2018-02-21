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


public class ANNSeverity {
	
	
	private final static int MaxLayers=21, folds = 10, places =3;
	private final static double Correction = 0.1, K=0.25;
	private static double R0,R1,R2,R3;
	
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
					//* 
					 FileReader("E:\\Maitrise Uqtr\\Sujet de recherche\\AnnConfig\\DATA\\Ant 15\\Severity"
					 		+ "\\Ant15SeverityD(Loc,Rfc).arff") );
					//*/
					
					/*
					FileReader("E:\\Maitrise Uqtr\\Sujet de recherche\\AnnConfig\\DATA\\Ant 15\\Normal\\Duplicated"
							+ "\\Ant15NBugsDuplicated.D(LOC,fanin).arff") );
					//*/
			
					/*
						FileReader("E:\\Maitrise Uqtr\\Sujet de recherche\\AnnConfig\\DATA\\Ant 17\\Binary\\Balanced\\Balanced 35-65"
							+ "\\Ant15NBugsDuplicated.D(LOC,Ce).arff") );
					//*/
			
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
		
		R0= (nvEval.recall(0) > 0) ? nvEval.recall(0) : Correction;
		R1= (nvEval.recall(1) > 0) ? nvEval.recall(1) : Correction;
		R2= (nvEval.recall(2) > 0) ? nvEval.recall(2) : Correction;
		R3= (nvEval.recall(3) > 0) ? nvEval.recall(3) : Correction;
		gMean= Math.pow(R0*R1*R2*R3,K);
		
		System.out.println(roundAvoid(nvEval.weightedTruePositiveRate(), places)+
 				"\t"+ roundAvoid(nvEval.weightedTrueNegativeRate(), places)+
						"\t"+ roundAvoid(gMean, places)+
							"\t"+ roundAvoid(nvEval.weightedFMeasure(), places)+
								"\t"+ roundAvoid(nvEval.weightedAreaUnderROC(), places));
		//*/
		
		//*J48
		J48 j48= new J48();
		Evaluation jEval =new Evaluation(data);
		jEval.crossValidateModel(j48, data, folds, new Random(1));
		
		R0= (jEval.recall(0) > 0) ? jEval.recall(0) : Correction;
		R1= (jEval.recall(1) > 0) ? jEval.recall(1) : Correction;
		R2= (jEval.recall(2) > 0) ? jEval.recall(2) : Correction;
		R3= (jEval.recall(3) > 0) ? jEval.recall(3) : Correction;
		gMean= Math.pow(R0*R1*R2*R3,K);
		
		System.out.println(roundAvoid(jEval.weightedTruePositiveRate(), places)+
 				"\t"+ roundAvoid(jEval.weightedTrueNegativeRate(), places)+
						"\t"+ roundAvoid(gMean, places)+
							"\t"+ roundAvoid(jEval.weightedFMeasure(), places)+
								"\t"+ roundAvoid(jEval.weightedAreaUnderROC(), places));
		//*/
		
		//*Random Forest
		RandomForest rf= new RandomForest();
		Evaluation rfEval =new Evaluation(data);
		rfEval.crossValidateModel(rf, data, folds, new Random(1));
		
		R0= (rfEval.recall(0) > 0) ? rfEval.recall(0) : Correction;
		R1= (rfEval.recall(1) > 0) ? rfEval.recall(1) : Correction;
		R2= (rfEval.recall(2) > 0) ? rfEval.recall(2) : Correction;
		R3= (rfEval.recall(3) > 0) ? rfEval.recall(3) : Correction;
		gMean= Math.pow(R0*R1*R2*R3,K);
		
		System.out.println(roundAvoid(rfEval.weightedTruePositiveRate(), places)+
 				"\t"+ roundAvoid(rfEval.weightedTrueNegativeRate(), places)+
						"\t"+ roundAvoid(gMean, places)+
							"\t"+ roundAvoid(rfEval.weightedFMeasure(), places)+
								"\t"+ roundAvoid(rfEval.weightedAreaUnderROC(), places));
		//*/
		
		//*RLog
		Logistic rLog= new Logistic();
		Evaluation rLogEval =new Evaluation(data);
		rLogEval.crossValidateModel(rLog, data, folds, new Random(1));

		R0= (rLogEval.recall(0) > 0) ? rLogEval.recall(0) : Correction;
		R1= (rLogEval.recall(1) > 0) ? rLogEval.recall(1) : Correction;
		R2= (rLogEval.recall(2) > 0) ? rLogEval.recall(2) : Correction;
		R3= (rLogEval.recall(3) > 0) ? rLogEval.recall(3) : Correction;
		gMean= Math.pow(R0*R1*R2*R3,K);
		
		System.out.println(roundAvoid(rLogEval.weightedTruePositiveRate(), places)+
 				"\t"+ roundAvoid(rLogEval.weightedTrueNegativeRate(), places)+
						"\t"+ roundAvoid(gMean, places)+
							"\t"+ roundAvoid(rLogEval.weightedFMeasure(), places)+
								"\t"+ roundAvoid(rLogEval.weightedAreaUnderROC(), places));		
		//*/
		
		//*SVM
		SMO svm= new SMO();
		Evaluation svmEval =new Evaluation(data);
		svmEval.crossValidateModel(svm, data, folds, new Random(1));

		R0= (svmEval.recall(0) > 0) ? svmEval.recall(0) : Correction;
		R1= (svmEval.recall(1) > 0) ? svmEval.recall(1) : Correction;
		R2= (svmEval.recall(2) > 0) ? svmEval.recall(2) : Correction;
		R3= (svmEval.recall(3) > 0) ? svmEval.recall(3) : Correction;
		gMean= Math.pow(R0*R1*R2*R3,K);
		
		System.out.println(roundAvoid(svmEval.weightedTruePositiveRate(), places)+
 				"\t"+ roundAvoid(svmEval.weightedTrueNegativeRate(), places)+
					"\t"+ roundAvoid(gMean, places)+
						"\t"+ roundAvoid(svmEval.weightedFMeasure(), places)+
							"\t"+ roundAvoid(svmEval.weightedAreaUnderROC(), places));
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
			
			R0= (eval.recall(0) > 0) ? eval.recall(0) : Correction;
			R1= (eval.recall(1) > 0) ? eval.recall(1) : Correction;
			R2= (eval.recall(2) > 0) ? eval.recall(2) : Correction;
			R3= (eval.recall(3) > 0) ? eval.recall(3) : Correction;
			gMean= Math.pow(R0*R1*R2*R3,K);
			
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
				
			R0= (eval.recall(0) > 0) ? eval.recall(0) : Correction;
			R1= (eval.recall(1) > 0) ? eval.recall(1) : Correction;
			R2= (eval.recall(2) > 0) ? eval.recall(2) : Correction;
			R3= (eval.recall(3) > 0) ? eval.recall(3) : Correction;
			gMean= Math.pow(R0*R1*R2*R3,K);

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

				R0= (eval.recall(0) > 0) ? eval.recall(0) : Correction;
				R1= (eval.recall(1) > 0) ? eval.recall(1) : Correction;
				R2= (eval.recall(2) > 0) ? eval.recall(2) : Correction;
				R3= (eval.recall(3) > 0) ? eval.recall(3) : Correction;
				gMean= Math.pow(R0*R1*R2*R3,K);
				
				if(gMean>MaxGMean){
					bestLayers= layer;
					best = eval;
					MaxGMean= gMean;
				}
			}
		}
			
		//*/
		
		System.out.println(roundAvoid(best.weightedTruePositiveRate(), places)+
				 		"\t"+ roundAvoid(best.weightedTrueNegativeRate(), places)+
							"\t"+ roundAvoid(MaxGMean, places)+
								"\t"+ roundAvoid(best.weightedFMeasure(), places)+
									"\t"+ roundAvoid(best.weightedAreaUnderROC(), places));
		System.out.println("Best hidden layers config is : "+bestLayers);
	}
	
}
