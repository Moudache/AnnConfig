import java.io.BufferedReader;

import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;

import java.io.FileReader;
import java.io.IOException;
import java.util.Random;


public class ANNMax {
	
	private final static int MaxLayers=21;
	
	public static void main(String[] args) throws Exception {
		System.setProperty( "file.encoding", "UTF-8" );
		Instances data=null;

		// loads data and set class index
		try {
			BufferedReader reader = new BufferedReader( new
					FileReader("X:\\AnnConfig\\DATA\\Ant 16\\Binary"
							+ "\\Balanced\\Ant16BugsBinaryBalancedLocCbo.arff") );

			data = new Instances(reader);
			reader.close();
			data.setClassIndex(data.numAttributes() - 1);
		}

		catch ( IOException e ){
			e.printStackTrace();
		}

		//*
		MultilayerPerceptron mlp= new MultilayerPerceptron();
		// Set Options
		mlp.setLearningRate(0.1);
		mlp.setMomentum(0.2);
		mlp.setTrainingTime(500);

		// other options
		//int seed  = 0; Todo use seed
		int folds = 10;
		double MaxGMean=-1, gMean =0;
		Evaluation best = null, eval= new Evaluation(data);
		int bestLayer1 =0, bestLayer2=0;
		String bestLayers="", layer="";
		
		//MyMLP10CrossValidation validation = new MyMLP10CrossValidation();
		
		//1st Layer Variation
		for (int l=1;l<MaxLayers; l++){
			layer=Integer.toString(l);
			System.out.println("Hidden layers config : "+layer);
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
			System.out.println("Hidden layers config : "+layer);
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
		
		System.out.println("\nMax Fmeasure = "+best.fMeasure(1)+
				 				"\nTPR ="+ best.truePositiveRate(1)+
									"\n TNR ="+ best.truePositiveRate(0)+
									"\n g-Mean = "+ MaxGMean);
		System.out.println("Best hidden layers config is : "+bestLayers);
	}
	
}
