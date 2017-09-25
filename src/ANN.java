import java.io.BufferedReader;

import weka.core.Instances;
import weka.classifiers.functions.MultilayerPerceptron;
import java.io.FileReader;
import java.io.IOException;

public class ANN {
	private final static int MaxLayers=21;
	
	public static void main(String[] args) {
		System.setProperty( "file.encoding", "UTF-8" );
		Instances data=null;

		// loads data and set class index
		try {
			BufferedReader reader = new BufferedReader( new
					FileReader("E:\\Maitrise Uqtr\\Sujet de recherche\\DATA\\Ant 15\\"
							+ "Ant15.BugsBinaryDuplicatedLocCa.arff") );

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
		int seed  = 0; //Todo use seed
		int folds = 10;
		double MaxGMean=0;
		double gMean =0;
		String bestLayers="";
		String layer="";
		
		MyMLP10CrossValidation validation = new MyMLP10CrossValidation();
		//1st Layer Variation
		for (int l=1;l<MaxLayers; l++){
			layer=Integer.toString(l);
			System.out.println("Hidden layers config : "+layer);
			mlp.setHiddenLayers(layer);
			gMean= validation.CrossValidationRun(data, mlp, seed, folds);
			if(gMean>MaxGMean){
				bestLayers= layer;
				MaxGMean= gMean;
			}
			System.out.println("\n\n");
			//*2nd Layer Variation
			for (int j=1;j<MaxLayers;j++){
				layer=Integer.toString(l)+","+Integer.toString(j);
				System.out.println("Hidden layers config : "+layer);
				mlp.setHiddenLayers(layer);
				gMean= validation.CrossValidationRun(data, mlp, seed, folds);
				if(gMean>MaxGMean){
					bestLayers= layer;
					MaxGMean= gMean;
				}
				//* 3rd Layer variation
				for (int k=1;k<MaxLayers;k++){
					layer=Integer.toString(l)+","+Integer.toString(j)+","+Integer.toString(k);
					System.out.println("Hidden layers config : "+layer);
					mlp.setHiddenLayers(layer);
					gMean= validation.CrossValidationRun(data, mlp, seed, folds);
					if(gMean>MaxGMean){
						bestLayers= layer;
						MaxGMean= gMean;
					}
				}
				//*/
			}
			//*/
		}
		System.out.println("\nMax g-Mean = "+ MaxGMean);
		System.out.println("Best hidden layers config is : "+bestLayers);
	}
	
}
