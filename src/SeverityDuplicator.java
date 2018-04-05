import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import weka.core.Instance;
import weka.core.Instances;


public class SeverityDuplicator {

	public static void main(String[] args) {
		System.setProperty( "file.encoding", "UTF-8" );
		Instances duplicateData, data = null;
		int numAttributes = 0, duplications = 0, high,normal;
	    try {
	         BufferedReader reader = new BufferedReader( new
	         FileReader("C:\\Users\\boussad\\Documents\\GitHub\\AnnConfig"
	         		+ "\\DATA\\Ant 15\\Severity\\Ant15.Severity.arff"));
	         
	         data = new Instances(reader);
	         reader.close();
	         numAttributes = data.numAttributes() - 1;
	         data.setClassIndex(data.numAttributes() - 1);
	        }

	    catch ( IOException e ){
	         e.printStackTrace();
	        }
	    int numInstances= data.numInstances();
	    System.out.println(numInstances);
	    duplicateData =new Instances(data);
	    duplicateData.delete();
	    System.out.println(numAttributes);
	    System.out.println(data.attribute(numAttributes));
	    System.out.println(data.attribute(numAttributes-2));
	    System.out.println(data.attribute(numAttributes-1));
	    
	    //*
	    for(int i= 0; i<numInstances; i++ ){
	    	Instance crt= data.instance(i);
	    	high =  (int) crt.value(numAttributes-2);
	    	normal= (int) crt.value(numAttributes-1);
	    	if (high==1 || (high ==0 && normal<2)){
	    		duplicateData.add(crt);
	    	}else{
	    		duplications = ((high >1)? high : normal);
	    		for(int j=0; j<duplications;j++){
	    			
	    			System.out.println("classe fautive");
	    			duplicateData.add(crt);
	    		}
	    	}	
	    }
	  
	    for(int i= 0; i<duplicateData.numInstances(); i++ ){
	    	Instance crt= duplicateData.instance(i);
	    	System.out.println("Duplicated");
	    	System.out.println(crt.classValue());
	    }
	    System.out.println(duplicateData.numInstances());
	    
	    BufferedWriter writer;
		try {
			writer = new BufferedWriter(new FileWriter(
					"C:\\Users\\b oussad\\Documents\\GitHub\\AnnConfig"
	         		+ "\\DATA\\Ant 15\\Severity\\Ant15.SeverityDuplicated.arff"));
			writer.write(duplicateData.toString());
		    writer.flush();
		    writer.close();

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		//*/
	}
}
