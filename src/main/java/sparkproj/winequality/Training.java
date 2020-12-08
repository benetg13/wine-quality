package sparkproj.winequality;

import java.io.IOException;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class Training 
{
	public static String training_path = "src/TrainingDataset.csv";

	public static void main(String[] args) 
	{
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
		
		SparkConf conf = new SparkConf()
				.setAppName("Main")
				.setMaster("local[*]");
        
        SparkSession spark = SparkSession
        		.builder()
        		.config(conf)
        		.getOrCreate();

        String[] predictionVariables = 
        		{
        			"fixed acidity",
        			"volatile acidity",
        			"citric acid",
        			"residual sugar",
        			"chlorides",
        			"free sulfur dioxide",
        			"total sulfar dioxide",
        			"density",
        			"pH",
        			"sulphates",
        			"alcohol"
        		};
        
        
        
        VectorAssembler assembler = new VectorAssembler()
        		.setInputCols(predictionVariables)
        		.setOutputCol("features");
		
		
        StructType schema = DataTypes.createStructType(new StructField[] 
        		{
	        		new StructField("fixed acidity", DataTypes.DoubleType, true, Metadata.empty()),
	        		new StructField("volatile acidity", DataTypes.DoubleType, true, Metadata.empty()),
	        		new StructField("citric acid", DataTypes.DoubleType, true, Metadata.empty()),
	        		new StructField("residual sugar", DataTypes.DoubleType, true, Metadata.empty()),
	        		new StructField("chlorides", DataTypes.DoubleType, true, Metadata.empty()),
	        		new StructField("free sulfur dioxide", DataTypes.DoubleType, true, Metadata.empty()),
	        		new StructField("total sulfar dioxide", DataTypes.DoubleType, true, Metadata.empty()),
	        		new StructField("density", DataTypes.DoubleType, true, Metadata.empty()),
	        		new StructField("pH", DataTypes.DoubleType, true, Metadata.empty()),
	        		new StructField("sulphates", DataTypes.DoubleType, true, Metadata.empty()),
	        		new StructField("alcohol", DataTypes.DoubleType, true, Metadata.empty()),
	        		new StructField("quality", DataTypes.IntegerType, true, Metadata.empty())
        		});

        Dataset<Row> training = spark.read()
        		.format("csv")
        		.option("header", "true")
        		.option("delimiter", ";")
        		.schema(schema)
        		.load(training_path);
        
        Dataset<Row> vectorized_training = assembler.transform(training);
        
        StringIndexerModel labelIndexer = new StringIndexer()
        		.setInputCol("quality")
        		.setOutputCol("indexedLabel")
        		.fit(vectorized_training);
        
        Dataset<Row> data = labelIndexer.transform(vectorized_training);

        RandomForestClassificationModel randomForestModel = new RandomForestClassifier()
        		  .setLabelCol("indexedLabel")
        		  .setFeaturesCol("features")
        		  .setNumTrees(10)
        		  .fit(data);
        		
        try
        {
  			randomForestModel.write().overwrite().save("model");
  		} 
        catch (IOException e) 
        {
  			e.printStackTrace();
  		}

	}

}
