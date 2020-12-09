package sparkproj.winequality;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
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

public class Predict 
{
	public static String validation_path = "src/ValidationDataset.csv";
	
	public static void main(String args[])
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
        
        Dataset<Row> validation = spark.read()
        		.format("csv")
        		.option("header", "true")
        		.option("delimiter", ";")
        		.schema(schema)
        		.load(validation_path);
        
        Dataset<Row> vectorized_validation = assembler.transform(validation);
        
        StringIndexerModel labelIndexer = new StringIndexer()
        		.setInputCol("quality")
        		.setOutputCol("indexedLabel")
        		.fit(vectorized_validation);
        
        Dataset<Row> data = labelIndexer.transform(vectorized_validation);

        
        RandomForestClassificationModel randomForestModel = RandomForestClassificationModel.read().load("model"); 

        Dataset<Row> predictions = randomForestModel.transform(data);

        predictions.select("prediction", "indexedLabel", "features").show(10);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setPredictionCol("prediction")
          .setMetricName("accuracy");
         
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Accuracy = " + accuracy);
        
        spark.stop();
        
	}

}
