package org.sdrc.sparkai.sparkai;

import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.slf4j.LoggerFactory;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;

public class HousePriceModel {
	public static void main(String[] args) {
		
		System.setProperty("hadoop.home.dir", "C:\\Program Files\\Hadoop");
		Logger root = (Logger)LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
		root.setLevel(Level.WARN);
		
		SparkSession spark = SparkSession.builder()
				.appName("Linear RegressionModel")
				.config("spark.warehouse.dir", "file:///c:/tmp/")
				.master("local[*]")
				.getOrCreate();
		
		Dataset<Row> houseData = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("E:\\python-workspace\\datasets\\kc_house_data.csv");
		houseData = houseData.withColumn("sqft_above_percentage", functions.col("sqft_above").divide(functions.col("sqft_living")));
		houseData.show();
		
		StringIndexer conditionIndex = new StringIndexer();
		conditionIndex.setInputCol("condition");
		conditionIndex.setOutputCol("conditionIndex");
		houseData = conditionIndex.fit(houseData).transform(houseData);
		
		StringIndexer gradeIndex = new StringIndexer();
		gradeIndex.setInputCol("grade");
		gradeIndex.setOutputCol("gardeIndex");
		houseData=gradeIndex.fit(houseData).transform(houseData);
		
		StringIndexer zipcodeIndex = new StringIndexer();
		zipcodeIndex.setInputCol("zipcode");
		zipcodeIndex.setOutputCol("zipcodeIndex");
		houseData = zipcodeIndex.fit(houseData).transform(houseData);
		
		OneHotEncoderEstimator encoder=new OneHotEncoderEstimator();
		encoder.setInputCols(new String[] {"conditionIndex", "gardeIndex", "zipcodeIndex"});
		encoder.setOutputCols(new String[] {"conditionVector", "gradeVector", "zipcodeVector"});
		houseData = encoder.fit(houseData).transform(houseData);
		
//		houseData.show();
		
		VectorAssembler vectorAssembler = new VectorAssembler();
		vectorAssembler.setInputCols(new String[] {"bedrooms", "bathrooms", "sqft_living", "floors","sqft_above_percentage", "conditionVector", 
				"gradeVector","zipcodeVector", "waterfront"});
		vectorAssembler.setOutputCol("features");
		
		Dataset<Row> inputData = vectorAssembler.transform(houseData)
				.select("price", "features")
				.withColumnRenamed("price", "label");
		
		Dataset<Row>[] dataSplits=inputData.randomSplit(new double[] {0.8, 0.2});
		Dataset<Row> trainingAndTestData = dataSplits[0];
		Dataset<Row> holdOutData = dataSplits[1];
		
		
		LinearRegression linearRegression = new LinearRegression();
		
		ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
		
		ParamMap[] paramMap = paramGridBuilder.addGrid(linearRegression.regParam(), new double[] {0.01,0.1,0.5})
				.addGrid(linearRegression.elasticNetParam(), new double[] {0.1,0.5,1})
				.build();
		
		TrainValidationSplit validator = new TrainValidationSplit()
				.setEstimator(linearRegression)
				.setEvaluator(new RegressionEvaluator().setMetricName("r2"))
				.setEstimatorParamMaps(paramMap);
		
		TrainValidationSplitModel tvModel = validator.fit(trainingAndTestData);
		LinearRegressionModel model = (LinearRegressionModel) tvModel.bestModel();
		
		System.out.println("Training R2 :: "+model.summary().r2());
		System.out.println("Training RMSE :: "+model.summary().rootMeanSquaredError());
		
		model.transform(holdOutData).show();
		
		System.out.println("Test R2 :: "+model.evaluate(holdOutData).r2());
		System.out.println("Test RMSE :: "+model.evaluate(holdOutData).rootMeanSquaredError());
		
		System.out.println("Best model ENP :: " +model.getElasticNetParam());
		System.out.println("Best model regParam :: " +model.getRegParam());
	}
}
