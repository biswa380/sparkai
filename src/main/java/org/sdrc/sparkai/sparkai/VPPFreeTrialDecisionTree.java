package org.sdrc.sparkai.sparkai;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.slf4j.LoggerFactory;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import static org.apache.spark.sql.functions.*;

public class VPPFreeTrialDecisionTree {

	public static UDF1<String,String> countryGrouping = new UDF1<String,String>() {

		@Override
		public String call(String country) throws Exception {
			List<String> topCountries =  Arrays.asList(new String[] {"GB","US","IN","UNKNOWN"});
			List<String> europeanCountries =  Arrays.asList(new String[] {"BE","BG","CZ","DK","DE","EE","IE","EL","ES","FR","HR","IT","CY","LV","LT","LU","HU","MT","NL","AT","PL","PT","RO","SI","SK","FI","SE","CH","IS","NO","LI","EU"});
			
			if (topCountries.contains(country)) return country; 
			if (europeanCountries .contains(country)) return "EUROPE";
			else return "OTHER";
		}
		
	};
	public static void main(String[] args) {
		System.setProperty("hadoop.home.dir", "C:\\Program Files\\Hadoop");
		Logger root = (Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
		root.setLevel(Level.WARN);

		SparkSession spark = SparkSession.builder()
				.appName("Linear RegressionModel")
				.config("spark.warehouse.dir", "file:///c:/tmp/")
				.master("local[*]").getOrCreate();

		Dataset<Row> vppFreeTrial = spark.read().option("header", true).option("inferSchema", true)
				.csv("E:\\python-workspace\\datasets\\Starting workspaces\\Chapter 10\\vppFreeTrials.csv");
		spark.udf().register("countryGrouping", countryGrouping, DataTypes.StringType);
		vppFreeTrial = vppFreeTrial.withColumn("country", callUDF("countryGrouping", col("country")))
				.withColumn("label", when(col("payments_made").geq(1), lit(1)).otherwise(lit(0)));
		
		StringIndexer countryIndexer = new StringIndexer();
		vppFreeTrial = countryIndexer.setInputCol("country")
						.setOutputCol("country_index")
						.fit(vppFreeTrial)
						.transform(vppFreeTrial);
		
		VectorAssembler vectorAssembler = new VectorAssembler();
		Dataset<Row> inputData = vectorAssembler.setInputCols(new String[] {"country_index", "rebill_period", 
				"chapter_access_count", "seconds_watched"})
		.setOutputCol("features")
		.transform(vppFreeTrial)
		.select("label", "features");
		inputData.show();
		
		Dataset<Row>[] trainingAndHoldOutData = inputData.randomSplit(new double[] {0.8, 0.2});
		Dataset<Row> trainingData = trainingAndHoldOutData[0];
		Dataset<Row> holdOutData = trainingAndHoldOutData[1];
		
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
		evaluator.setMetricName("accuracy");
		
//		Using Decision Tree Classifier
		DecisionTreeClassifier dTreeClassifier = new DecisionTreeClassifier();
		dTreeClassifier.setMaxDepth(5);
		DecisionTreeClassificationModel dtModel = dTreeClassifier.fit(trainingData);
		Dataset<Row> predictions = dtModel.transform(holdOutData);
		predictions.show();
		System.out.println(dtModel.toDebugString());
		System.out.println("The accuracy of the decision tree model is "+evaluator.evaluate(predictions));
		
		
//		Using Random Forest Classifier
		RandomForestClassifier rfClassifier = new RandomForestClassifier();
		rfClassifier.setMaxDepth(5);
		RandomForestClassificationModel rfModel = rfClassifier.fit(trainingData);
		Dataset<Row> randomForestPredictions=rfModel.transform(holdOutData);
		randomForestPredictions.show();
		System.out.println(rfModel.toDebugString());
		System.out.println("The accuracy of the random forest model is "+evaluator.evaluate(randomForestPredictions));
	}

}
