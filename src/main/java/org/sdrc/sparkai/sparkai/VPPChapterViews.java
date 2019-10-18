package org.sdrc.sparkai.sparkai;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.slf4j.LoggerFactory;
import org.spark_project.dmg.pmml.TrainingInstances;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;

import static org.apache.spark.sql.functions.when;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
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

public class VPPChapterViews {
	public static void main(String[] args) {

		System.setProperty("hadoop.home.dir", "C:\\Program Files\\Hadoop");
		Logger root = (Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
		root.setLevel(Level.WARN);

		SparkSession spark = SparkSession.builder().appName("Linear RegressionModel")
				.config("spark.warehouse.dir", "file:///c:/tmp/").master("local[*]").getOrCreate();

		Dataset<Row> vppChapterViewData = spark.read().option("header", true).option("inferSchema", true)
				.csv("E:\\python-workspace\\datasets\\vppChapterViews\\*.csv");

		vppChapterViewData = vppChapterViewData.filter("is_cancelled = false");

		Dataset<Row> vppChapterViewInputData = vppChapterViewData
				.withColumn("firstSub_cleaned",
						when(functions.col("firstSub").isNull(), 0).otherwise(functions.col("firstSub")))
				.withColumn("last_month_views_cleaned",
						when(functions.col("last_month_views").isNull(), 0)
								.otherwise(functions.col("last_month_views")))
				.withColumn("all_time_views_cleaned",
						when(functions.col("all_time_views").isNull(), 0).otherwise(functions.col("all_time_views")))
				.withColumn("next_month_views_cleaned",
						when(functions.col("next_month_views").isNull(), 0)
								.otherwise(functions.col("next_month_views")))
				.withColumnRenamed("next_month_views_cleaned", "label")
				.drop("last_month_views", "firstSub", "all_time_views", "next_month_views", "is_cancelled");

		vppChapterViewInputData.show();

		Dataset<Row>[] traindTestAndHoldOutData = vppChapterViewInputData.randomSplit(new double[] { 0.9, 0.1 });
		Dataset<Row> trainAndTestData = traindTestAndHoldOutData[0];
		Dataset<Row> holdOutData = traindTestAndHoldOutData[1];

		StringIndexer paymentMethodIndexer = new StringIndexer();
		paymentMethodIndexer.setInputCol("payment_method_type");
		paymentMethodIndexer.setOutputCol("payment_method_type_indexer");

		StringIndexer countryIndexer = new StringIndexer();
		countryIndexer.setInputCol("country");
		countryIndexer.setOutputCol("country_indexer");

		StringIndexer rebillPeriodIndexer = new StringIndexer();
		rebillPeriodIndexer.setInputCol("rebill_period_in_months");
		rebillPeriodIndexer.setOutputCol("rebill_period_indexer");

		OneHotEncoderEstimator estimator = new OneHotEncoderEstimator();
		estimator.setInputCols(new String[] { "payment_method_type_indexer", "country_indexer", "rebill_period_indexer" });
		estimator.setOutputCols(new String[] { "payment_method_type_vector", "country_vector", "rebill_period_vector" });

		VectorAssembler vectorAssembler = new VectorAssembler()
				.setInputCols(new String[] { "payment_method_type_vector", "country_vector", "rebill_period_vector", "age", 
						"firstSub_cleaned", "last_month_views_cleaned", "all_time_views_cleaned" })
				.setOutputCol("features");

		LinearRegression linearRegression = new LinearRegression();

		ParamGridBuilder paramGridBuilder = new ParamGridBuilder();

		ParamMap[] paramMaps = paramGridBuilder.addGrid(linearRegression.regParam(), new double[] { 0.1, 0.3, 0.5, 0.7, 1 })
				.addGrid(linearRegression.elasticNetParam(), new double[] { 0.01, 0.5, 1 }).build();

		TrainValidationSplit trainValidationSplit = new TrainValidationSplit().setEstimator(linearRegression)
				.setEvaluator(new RegressionEvaluator().setMetricName("r2")).setEstimatorParamMaps(paramMaps)
				.setTrainRatio(0.9);

		Pipeline pipeline = new Pipeline();
		pipeline.setStages(new PipelineStage[] { paymentMethodIndexer, countryIndexer, rebillPeriodIndexer, estimator, vectorAssembler, trainValidationSplit });
		PipelineModel pipelineModel = pipeline.fit(trainAndTestData);

		TrainValidationSplitModel trainValidationSplitModel = (TrainValidationSplitModel) pipelineModel.stages()[5];
		LinearRegressionModel lrModel = (LinearRegressionModel) trainValidationSplitModel.bestModel();

		pipelineModel.transform(trainAndTestData).show();

		System.out.println("Training data RMSE :: " + lrModel.summary().rootMeanSquaredError() + " r2 :: "
				+ lrModel.summary().r2());
//		System.out.println("Test data RMSE :: " + lrModel.evaluate(holdOutData).rootMeanSquaredError() + " r2 :: "
//				+ lrModel.evaluate(holdOutData).r2());
	}
}
