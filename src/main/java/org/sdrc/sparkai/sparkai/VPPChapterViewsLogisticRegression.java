package org.sdrc.sparkai.sparkai;

import static org.apache.spark.sql.functions.when;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionSummary;
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
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.slf4j.LoggerFactory;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;

public class VPPChapterViewsLogisticRegression {
	
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
					when(functions.col("next_month_views").$greater(0), 0)
							.otherwise(1))
			.withColumnRenamed("next_month_views_cleaned", "label")
			.drop("last_month_views", "firstSub", "all_time_views", "next_month_views", "is_cancelled");

	vppChapterViewInputData.show();

	vppChapterViewInputData = new StringIndexer()
				.setInputCol("payment_method_type")
				.setOutputCol("payment_method_type_indexer")
				.fit(vppChapterViewInputData)
				.transform(vppChapterViewInputData);

	vppChapterViewInputData = new StringIndexer()
				.setInputCol("country")
				.setOutputCol("country_indexer")
				.fit(vppChapterViewInputData)
				.transform(vppChapterViewInputData);

	vppChapterViewInputData = new StringIndexer()
				.setInputCol("rebill_period_in_months")
				.setOutputCol("rebill_period_indexer")
				.fit(vppChapterViewInputData)
				.transform(vppChapterViewInputData);

	vppChapterViewInputData = new OneHotEncoderEstimator()
				.setInputCols(new String[] { "payment_method_type_indexer", "country_indexer", "rebill_period_indexer" })
				.setOutputCols(new String[] { "payment_method_type_vector", "country_vector", "rebill_period_vector" })
				.fit(vppChapterViewInputData)
				.transform(vppChapterViewInputData);

	vppChapterViewInputData = new VectorAssembler()
				.setInputCols(new String[] { "payment_method_type_vector", "country_vector", "rebill_period_vector", "age" })
				.setOutputCol("features")
				.transform(vppChapterViewInputData)
				.select("label", "features");

	LogisticRegression logisticRegression = new LogisticRegression();

	ParamGridBuilder paramGridBuilder = new ParamGridBuilder();

	ParamMap[] paramMaps = paramGridBuilder.addGrid(logisticRegression.regParam(), new double[] { 0.01, 0.1, 0.3, 0.5, 0.7, 1 })
			.addGrid(logisticRegression.elasticNetParam(), new double[] { 0, 0.5, 1 }).build();

	TrainValidationSplit trainValidationSplit = new TrainValidationSplit().setEstimator(logisticRegression)
			.setEvaluator(new RegressionEvaluator().setMetricName("r2")).setEstimatorParamMaps(paramMaps)
			.setTrainRatio(0.8);

	
	Dataset<Row>[] traindTestAndHoldOutData = vppChapterViewInputData.randomSplit(new double[] { 0.9, 0.1 });
	Dataset<Row> trainAndTestData = traindTestAndHoldOutData[0];
	Dataset<Row> holdOutData = traindTestAndHoldOutData[1];

	TrainValidationSplitModel trainValidationSplitModel = trainValidationSplit.fit(trainAndTestData);
	LogisticRegressionModel lrModel = (LogisticRegressionModel) trainValidationSplitModel.bestModel();

	System.out.println("Accuracy of model is :: " + lrModel.summary().accuracy());
	
	lrModel.transform(trainAndTestData).show();
	LogisticRegressionSummary summary = lrModel.evaluate(holdOutData);
	double truePositive = summary.truePositiveRateByLabel()[1];
	double falsePositive = summary.falsePositiveRateByLabel()[0];
	
	System.out.println("The likelihood of a positive value of being correct is "+truePositive/(truePositive+falsePositive));
	
	lrModel.transform(holdOutData).groupBy("label", "prediction").count().show();
}}
