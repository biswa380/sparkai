package org.sdrc.sparkai.sparkai;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.LoggerFactory;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;

public class HouseModelFields {
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
	
	houseData.describe().show();
	
	houseData = houseData.drop("id", "date", "waterfront", "view", "condition", "grade", "yr_renovated", "zipcode", "lat", "long");
	for(String col : houseData.columns())
		System.out.println("The correlation between price and "+col+" is " + houseData.stat().corr("price", col));
	
	//removing variables with low correlation
	houseData = houseData.drop("sqft_lot", "yr_built", "sqft_lot15");
	
	
	//Finding cross correlation between all variables
	for(String col : houseData.columns()) {
		for(String col1 : houseData.columns()) {
		System.out.println("The correlation between "+col+" and "+col1+" is " + houseData.stat().corr(col1, col));
		}
	}
}
}
