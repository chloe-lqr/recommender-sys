package test

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, ALS, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by JChubby on 2015/6/29.
 * 流数据测试
 */
object StreamingTest {
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val sparkConf = new SparkConf().setMaster("spark://cloud1:7077")
      .setMaster("spark://cloud1:7077")
      .setAppName("StreamingTest")
      .set("spark.akka.frameSize", "2000")
      .set("spark.network.timeout", "1200")
    val sparkContext = new SparkContext(sparkConf)

    val loadDataTime = System.nanoTime()
    //读取训练数据
    val trainingData = sparkContext.textFile("hdfs://cloud1:9000/data/trainingData").map { line =>
      val fields = line.split("::")
      new Rating(fields(0).toInt, fields(1).toInt, fields(2).toInt)
    }.cache()
    println("[DEBUG]get " + trainingData.count() + " training data from trainingData")
    //读取测试数据
    val testData = sparkContext.textFile("hdfs://cloud1:9000/data/testData").map { line =>
      val fields = line.split("::")
      new Rating(fields(0).toInt, fields(1).toInt, fields(2).toInt)
    }.cache()
    println("[DEBUG]get " + testData.count() + " test data from testData")
    println("[DEBUG]load data consuming:" + (System.nanoTime() - loadDataTime) / 1000000000 + "s")

    val rank = 5
    val lambda = 0.1
    val numIter = 7

    val startTime = System.nanoTime()
    val model = ALS.train(trainingData, rank, numIter, lambda)
    println("[DEBUG]training consuming:" + ((System.nanoTime() - startTime) / 1000000000) + " s")
    val mse_rmse = mseAndRmse(model, testData)
    println("[DEBUG]MSE in this model is :" + mse_rmse._1 + " RMSE is : " + mse_rmse._2)

    //开始接受Socket流数据并更新模型
    val streamingContext = new StreamingContext(sparkContext, Seconds(60))
    val socketRatings = streamingContext.socketTextStream("cloud1", 9999, StorageLevel.MEMORY_ONLY)
      .flatMap(_.split(" ")).map { lines =>
      val fields = lines.split("::")
      new Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }

    var allData = trainingData
    allData.cache().count()
    var index = 0
    socketRatings.foreachRDD { rdd =>
      index += 1
      println("\n[DEBUG]this round (" + index + ") received: " + rdd.count + " data lines.")
      val refreshStartTime = System.nanoTime()
      val tmpData = allData.union(rdd).cache()
      tmpData.count()
      allData = tmpData
      tmpData.unpersist()
      val model = ALS.train(allData, rank, numIter, lambda)
      val refreshEndTime = System.nanoTime() - refreshStartTime
      println("[DEBUG]training consuming:" + refreshEndTime / 1000000000 + " s")
      val mse_rmse = mseAndRmse(model, testData)
      println("[DEBUG]MSE in this model is :" + mse_rmse._1 + " RMSE is : " + mse_rmse._2)
    }
    sparkContext.stop()
  }

  //计算MSE和RMSE
  def mseAndRmse(model: MatrixFactorizationModel, testData: RDD[Rating]): (Double, Double) = {
    val prediction = model.predict(testData.map(x => (x.user, x.product)))
    val preAndActual = prediction.map(x => ((x.user, x.product), x.rating))
      .join(testData.map(x => ((x.user, x.product), x.rating)))
      .values

    val mse = preAndActual.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / testData.count()
    val rmse = math.sqrt(preAndActual.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / testData.count())

    (mse, rmse)
  }
}
