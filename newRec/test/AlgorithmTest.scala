package test

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

/**
 * Created by JChubby on 2015/6/23.
 * 算法最佳参数测试
 */
object AlgorithmTest {
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val sparkConf = new SparkConf().setMaster("spark://cloud1:7077").setAppName("AlgorithmTest")
      .set("spark.akka.frameSize", "2000")
      .set("spark.network.timeout", "1200")
    val sparkContext = new SparkContext(sparkConf)

    val loadDataTime = System.nanoTime()
    //读取评分数据
    val ratingsData = sparkContext.textFile("hdfs://ns1/data/movie/ratings.dat").map { line =>
      val fields = line.split("::")
      new Rating(fields(0).toInt, fields(1).toInt, fields(2).toInt)
    }.cache()
    ratingsData.count()
    println("[DEBUG]load data consuming:" + (System.nanoTime() - loadDataTime) / 1000000000 + "s")
    val users = ratingsData.map(_.user).distinct().collect()
    val products = ratingsData.map(_.product).distinct().collect()
    println("[DEBUG]get " + ratingsData.count() + " ratings from " + users.length + " users on " + products.length + " products.")
    //从中随机划分出训练和测试数据
    val splits = ratingsData.randomSplit(Array(0.8, 0.2), seed = 111l)
    val trainingData = splits(0).cache()
    val testData = splits(1).cache()
    println("[DEBUG]get " + trainingData.count() + " training data from ratings.dat")
    println("[DEBUG]get " + testData.count() + " test data from ratings.dat")


    //参数组合
    val rank = List(5, 8, 10)
    val lambda = List(0.1, 0.01)
    val numIter = 7

    var bestRank = 0
    var bestLambda = 0.0
    var bestRmse = 1.0
    //循环训练模型，对比准确度，选出最佳的参数组合
    for (r <- rank; l <- lambda) {
      val traningTime = System.nanoTime()
      val model = ALS.train(trainingData, r, numIter, l)
      println("[DEBUG]traning consuming:" + (System.nanoTime() - traningTime) / 1000000000 + "s")
      val mse_rmse = mseAndRmse(model, testData)
      println("[DEBUG]MSE in this model is :" + mse_rmse._1 + " RMSE is : " + mse_rmse._2)
      println(s"[DEBUG]rank = $r , lambda = $l , iter = $numIter")
      if (mse_rmse._2 < bestRmse) {
        bestRmse = mse_rmse._2
        bestRank = r
        bestLambda = l
      }
      println()
    }
    println(s"the bestRank = $bestRank,bestLambda = $bestLambda on model,RMSE is $bestRmse")

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

  //计算召回率和查全率
  def recallAndPrecision(allRec: Array[(Int, Array[Rating])], trainingData: Array[Rating], usersLen: Int): (Double, Double) = {
    var totalRecall = 0
    var totalPrecision = 0
    println("======allRec======" + allRec.length)
    for (r <- allRec) {
      val intersect = r._2.intersect(trainingData).length
      totalRecall = totalRecall + (intersect / trainingData.length)
      totalPrecision = totalPrecision + (intersect / r._2.length)
    }
    (totalRecall / usersLen, totalPrecision / usersLen)
  }

  //计算覆盖率
  def coverage(allRec: Array[(Int, Array[Rating])], productsLen: Int): Double = {
    var totalProducts = new ArrayBuffer[Int]()
    for (r <- allRec) {
      totalProducts ++= r._2.map(r => r.product)
    }
    totalProducts.distinct.length / productsLen
  }

  //计算多样性
  /*  def diversity(users: Array[Int], topK: Int, model: MatrixFactorizationModel): Double = {
      var totalDiversity = 0.0
      for (u <- users) {
        val products = model.recommendProducts(u, topK).map(_.product)
        val features = model.productFeatures.filter(x => products.contains(x._1)).map(_._2).collect()
        var totalSimilary = 0.0
        for (f <- 0 until features.length - 1) {
          val similary = cosineSimilarity(new DoubleMatrix(features(f)), new DoubleMatrix(features(f + 1)))
          totalSimilary = totalSimilary + similary
        }
        val diversity = 1 - (totalSimilary / (0.5 * topK * (topK - 1)))
        totalDiversity = totalDiversity + diversity
      }
      totalDiversity / users.length
    }

    def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double = {
      vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
    }*/

  //计算新颖度
  def popularity(allRec: Array[(Int, Array[Rating])], trainingData: Array[Rating], usersLen: Int): Double = {
    var total_popularity = 0
    for (r <- allRec) {
      val products = r._2.map(r => r.product)
      var s_popularity = 0
      for (p <- products) {
        val times = trainingData.find(x => x.product == p).size
        s_popularity = s_popularity + (times / usersLen)
      }
      val avg_popularity = s_popularity / products.length
      total_popularity = total_popularity + avg_popularity
    }
    total_popularity / usersLen
  }
}
