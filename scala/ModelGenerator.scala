
import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.client.Result
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.mapreduce.TableInputFormat
import org.apache.hadoop.hbase.util.Bytes
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}


object ModelGenerator {
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    println("=====================step 1 initial conf==========================")
  
    val sparkConf = new SparkConf()
      .setAppName("ModelGenerator")
      .set("spark.akka.frameSize", "2000")
      .set("spark.network.timeout", "1200")
    val sparkContext = new SparkContext(sparkConf)
    val hbaseConf = HBaseConfiguration.create()
    //hbaseConf.set("hbase.zookeeper.quorum", "cloud4,cloud5,cloud6")
    //hbaseConf.set("hbase.zookeeper.property.clientPort", "2181")
    //hbaseConf.set("zookeeper.session.timeout", "6000000")

    println("\n=====================step 2 load data==========================")
  
    hbaseConf.set(TableInputFormat.INPUT_TABLE, "t_ratings")
    val ratingsData = sparkContext.newAPIHadoopRDD(hbaseConf, classOf[TableInputFormat],
      classOf[ImmutableBytesWritable],
      classOf[Result])
    val hbaseRatings = ratingsData.map { case (_, res) =>
      val userId = Bytes.toString(res.getValue(Bytes.toBytes("msg"), Bytes.toBytes("userId")))
      val movieId = Bytes.toString(res.getValue(Bytes.toBytes("msg"), Bytes.toBytes("movieId")))
      val rating = Bytes.toString(res.getValue(Bytes.toBytes("msg"), Bytes.toBytes("rating")))
      new Rating(userId.toInt, movieId.toInt, rating.toDouble)
    }.cache()
    val numTrainRatings = hbaseRatings.count()
    println(s"[DEBUG]get $numTrainRatings train data from hbase")

    val rank = 5
    val lambda = 0.1
    val numIter = 7

    if (args.length != 0) {
      println("\n=====================system initiallizing...==========================")
      println("\n[DEBUG]training model...")
      val firstTrainTime = System.nanoTime()
      val model = ALS.train(hbaseRatings, rank, numIter, lambda)
      val firstTrainEndTime = System.nanoTime() - firstTrainTime
      println("[DEBUG]first training consuming:" + firstTrainEndTime / 1000000000 + "s")

      println("\n[DEBUG]save recommended data to hbase...")
      val firstPutTime = System.nanoTime()
    
      for (i <- 1 to 4000) {
        val topRatings = model.recommendProducts(i, 10)
        var recMoives = ""
        for (r <- topRatings) {
          val rating = r.rating.toString.substring(0, 4)
          recMovies += r.product + ":" + rating + ","
        }
        HBaseHelper.put("t_users", i.toString, "msg", "recMovies", recMovies.substring(0, recMovies.length - 1))
      }
      val firstPutEndTime = System.nanoTime() - firstPutTime
      println("[DEBUG]finish job consuming:" + firstPutEndTime / 1000000000 + "s")
      System.exit(1)
    }

    println("\n=====================start real-time recommendation engine...==========================")
    val streamingTime = 120
    println(s"[DEBUG]The time interval to refresh model is: $streamingTime s")
  
    val streamingContext = new StreamingContext(sparkContext, Seconds(streamingTime))
    val socketRatings = streamingContext.socketTextStream("cloud1", 9999, StorageLevel.MEMORY_ONLY)
      .flatMap(_.split(" ")).map { lines =>
      val fields = lines.split("\t")
      new Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }
    var allData = hbaseRatings
    allData.cache.count()
    hbaseRatings.unpersist()
    var index = 0
    socketRatings.foreachRDD { rdd =>
      index += 1
      println("\n[DEBUG]this round (" + index + ") received: " + rdd.count + " data lines.")
      val refreshStartTime = System.nanoTime()
      val tmpData = allData.union(rdd).cache
      tmpData.count()
      allData = tmpData
      tmpData.unpersist()
      //allData = allData.union(rdd).repartition(10).cache()
      val model = ALS.train(allData, rank, numIter, lambda)
      val refreshEndTime = System.nanoTime() - refreshStartTime
      println("[DEBUG]training consuming:" + refreshEndTime / 1000000000 + " s")
      println("[DEBUG]begin refresh hbase user's recBooks...")
      val refreshAgainStartTime = System.nanoTime()
      val usersId = rdd.map(_.user).distinct().collect()
      for (u <- usersId) {
        val topRatings = model.recommendProducts(u, 10)
        var recMovies = ""
        for (r <- topRatings) {
          val rating = r.rating.toString.substring(0, 4)
          recMovies += r.product + ":" + rating + ","
        }
        HBaseHelper.put("t_users", u.toString, "msg", "recMovies", recMovies.substring(0, recMovies.length - 1))
      }
      val refreshAgainConsumingTime = System.nanoTime() - refreshAgainStartTime
      println("[DEBUG]finish refresh job,consuming:" + refreshAgainConsumingTime / 1000000000 + " s")
    }
    streamingContext.start()
    streamingContext.awaitTermination()

    sparkContext.stop()
  }
}
