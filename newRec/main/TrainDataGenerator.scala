package main

import java.util.Random

import org.apache.log4j.{Logger, Level}
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by JChubby_ on 2015/6/24.
 * 模拟用户行为，向HBase中写数据
 */

case class Book(id: String, name: String, price: String, info: String, image: String)

object TrainDataGenerator {
  def main(args: Array[String]) {
    if (args.length < 4) {
      System.err.println("Usage:<type> <startIndex> <seed> <sleepTime:ms>")
      System.exit(1)
    }

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val sparkConf = new SparkConf().setMaster("spark://cloud1:7077").setAppName("TrainDataGenerator")
    val sc = new SparkContext(sparkConf)
    //初始化图书表
    if (args(0) == "t_books") {
      val books = sc.textFile("hdfs://cloud1:9000/data/books.txt").map { lines =>
        val fields = lines.split("\t")
        new Book(fields(0), fields(1), fields(2), fields(3), fields(4))
      }.cache()
      books.foreach { book =>
        HBaseHelper.put("t_books", book.id, "msg", "name", book.name)
        HBaseHelper.put("t_books", book.id, "msg", "price", book.price)
        HBaseHelper.put("t_books", book.id, "msg", "info", book.info)
        HBaseHelper.put("t_books", book.id, "msg", "image", book.image)
      }
      System.exit(1)
    }
    //初始化用户表
    if (args(0) == "t_users") {
      for (i <- 1 to 4000) {
        val uid = "jchubby" + i.toString
        val pwd = "jchubby"
        HBaseHelper.put("t_users", i.toString, "msg", "uid", uid)
        HBaseHelper.put("t_users", i.toString, "msg", "pwd", pwd)
      }
      System.exit(1)
    }
    val seed = args(2).toInt
    val ranUser = new Random(seed)
    val ranBook = new Random(seed + 1)
    val ranRating = new Random(seed + 2)
    //初始化评分表，每个用户至少给10本图书评分过，没本图书至少被10个用户评分过（）
    if (args(0) == "t_ratings") {
      var rowId = 0
      for (j <- 1 to 10; i <- 1 to 5060) {
        rowId += 1
        val userId = ranUser.nextInt(3999) + 1
        val rating = ranRating.nextInt(4) + 1
        HBaseHelper.put("t_ratings", rowId.toString, "msg", "userId", userId.toString)
        HBaseHelper.put("t_ratings", rowId.toString, "msg", "bookId", i.toString)
        HBaseHelper.put("t_ratings", rowId.toString, "msg", "rating", rating.toString)
      }
      System.exit(1)
    }
    //随机模拟用户评分行为
    if (args(0) == "gogogo") {
      var i = args(1).toInt
      println("====================success -- " + i + "===============================")
      while (true) {
        val userId = ranUser.nextInt(3999) + 1
        val bookId = ranBook.nextInt(5059) + 1
        val rating = ranRating.nextInt(4) + 1
        i += 1
        HBaseHelper.put("t_ratings", i.toString, "msg", "userId", userId.toString)
        HBaseHelper.put("t_ratings", i.toString, "msg", "bookId", bookId.toString)
        HBaseHelper.put("t_ratings", i.toString, "msg", "rating", rating.toString)
        Thread.sleep(args(3).toInt)
      }
      System.exit(1)
    }
    sc.stop()
  }
}
