
import java.util.Random

import org.apache.log4j.{Logger, Level}
import org.apache.spark.{SparkContext, SparkConf}



case class Movie(id: String, name: String)
case class RatingTmp(user: String, movie: String, rating: String, time: String)

object TrainDataGenerator {
  def main(args: Array[String]) {
    if (args.length < 1) {
      System.err.println("Usage:<type>")
      System.exit(1)
    }

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    //val sparkConf = new SparkConf().setMaster("spark://proj10:7077").setAppName("TrainDataGenerator")
    val sparkConf = new SparkConf().setAppName("TrainDataGenerator")

    val sc = new SparkContext(sparkConf)

    
    if (args(0) == "t_movies") {
      val movies = sc.textFile("/recsys/movies.csv").map { lines =>
        val fields = lines.split(',')
        new Movie(fields(0), fields(1))
      }.cache()
      movies.foreach { book =>
        HBaseHelper.put("t_books", book.id, "msg", "name", book.name)
        HBaseHelper.put("t_books", book.id, "msg", "price", book.price)
      }
      System.exit(1)
    }
    
    if (args(0) == "t_users") {
      for (i <- 1 to 138493) {
        HBaseHelper.put("t_users", i.toString, "msg", "uid", i)
      }
      System.exit(1)
    }
  
    if (args(0) == "t_ratings") {
        val ratings = sc.textFile("/recsys/test.csv").map { lines =>
        val fields = lines.split(',')
        new RatingTmp(fields(0), fields(1), fields(2), fields(3))
      }.cache()
      
      ratings.foreach{ r =>
        HBaseHelper.put("t_ratings", r.time, "msg", "userId", r.user)
        HBaseHelper.put("t_ratings", r.time, "msg", "movieId", r.movie)
        HBaseHelper.put("t_ratings", r.time, "msg", "rating", r.rating)
      }
      System.exit(1)
    }
    
    if (args(0) == "t_ratings1") {
        val ratings = sc.textFile("/recsys/test.csv").map { lines =>
        val fields = lines.split(',')
        
        //HBaseHelper.put("t_ratings", fields(3), "msg", "userId", fields(0))
       // HBaseHelper.put("t_ratings", fields(3), "msg", "movieId", fields(1))
        //HBaseHelper.put("t_ratings", fields(3), "msg", "rating", fields(2))
      }.collect()
      System.exit(1)
    }

    sc.stop()
  }
}
