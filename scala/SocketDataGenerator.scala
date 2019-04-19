import java.io.PrintWriter
import java.net.ServerSocket
import java.util.Random
import java.io.{BufferedReader, InputStreamReader}
import org.apache.hadoop.fs.FSDataInputStream

object SocketDataGenerator {
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage:<seed> <type>")
      System.exit(1)
    }
    println("=====================")
    val listener = new ServerSocket(9999)

    var inputStream: FSDataInputStream = null
    var bufferedReader: BufferedReader = null
    inputStream = HDFSUtil.getFSDataInputStream("/recsys/stream.csv")
    bufferedReader = new BufferedReader(new InputStreamReader(inputStream))
    var lineTxt: String = bufferedReader.readLine()

    while (true) {
      val socket = listener.accept()
      new Thread() {
        override def run = {
          println("Get client connected from:" + socket.getInetAddress)
          val out = new PrintWriter(socket.getOutputStream(), true)
      
          if (args(1) == "all") {
            while (lineTxt != null) {
              Thread.sleep(500)
              
              lineTxt = bufferedReader.readLine()
              val field = lineTxt.split(',')
              val content = field(0) + "\t" + field(1) + "\t" + field(2)
              println(content)
              out.write(content + '\n')
              out.flush()
            }
          }
	  /*
          else {
            while (true) {
              Thread.sleep(2000)
              val userId = 1
              val bookId = ranBook.nextInt(300) + 1
              val rating = ranRating.nextInt(2) + 3
              val content = userId + "\t" + bookId + "\t" + rating
              println(content)
              out.write(content + '\n')
              out.flush()
            }
          }
          */
          socket.close()
        }
      }.start()
    }
  }
}
