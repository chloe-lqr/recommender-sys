package main

import java.io.PrintWriter
import java.net.ServerSocket
import java.util.Random

/**
 * Created by JChubby_ on 2015/6/24.
 * 随机模拟用户实时产生的行为数据
 */
object SocketDataGenerator {
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage:<seed> <type>")
      System.exit(1)
    }
    println("=====================")
    val listener = new ServerSocket(9999)
    val seed = args(0).toInt
    val ranUser = new Random(seed)
    val ranBook = new Random(seed + 1)
    val ranRating = new Random(seed + 2)
    while (true) {
      val socket = listener.accept()
      new Thread() {
        override def run = {
          println("Get client connected from:" + socket.getInetAddress)
          val out = new PrintWriter(socket.getOutputStream(), true)
	  //随机产生所有
          if (args(1) == "all") {
            while (true) {
              Thread.sleep(500)
              val userId = ranUser.nextInt(4000) + 1
              val bookId = ranBook.nextInt(5060) + 1
              val rating = ranRating.nextInt(5) + 1
              val content = userId + "\t" + bookId + "\t" + rating
              println(content)
              out.write(content + '\n')
              out.flush()
            }
          }
	  //随机产生前指定用户的评分行为
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
          socket.close()
        }
      }.start()
    }
  }
}
