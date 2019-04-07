package main

import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.client.{Scan, Put, HTable}
import org.apache.hadoop.hbase.util.Bytes
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
 * Created by JChubby_ on 2015/6/24.
 * 读取/写入HBase
 */
object HBaseHelper {
  val hbaseConf = HBaseConfiguration.create()
  //hbaseConf.set("hbase.zookeeper.quorum", "cloud1,cloud2,cloud3")
  //hbaseConf.set("hbase.zookeeper.property.clientPort", "2181")
  //hbaseConf.set("zookeeper.session.timeout", "6000000")
  val tables = new mutable.HashMap[String, HTable]()

  //获得当前表
  def getTable(tableName: String): HTable = {
    tables.getOrElse(tableName, {
      val tb = new HTable(hbaseConf, tableName)
      /*tb.setAutoFlush(false, false)
      tb.setWriteBufferSize(1024 * 1024)
      tb.flushCommits()*/
      tables(tableName) = tb
      tb
    })
  }

  //写入
  def put(tableName: String, rowKey: String, family: String, qualifier: String, value: String) {
    val table = getTable(tableName)
    val put = new Put(Bytes.toBytes(rowKey))
    /*qualifierValue.map(x => {
      if (!(x._2.isEmpty))
        put.add(Bytes.toBytes(family), Bytes.toBytes(x._1), Bytes.toBytes(x._2))
    })*/
    put.add(Bytes.toBytes(family), Bytes.toBytes(qualifier), Bytes.toBytes(value))
    table.put(put)
  }

  //获得所有行健
  def getAllRow(tableName: String): Array[String] = {
    val table = getTable(tableName)
    val resultScaner = table.getScanner(new Scan())
    val resIter = resultScaner.iterator()
    var resArr = new ArrayBuffer[String]()
    while (resIter.hasNext) {
      val res = resIter.next()
      if (res != null && !res.isEmpty) {
        resArr += Bytes.toString(res.getRow)
      }
    }
    resArr.toArray
  }
}
