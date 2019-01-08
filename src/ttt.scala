import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global
import java.util.concurrent.Executors
import scala.concurrent.{ExecutionContext, Await, Future}
import scala.concurrent.duration._

object ttt extends App{

  var nonEmptyDocCount: Long = 0L
  val ids = new Array[Int](6)
  ids.update(0, 0)
  ids.update(1, 1)
  ids.update(2, 2)
  ids.update(3, 3)
  ids.update(4, 4)
  ids.update(5, 5)

  val parameter = ids.clone()
  ids.foreach { case i =>
    print(i)
    print("\t")
  }
  println()

  parameter.foreach { case i =>
    print(i)
    print("\t")
  }
  println()

  val tasks: Seq[Future[Int]] = for (i <- ids) yield Future {
    nonEmptyDocCount+=1
    println("Executing task " + i)
    val x = parameter.apply(0)
    Thread.sleep(x * 1000L)
    var s = i.toString
    parameter.foreach { case a =>
      s += "\t"
      s += a.toString
    }
    println(s)
    println("Finish", i)
    parameter.apply(0)
  }

  val aggregated: Future[Seq[Int]] = Future.sequence(tasks)

  val squares: Seq[Int] = Await.result(aggregated, 20.seconds)
  println("Squares: " + squares)
}
