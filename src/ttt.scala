object ttt extends App{
  import scala.concurrent.duration._
  import scala.concurrent.ExecutionContext.Implicits.global
  import java.util.concurrent.Executors
  import scala.concurrent.{ExecutionContext, Await, Future}
  import scala.concurrent.duration._

  var nonEmptyDocCount: Long = 0L
  val tasks: Seq[Future[Int]] = for (i <- 1 to 10) yield Future {
    nonEmptyDocCount+=1
    println("Executing task " + i)
    Thread.sleep(i * 1000L)
    println("Finish", i)
    1
  }

  val aggregated: Future[Seq[Int]] = Future.sequence(tasks)

  val squares: Seq[Int] = Await.result(aggregated, 20.seconds)
  println("Squares: " + squares)
}
