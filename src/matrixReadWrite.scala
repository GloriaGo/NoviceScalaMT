import java.util.Random
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.stats.distributions.{Gamma, RandBasis}


object MatrixReadWrite extends App {
  println("hello world!")
  //hyper parameters
  val k = 100
  val vocabSize = 102660
  val corpusSize = 800
  val eta = 1.0 / k
  val workersize = 1
  val maxRecursive = 9.0
  val seed = 0L
  val randomGenerator = new Random(seed)
  val alpha = BDV.fill[Double](k, 1.0 / k)
  val gammaShape = 100
  val index = 1L


  val multipleRound = 10
  val step = 1
  val max = 30
  val min = 1

  val timer2 = new Array[Long](max / step + 2)
  println("Start Timer 1 ......")
  var x = 0.0
  for (a <- max to min by -step) {
    for (b <- 1 to multipleRound) {
      val startTime = System.nanoTime()
      val partT = BDM.zeros[Double](k, a * 100)
      x += partT.apply(1, 0)
      val endTime = System.nanoTime()
      timer2.update(a / step, timer2.apply(a / step) + endTime - startTime)
    }
  }
  println(x)
  for (a <- 1 until timer2.length) {
    println("Creating a matrix with size " + a * 100 + " * 102660 needs " + (1.0 * timer2.apply(a) / (1e6 * multipleRound)).toString + " (ms)")
  }

  val timer = new Array[Long](max / step + 2)
  println("Start Timer 2 ......")
  for (a <- max to min by -step) {
    val lambdaQ = createBigMatrix(k, vocabSize)
    for (b <- 1 to multipleRound) {
      val randomIds = createRandomList(a * 100, vocabSize - 1)
      val startTime = System.nanoTime()
      val partQ: BDM[Double] = lambdaQ(::, randomIds).toDenseMatrix
      x += partQ.apply(1, 4)
      val endTime = System.nanoTime()
      timer.update(a / step, timer.apply(a / step) + endTime - startTime)
    }
  }
  println(x)
  for (a <- 1 until timer.length) {
    println("Selecting from big matrix with size " + a * 100 + " * 102660 needs " + (1.0 * timer.apply(a) / (1e6 * multipleRound)).toString + " (ms)")
  }

  def createRandomList(listSize: Int, listRange: Int): List[Int] = {
    {
      1 to listSize
    }.map { _ => randomGenerator.nextInt(listRange) }.sorted.distinct.toList
  }

  def createBigMatrix(rows: Int, cols: Int): BDM[Double] = {
    val matrix = getGammaMatrix(rows, cols)
    matrix
  }

  //  val i_2 = {1 to 100}.map{ i=>
  //    testMatrixRead(i)
  //    i*i
  //  }

  //  def testMatrixRead(i:Int): Unit ={
  //    val tmp_docs: List[Int] = {i*10000 to (i+1)*10000}.toList
  //    val group1 = tmp_docs.grouped(10000).toList
  //    val group2 = tmp_docs.grouped(1).toList
  //    showTime(group1)
  //    showTime(group2)
  //  }
  //
  //  def showTime(group: List[List[Int]]){
  //    val startTime = System.nanoTime()
  //    val result = group.map {ids: List[Int] =>
  //      // Step 1 : read shared parameters
  //      val PartQ = lambdaInit(::, ids).toDenseMatrix
  //      1
  //    }
  //    val a = result.size
  //    println("Testing Time: " + (1.0 * (System.nanoTime() - startTime) / 1e6).toString)
  //  }

  def getGammaMatrix(row: Int, col: Int): BDM[Double] = {
    val randBasis = new RandBasis(new org.apache.commons.math3.random.MersenneTwister(
      randomGenerator.nextLong()))
    val gammaRandomGenerator = new Gamma(gammaShape, 1.0 / gammaShape)(randBasis)
    val temp = gammaRandomGenerator.sample(row * col).toArray
    new BDM[Double](col, row, temp).t
  }
}
