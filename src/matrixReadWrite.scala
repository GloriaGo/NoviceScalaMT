import java.util.Random

import breeze.linalg.{sum, DenseMatrix => BDM, DenseVector => BDV}
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


  val manyRounds = 5
  val step = 1
  val max = 10
  val min = 1

  def testCreateTime(): Unit = {
    val timer2 = new Array[Long](max / step + 2)
    println("Start Timer 1 ......")
    var x = 0.0
    for (a <- max to min by -step) {
      for (b <- 1 to manyRounds) {
        val startTime = System.nanoTime()
        val partT = BDM.zeros[Double](k, a * 100)
        x += partT.apply(1, 0)
        val endTime = System.nanoTime()
        timer2.update(a / step, timer2.apply(a / step) + endTime - startTime)
      }
    }
    println(x)
    for (a <- 1 until timer2.length) {
      println("Creating a matrix with size " + a * 100 + " * 102660 needs " + (1.0 * timer2.apply(a) / (1e6 * manyRounds)).toString + " (ms)")
    }
  }

  def testSelectTime(): Unit = {
    val timer = new Array[Long](max / step + 2)
    var x = 0.0
    println("Start Timer 2 ......")
    for (a <- max to min by -step) {
      val lambdaQ = createBigMatrix(k, vocabSize)
      for (b <- 1 to manyRounds) {
        val randomIds = createRandomList(a * 100, vocabSize - 1)
        val startTime = System.nanoTime()
        val partQ = lambdaQ(::, randomIds)
        x += partQ.apply(1, 4)
        val endTime = System.nanoTime()
        timer.update(a / step, timer.apply(a / step) + endTime - startTime)
      }
    }
    println(x)
    for (a <- 1 until timer.length) {
      println("Selecting from big matrix with size " + a * 100 + " * 102660 needs " + (1.0 * timer.apply(a) / (1e6 * manyRounds)).toString + " (ms)")
    }
  }

  testWriteTime()

  def testWriteTime(): Unit = {
    val timer = new Array[Long](2)
    var x = 0.0
    println("Start Timer ......")
    var startTime = 0L
    var endTime = 0L
    for (a <- 1 to manyRounds) {
      // Only to make sure JVM is big enough
      val lambda = getGammaMatrixH(k, vocabSize)
      val rowSum : BDV[Double] = sum(lambda(breeze.linalg.*, ::)) // 1 * k <- k * v
      val linkpart = {max to min by -step}.map { case b =>
        val randomIds = createRandomList(b * 100, vocabSize - 1)
        val part = lambda(::, randomIds).toDenseMatrix
        val partBeta: BDM[Double] = part * 0.1 + 2.0
        val partSum : BDV[Double] = sum(partBeta(breeze.linalg.*, ::))
        (part, partBeta, randomIds)
      }
      linkpart.foreach{ case (part, partBeta, ids) =>
        lambda(::, ids) := part + partBeta
      }
      // Start from here
      val lambdaQ = getGammaMatrixH(k, vocabSize)
      val rowSumQ : BDV[Double] = sum(lambdaQ(breeze.linalg.*, ::)) // 1 * k <- k * v
      println("lambdaQ old rowSum : " + rowSumQ.apply(1))

      for (b <- max to min by -step) {
        val randomIds = createRandomList(b * 100, vocabSize - 1)
        val partQ = lambdaQ(::, randomIds).toDenseMatrix
        val partBetaQ: BDM[Double] = partQ * 0.1 + 2.0
        val partSumQ : BDV[Double] = sum(partBetaQ(breeze.linalg.*, ::))
        // println(partSumQ.apply(1))
        startTime = System.nanoTime()
        lambdaQ(::, randomIds) := partQ + partBetaQ
        endTime = System.nanoTime()
        timer.update(0, timer.apply(0) + endTime - startTime)
      }
      println("lambdaQ new rowSum : " + sum(lambdaQ(breeze.linalg.*, ::)).apply(1))


      val lambdaP = getGammaMatrixH(k, vocabSize)
      val rowSumP : BDV[Double] = sum(lambdaP(breeze.linalg.*, ::)) // 1 * k <- k * v

      println("lambdaP old rowSum : " + rowSumP.apply(1))

      val linkpartP = {max to min by -step}.map { case b =>
        val randomIds = createRandomList(b * 100, vocabSize - 1)
        val partP = lambdaP(::, randomIds).toDenseMatrix
        val partBetaP: BDM[Double] = partP * 0.1 + 2.0
        val partSumP : BDV[Double] = sum(partBetaP(breeze.linalg.*, ::))
        (partP, partBetaP, randomIds)
      }
      println(linkpartP.apply(1)._3.length)

      startTime = System.nanoTime()
      linkpartP.foreach{ case (partP, partBetaP, ids) =>
        lambdaP(::, ids) := partP + partBetaP
      }
      endTime = System.nanoTime()
      timer.update(1, timer.apply(1) + endTime - startTime)
      println("lambdaP new rowSum : " + sum(lambdaP(breeze.linalg.*, ::)).apply(1))

    }
    for (a <- 0 until timer.length) {
      println(a.toString + " needs " + (1.0 * timer.apply(a) / (1e6 * manyRounds)).toString + " (ms)")
    }
  }

  def testVHTime(): Unit = {
    val timer = new Array[Long](2)
    var x = 0.0
    println("Start Timer ......")
    var startTime = 0L
    var endTime = 0L
    for (a <- 1 to manyRounds) {

      val lambdaP = getGammaMatrixH(k, vocabSize)
      startTime = System.nanoTime()
      val rowSumP : BDV[Double] = sum(lambdaP(breeze.linalg.*, ::)) // 1 * k <- k * v
      // println("rowSumP: " + rowSumP.length)
      for (b <- max to min by -step) {
        val randomIds = createRandomList(b * 100, vocabSize - 1)
        val partP = lambdaP(::, randomIds).toDenseMatrix
      }
      endTime = System.nanoTime()
      timer.update(1, timer.apply(1) + endTime - startTime)

      val lambdaQ = getGammaMatrixV(k, vocabSize)
      startTime = System.nanoTime()
      val rowSumQ : BDV[Double] = sum(lambdaQ(breeze.linalg.*, ::)) // 1 * k <- k * v
      // println("rowSumQ: " + rowSumQ.length)
      for (b <- max to min by -step) {
        val randomIds = createRandomList(b * 100, vocabSize - 1)
        val partQ = lambdaQ(::, randomIds).toDenseMatrix
      }
      endTime = System.nanoTime()
      timer.update(0, timer.apply(0) + endTime - startTime)

    }
    for (a <- 0 until timer.length) {
      println(a.toString + " needs " + (1.0 * timer.apply(a) / (1e6 * manyRounds)).toString + " (ms)")
    }
  }

  def createRandomList(listSize: Int, listRange: Int): List[Int] = {
    {
      1 to listSize
    }.map { _ => randomGenerator.nextInt(listRange) }.sorted.distinct.toList
  }

  def createBigMatrix(rows: Int, cols: Int): BDM[Double] = {
    val matrix = getGammaMatrixV(rows, cols)
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

  def getGammaMatrixV(row: Int, col: Int): BDM[Double] = {
    val randBasis = new RandBasis(new org.apache.commons.math3.random.MersenneTwister(
      randomGenerator.nextLong()))
    val gammaRandomGenerator = new Gamma(gammaShape, 1.0 / gammaShape)(randBasis)
    val temp = gammaRandomGenerator.sample(row * col).toArray
    new BDM[Double](col, row, temp).t
  }

  def getGammaMatrixH(row: Int, col: Int): BDM[Double] = {
    val randBasis = new RandBasis(new org.apache.commons.math3.random.MersenneTwister(
      randomGenerator.nextLong()))
    val gammaRandomGenerator = new Gamma(gammaShape, 1.0 / gammaShape)(randBasis)
    val temp = gammaRandomGenerator.sample(row * col).toArray
    new BDM[Double](row, col, temp)
  }
}
