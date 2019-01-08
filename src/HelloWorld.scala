import java.util.Random

import breeze.linalg.{max, sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import breeze.stats.distributions.{Gamma, RandBasis}
import org.apache.commons.math3.util.FastMath

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.{Await, Future}
import scala.concurrent.duration._


object HelloWorld extends App {
  println("hello whold!")
  //hyper parameters
  val k = 100
  val vocabSize = 102660
  val eta = 1.0 / k
  val workersize = 8
  val maxRecursive = 9.0
  val seed = 0L
  val randomGenerator = new Random(seed)
  val alpha = BDV.fill[Double](k, 1.0 / k)
  val gammaShape = 100
  val lambdaInit = getGammaMatrix(k, vocabSize) //  k * v, 1.5 s
  val index = 1L

  val corpusSize = 8000
  val fileName = "datasets/nytimes_libsvm_8k.txt"
  // Step 1 : Original Process
  testSerialProcess(fileName)
  // Step 2 : Singel Thread Gouped
  testSingleThread(fileName, 1)
  testSingleThread(fileName, 4)
  // Step 3 : Multi Threads
  testMultiThread(fileName, 4)

  def testSerialProcess(fileName: String): Unit = {
    val startTime = System.nanoTime()
    val corpus = generateDocs(fileName)
    val finalResult = SerialProcess(index, corpus)
    println("finalResult: " + sum(finalResult(1, ::)))
    println("Serial Process Training Time: " + (1.0 * (System.nanoTime() - startTime) / 1e9).toString)
    val perplexity = Perplexity(finalResult.t)
    println("-----------perplexity: " + perplexity.toString)
  }

  def testSingleThread(fileName: String, threadNumber: Int): Unit = {
    val startTime = System.nanoTime()
    val corpus = generateDocs(fileName)
    val finalResult = SingleThread(index, corpus, threadNumber)
    // println("finalResult: " + sum(finalResult(1, ::)))
    println("SingleThread Training Time: " + (1.0 * (System.nanoTime() - startTime) / 1e9).toString)
    val perplexity = Perplexity(finalResult.t)
    println("----------perplexity: " + perplexity.toString)
  }

  def testMultiThread(fileName: String, threadNumber: Int): Unit = {
    val startTime = System.nanoTime()
    val corpus = generateDocs(fileName)
    val finalResult = MultiThread(index, corpus, threadNumber)
    // println("finalResult: " + sum(finalResult(1, ::)))
    println("Multi Thread Training Time: " + (1.0 * (System.nanoTime() - startTime) / 1e9).toString)
    val perplexity = Perplexity(finalResult.t)
    println("----------perplexity: " + perplexity.toString)
  }

  def SerialProcess(index: Long, docs: Iterator[(Long, List[(Int, Double)])]): BDM[Double] = {
    val iter = 1
    val weight = math.pow((4096 + iter) * workersize, -0.6)
    val a1 = 1.0 - weight
    val a2 = weight * corpusSize
    val a3 = weight * eta

    var a1factorial = 1.0
    var a1factsum = 0.0
    val LambdaQ = new BDM[Double](k, vocabSize, lambdaInit.toArray) // k * v
    var rowSumQ: BDV[Double] = sum(LambdaQ(breeze.linalg.*, ::)) // 1 * k <- k * v
    println("rowSumQ[1]:" + rowSumQ.apply(1))
    println(sum(LambdaQ(1, ::)))

    docs.foreach { case (docId: Long, termCounts: List[(Int, Double)]) =>
      // YY Sparse Words
      val (ids, cts) = initDoc(termCounts)

      val tmp1 = a3 * a1factsum
      // YY matrix read to get PartLambda  --- 2.5 ms
      // YY Todo: Multi Thread
      val PartQ = LambdaQ(::, ids).toDenseMatrix //  k * ids
    val PartLambda: BDM[Double] = PartQ * a1factorial + tmp1 // k * ids

      // YY get RowSum --- 0.0025 ms
      val tmp2 = tmp1 * vocabSize
      val rowSum = rowSumQ * a1factorial + tmp2 // k

      // YY Todo: Multi Thread
      val PartExpElogBetaD = exp(dirExpLowPrecision(PartLambda,
        rowSum, maxRecursive)).t.toDenseMatrix // ids * k

      // E-Step
      // YY Local VI with sparse expElogbeta, sstats(k * ids) --- 5.3 ms
      // YY Todo: Multi Thread
      val (gammad, sstats) = partLowPVI(PartExpElogBetaD, alpha, gammaShape, k,
        maxRecursive, seed + index, iter, ids, cts)

      // YY real delta -> lazy update delta  --- 0.37 ms
      val tmp3 = a2 / (a1factorial * a1)
      // YY Todo: Multi Thread
      val DeltaLambdaQ = (sstats * tmp3) *:* PartExpElogBetaD.t // k * ids
      // YY prepare for next lambdaQ --- 1.65 ms
      PartQ := PartQ + DeltaLambdaQ // k * ids
      LambdaQ(::, ids) := PartQ // Sparse Write

      // YY prepare for next sumQ --- 0.13 ms
      // YY Todo: Multi Thread
      val deltaRowSum = sum(DeltaLambdaQ(breeze.linalg.*, ::)) // k * ids
      rowSumQ := rowSumQ + deltaRowSum

      // YY prepare for next lazy update
      a1factsum = a1factsum + a1factorial
      a1factorial = a1factorial * a1
    }
    // YY reconstract real Matrix value --- 246 ms
    val tmp4 = a3 * a1factsum
    LambdaQ := LambdaQ * a1factorial + tmp4 // k * v
    LambdaQ
  }

  def SingleThread(index: Long, docs: Iterator[(Long, List[(Int, Double)])], threadNumber: Int): BDM[Double] = {
    val iter = 1
    val weight = math.pow((4096 + iter) * workersize, -0.6)
    val a1 = 1.0 - weight
    val a2 = weight * corpusSize
    val a3 = weight * eta

    var a1factorial = 1.0
    var a1factsum = 0.0
    val LambdaQ = new BDM[Double](k, vocabSize, lambdaInit.toArray) // k * v
    var rowSumQ: BDV[Double] = sum(LambdaQ(breeze.linalg.*, ::)) // 1 * k <- k * v
    // println("rowSumQ[1]:" + rowSumQ.apply(1))
    // println(sum(LambdaQ(1, ::)))

    val groupedIt = docs.grouped(threadNumber)
    while (groupedIt.hasNext) {
      val partSeq = groupedIt.next.map { case (docId: Long, termCounts: List[(Int, Double)]) =>
        val (ids, cts) = initDoc(termCounts)
        // Step 1 : read shared parameters
        val partQ = LambdaQ(::, ids).toDenseMatrix //  k * ids
        (ids, cts, partQ)
      }

      val rowSum = rowSumQ * a1factorial + a3 * a1factsum * vocabSize // k

      val deltaSeq = partSeq.map { case (ids: List[Int], cts: Array[Double], partQ: BDM[Double]) =>
        // Step 2 : calculate delta
        val (deltaRowSum, deltaLambdaQ) = calculateDelta(partQ, rowSum, a1factorial, a1factsum,
          ids, cts, iter, a1, a2, a3)
        (ids, deltaLambdaQ, deltaRowSum)
      }
      // Step 3 : write shared parameters
      deltaSeq.foreach { case (ids, deltaLambdaQ, deltaRowSum) =>
        LambdaQ(::, ids) := LambdaQ(::, ids) + deltaLambdaQ // Sparse Write
        rowSumQ := rowSumQ + deltaRowSum
        a1factsum = a1factsum + a1factorial
        a1factorial = a1factorial * a1
      }
    }
    // YY reconstract real Matrix value --- 246 ms
    val tmp4 = a3 * a1factsum
    LambdaQ := LambdaQ * a1factorial + tmp4 // k * v
    LambdaQ
  }

  def MultiThread(index: Long, docs: Iterator[(Long, List[(Int, Double)])], threadNumber: Int): BDM[Double] = {
    val iter = 1
    val weight = math.pow((4096 + iter) * workersize, -0.6)
    val a1 = 1.0 - weight
    val a2 = weight * corpusSize
    val a3 = weight * eta

    var a1factorial = 1.0
    var a1factsum = 0.0
    val LambdaQ = new BDM[Double](k, vocabSize, lambdaInit.toArray) // k * v
    var rowSumQ: BDV[Double] = sum(LambdaQ(breeze.linalg.*, ::)) // 1 * k <- k * v
    // println("rowSumQ[1]:" + rowSumQ.apply(1))
    // println(sum(LambdaQ(1, ::)))

    val groupedIt = docs.grouped(threadNumber)
    while (groupedIt.hasNext) {
      val partSeq = groupedIt.next.map { case (docId: Long, termCounts: List[(Int, Double)]) =>
        val (ids, cts) = initDoc(termCounts)
        // Step 1 : read shared parameters
        val partQ = LambdaQ(::, ids).toDenseMatrix //  k * ids
        (ids, cts, partQ)
      }

      val rowSum = rowSumQ * a1factorial + a3 * a1factsum * vocabSize // k

      // customize the execution context to use the specified number of threads
      val tasks: List[Future[(List[Int], BDM[Double], BDV[Double])]] =
        for ((ids, cts, partQ) <- partSeq) yield Future {
          // Step 2 : calculate delta
          val (deltaRowSum, deltaLambdaQ) = calculateDelta(partQ, rowSum, a1factorial, a1factsum,
            ids, cts, iter, a1, a2, a3)
          (ids, deltaLambdaQ, deltaRowSum)
        }
      val aggregated: Future[Seq[(List[Int], BDM[Double], BDV[Double])]] = Future.sequence(tasks)
      val deltaSeq: Seq[(List[Int], BDM[Double], BDV[Double])] = Await.result(aggregated, 1.seconds)

      // Step 3 : write shared parameters
      deltaSeq.foreach { case (ids, deltaLambdaQ, deltaRowSum) =>
        LambdaQ(::, ids) := LambdaQ(::, ids) + deltaLambdaQ // Sparse Write
        rowSumQ := rowSumQ + deltaRowSum
        a1factsum = a1factsum + a1factorial
        a1factorial = a1factorial * a1
      }
    }
    // YY reconstract real Matrix value --- 246 ms
    val tmp4 = a3 * a1factsum
    LambdaQ := LambdaQ * a1factorial + tmp4 // k * v
    LambdaQ
  }

  //  def testMatrixRead(): Unit = {
  //    var LambdaQ: BDM[Double] = lambdaInit // k * v
  //    val tmp_docs: List[Int] = {
  //      0 to 1000
  //    }.toList
  //    val group1 = tmp_docs.grouped(1000).toList
  //    val group2 = tmp_docs.grouped(1).toList
  //
  //
  //    val startTime2 = System.nanoTime()
  //    val result2 = group2.map { case ids: List[Int] =>
  //      // Step 1 : read shared parameters
  //      val PartQ = LambdaQ(::, ids).toDenseMatrix
  //      1
  //    }
  //    println(result2.size)
  //    println("Testing2 Time: " + (1.0 * (System.nanoTime() - startTime2) / 1e6).toString)
  //
  //
  //    val startTime1 = System.nanoTime()
  //    val result1 = group1.map { case ids: List[Int] =>
  //      // Step 1 : read shared parameters
  //      val PartQ = LambdaQ(::, ids).toDenseMatrix
  //      1
  //    }
  //    println(result1.size)
  //    println("Testing1 Time: " + (1.0 * (System.nanoTime() - startTime1) / 1e6).toString)
  //
  //
  //    val startTime3 = System.nanoTime()
  //    val result3 = group2.map { case ids: List[Int] =>
  //      // Step 1 : read shared parameters
  //      val PartQ = LambdaQ(::, ids).toDenseMatrix
  //      1
  //    }
  //    println(result3.size)
  //    println("Testing2 Time: " + (1.0 * (System.nanoTime() - startTime3) / 1e6).toString)
  //
  //  }
  //  def FeiFeiMultiThread(index: Long, docs: Iterator[(Long, List[(Int, Double)])], numThreads: Int): BDM[Double] = {
  //    // customize the execution context to use the specified number of threads
  //    implicit val ec = ExecutionContext.fromExecutor(Executors.newFixedThreadPool(numThreads))
  //
  //    val tasks: Iterator[Future[Long]] = for ((docId: Long, termCounts: List[(Int, Double)]) <- docs) yield Future{
  //      val len = termCounts.length
  //      // YY Sparse Words
  //      val idsA = new Array[Int](len)
  //      val cts = new Array[Double](len)
  //      var i: Int = 0
  //      while(i < len){
  //        idsA.update(i, termCounts.apply(i)._1)
  //        cts.update(i, termCounts.apply(i)._2)
  //        i += 1
  //      }
  //      val ids = idsA.toList
  //      val tmp1 = a3 * a1factsum
  //
  //      // YY matrix read to get PartLambda  --- 2.5 ms
  //      // YY Todo: Multi Thread
  //      val PartQ = LambdaQ(::, ids).toDenseMatrix //  k * ids
  //      val PartLambda: BDM[Double] = PartQ * a1factorial + tmp1 // k * ids
  //
  //      // YY get RowSum --- 0.0025 ms
  //      val tmp2 = tmp1 * vocabSize
  //      val rowSum = rowSumQ * a1factorial + tmp2 // k
  //
  //      // YY Todo: Multi Thread
  //      val PartExpElogBetaD = exp(dirExpLowPrecision(PartLambda,
  //        rowSum, maxRecursive)).t.toDenseMatrix // ids * k
  //
  //      // E-Step
  //      // YY Local VI with sparse expElogbeta, sstats(k * ids) --- 5.3 ms
  //      // YY Todo: Multi Thread
  //      val (gammad, sstats) = partLowPVI(PartExpElogBetaD, alpha, gammaShape, k,
  //        maxRecursive, seed + index, iter, ids, cts)
  //
  //      // YY real delta -> lazy update delta  --- 0.37 ms
  //      val tmp3 = a2 / (a1factorial * a1)
  //      // YY Todo: Multi Thread
  //      val DeltaLambdaQ = (sstats * tmp3) *:* PartExpElogBetaD.t // k * ids
  //      // YY prepare for next lambdaQ --- 1.65 ms
  //      PartQ := PartQ + DeltaLambdaQ // k * ids
  //      LambdaQ(::, ids) := PartQ // Sparse Write
  //
  //      // YY prepare for next sumQ --- 0.13 ms
  //      // YY Todo: Multi Thread
  //      val deltaRowSum = sum(DeltaLambdaQ(breeze.linalg.*, ::)) // k * ids
  //      rowSumQ := rowSumQ + deltaRowSum
  //
  //      // YY prepare for next lazy update
  //      a1factsum = a1factsum + a1factorial
  //      a1factorial = a1factorial * a1
  //      docId
  //    }
  //
  //    val aggregated: Future[Iterator[Long]] = Future.sequence(tasks)
  //
  //    val squares: Iterator[Long] = Await.result(aggregated, 20.seconds)
  //    println("Squares: " + squares)
  //
  //    // YY reconstract real Matrix value --- 246 ms
  //    val tmp4 = a3 * a1factsum
  //    LambdaQ := LambdaQ * a1factorial + tmp4 // k * v
  //    LambdaQ
  //  }

  def generateDocs(fileName: String): Iterator[(Long, List[(Int, Double)])] = {
    // val fileName = "datasets/nytimes_libsvm_800.txt"

    val lines = scala.io.Source.fromFile(fileName).getLines
    val documents = lines.map { line =>
      val parts = line.split(" ")
      val docId = parts.apply(0).toLong
      val termCounts = parts.filter(part => part.contains(":")).map { part =>
        val pair = part.split(":")
        val id = Integer.parseInt(pair.apply(0))
        val ct = Integer.parseInt(pair.apply(1)).toDouble
        (id, ct)
      }
      (docId, termCounts.toList)
    }
    documents
  }

  def initDoc(termCounts: List[(Int, Double)]): (List[Int], Array[Double]) = {
    val len = termCounts.length
    val idsA = new Array[Int](len)
    val cts = new Array[Double](len)
    var i: Int = 0
    while (i < len) {
      idsA.update(i, termCounts.apply(i)._1)
      cts.update(i, termCounts.apply(i)._2)
      i += 1
    }
    val ids = idsA.toList
    (ids, cts)
  }

  def calculateDelta(PartQ: BDM[Double], rowSum: BDV[Double], a1factorial: Double, a1factsum: Double,
                     ids: List[Int], cts: Array[Double], iter: Int,
                     a1: Double, a2: Double, a3: Double): (BDV[Double], BDM[Double]) = {
    val PartLambda: BDM[Double] = PartQ * a1factorial + a3 * a1factsum // k * ids
    val PartExpElogBetaD = exp(dirExpLowPrecision(PartLambda,
      rowSum, maxRecursive)).t.toDenseMatrix // ids * k
    val (gammad, sstats) = partLowPVI(PartExpElogBetaD, alpha, gammaShape, k,
      maxRecursive, seed + index, iter, ids, cts)
    val tmp3 = a2 / (a1factorial * a1)
    val deltaLambdaQ = (sstats * tmp3) *:* PartExpElogBetaD.t // k * ids
    val deltaRowSum = sum(deltaLambdaQ(breeze.linalg.*, ::)) // k * ids
    (deltaRowSum, deltaLambdaQ)
  }

  def Perplexity(lambdaD: BDM[Double]): Double = {
    val fileName1 = "datasets/nytimes_libsvm_800.txt"
    val docs = generateDocs(fileName1)
    val ElogbetaD = dirichletExpectation(lambdaD.t).t
    val expElogbetaD = exp(ElogbetaD)
    var tokenCount: Double = 0.0

    val corpusPart = docs.map { case (id: Long, termCounts: List[(Int, Double)]) =>
      val len = termCounts.length
      val idsA = new Array[Int](len)
      val cts = new Array[Double](len)
      var i: Int = 0
      while (i < len) {
        idsA.update(i, termCounts.apply(i)._1)
        cts.update(i, termCounts.apply(i)._2)
        tokenCount += cts.apply(i)
        i += 1
      }
      val ids = idsA.toList
      var docBound = 0.0D
      val (gammad: BDV[Double], _, _) = lowPrecisionVI(
        ids, cts, expElogbetaD, alpha, gammaShape, k, maxRecursive, seed + id)
      val Elogthetad: BDV[Double] = dirExpLowPrecision(gammad, maxRecursive)
      // E[log p(doc | theta, beta)]
      termCounts.foreach { case (idx, count) =>
        val tmp = Elogthetad + ElogbetaD(idx, ::).t
        val tmp2 = logSumExp(tmp)
        docBound += count * tmp2
      }
      // E[log p(theta | alpha) - log q(theta | gamma)]
      docBound += sum((alpha - gammad) *:* Elogthetad)
      docBound += sum(lgamma(gammad) - lgamma(alpha))
      docBound += lgamma(sum(alpha)) - lgamma(sum(gammad))
      docBound
    }.sum
    // println("corpusPart: " + corpusPart)

    val sumEta = eta * vocabSize
    val topicsPart = sum((eta - lambdaD) *:* ElogbetaD) +
      sum(lgamma(lambdaD) - lgamma(eta)) +
      sum(lgamma(sumEta) - lgamma(sum(lambdaD(::, breeze.linalg.*))))
    // println("topicsPart: " + topicsPart)
    // println("tokenCount: " + tokenCount)
    val bound = topicsPart + corpusPart
    exp(-bound / tokenCount)
  }

  def getGammaMatrix(row: Int, col: Int): BDM[Double] = {
    val randBasis = new RandBasis(new org.apache.commons.math3.random.MersenneTwister(
      randomGenerator.nextLong()))
    val gammaRandomGenerator = new Gamma(gammaShape, 1.0 / gammaShape)(randBasis)
    val temp = gammaRandomGenerator.sample(row * col).toArray
    new BDM[Double](col, row, temp).t
  }

  def dirichletExpectation(alpha: BDM[Double]): BDM[Double] = {
    val rowSum = sum(alpha(breeze.linalg.*, ::))
    partDirExp(alpha, rowSum)
  }

  def partDirExp(newAlpha: BDM[Double], rowSum: BDV[Double]): BDM[Double] = {
    val digAlpha = digamma(newAlpha)
    val digRowSum = digamma(rowSum)
    val result = digAlpha(::, breeze.linalg.*) - digRowSum
    result // k * ids
  }

  def logSumExp(x: BDV[Double]): Double = {
    val a = max(x)
    a + log(sum(exp(x -:- a)))
  }

  def partLowPVI(expElogbetad: BDM[Double], alpha: breeze.linalg.Vector[Double], gammaShape: Double, k: Int,
                 maxRecursive: Double, seed: Long, iter: Int, ids: List[Int], cts: Array[Double]): (BDV[Double], BDM[Double]) = {
    // Initialize the variational distribution q(theta|gamma) for the mini-batch
    val randBasis = new RandBasis(new org.apache.commons.math3.random.MersenneTwister(seed))
    val gammad: BDV[Double] =
      new Gamma(gammaShape, 1.0 / gammaShape)(randBasis).samplesVector(k) // K
    val expElogthetad: BDV[Double] = exp(dirExpLowPrecision(gammad, maxRecursive)) // K
    // YY ignored the original version
    // val expElogbetad = expElogbeta(ids, ::).toDenseMatrix                        // ids * K

    val phiNorm: BDV[Double] = expElogbetad * expElogthetad +:+ 1e-100 // ids
    var meanGammaChange = 1D
    val ctsVector = new BDV[Double](cts) // ids

    // Iterate between gamma and phi until convergence
    while (meanGammaChange > 1e-2) {
      val lastgamma = gammad.copy
      //        K                  K * ids               ids
      gammad := (expElogthetad *:* (expElogbetad.t * (ctsVector /:/ phiNorm))) +:+ alpha
      expElogthetad := exp(dirExpLowPrecision(gammad, maxRecursive))
      // TODO: Keep more values in log space, and only exponentiate when needed.
      phiNorm := expElogbetad * expElogthetad +:+ 1e-100
      meanGammaChange = sum(abs(gammad - lastgamma)) / k
    }

    val sstatsd = expElogthetad.asDenseMatrix.t * (ctsVector /:/ phiNorm).asDenseMatrix
    (gammad, sstatsd)
  }

  def lowPrecisionVI(ids: List[Int], cts: Array[Double], expElogbeta: BDM[Double], alpha: breeze.linalg.Vector[Double],
                     gammaShape: Double, k: Int, maxRecursive: Double, seed: Long): (BDV[Double], BDM[Double], List[Int]) = {
    // Spark 2.4.0 Version
    // Initialize the variational distribution q(theta|gamma) for the mini-batch
    val randBasis = new RandBasis(new org.apache.commons.math3.random.MersenneTwister(seed))
    val gammad: BDV[Double] =
      new Gamma(gammaShape, 1.0 / gammaShape)(randBasis).samplesVector(k) // K
    // val expElogthetad: BDV[Double] = exp(LDAUtils.dirichletExpectation(gammad))  // K
    val expElogthetad: BDV[Double] = exp(dirExpLowPrecision(gammad, maxRecursive))
    val expElogbetad = expElogbeta(ids, ::).toDenseMatrix // ids * K

    val phiNorm: BDV[Double] = expElogbetad * expElogthetad +:+ 1e-100 // ids
    var meanGammaChange = 1D
    val ctsVector = new BDV[Double](cts) // ids

    // Iterate between gamma and phi until convergence
    while (meanGammaChange > 1e-2) {
      val lastgamma = gammad.copy
      //        K                  K * ids               ids
      gammad := (expElogthetad *:* (expElogbetad.t * (ctsVector /:/ phiNorm))) +:+ alpha
      // expElogthetad := exp(LDAUtils.dirichletExpectation(gammad))
      expElogthetad := exp(dirExpLowPrecision(gammad, maxRecursive))
      // TODO: Keep more values in log space, and only exponentiate when needed.
      phiNorm := expElogbetad * expElogthetad +:+ 1e-100
      meanGammaChange = sum(abs(gammad - lastgamma)) / k
    }
    val sstatsd = expElogthetad.asDenseMatrix.t * (ctsVector /:/ phiNorm).asDenseMatrix
    (gammad, sstatsd, ids)
  }

  def digammaLowPrecision(x: Double, maxRecursive: Double): Double = {
    var v = 0.0
    if (x > 0 && x <= 1e-5) { // use method 5 from Bernardo AS103
      // accurate to O(x)
      v = -0.577215664901532 - 1 / x
    }
    else if (x >= maxRecursive) { // use method 4 (accurate to O(1/x^8)
      //            1       1        1         1
      // log(x) -  --- - ------ + ------- - -------
      //           2 x   12 x^2   120 x^4   252 x^6
      val inv = 1 / (x * x)
      v = FastMath.log(x) - 0.5 / x - inv * ((1.0 / 12) + inv * (1.0 / 120 - inv / 252))
    }
    else {
      val firstPart = maxRecursive + x - x.toInt
      val inv = 1 / (firstPart * firstPart)
      v = FastMath.log(firstPart) - 0.5 / firstPart -
        inv * ((1.0 / 12) + inv * (1.0 / 120 - inv / 252))
      var i = x
      while (i < maxRecursive) {
        v -= 1 / i
        i = i + 1
      }
    }
    return v
  }

  def dirExpLowPrecision(alpha: BDV[Double], maxRecursive: Double): BDV[Double] = {
    val digammaSum = digammaLowPrecision(sum(alpha), maxRecursive)
    val digAlpha = alpha.map(x => digammaLowPrecision(x, maxRecursive))
    digAlpha - digammaSum
  }

  def dirExpLowPrecision(alpha: BDM[Double], rowSum: BDV[Double], maxRecursive: Double): BDM[Double] = {
    val digAlpha = alpha.map(x => digammaLowPrecision(x, maxRecursive))
    val digRowSum = rowSum.map(x => digammaLowPrecision(x, maxRecursive))
    val result = digAlpha(::, breeze.linalg.*) - digRowSum
    result // k * v
  }
}
