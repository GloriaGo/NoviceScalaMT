import java.util.Random

import breeze.linalg.{sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.{abs, exp}
import breeze.stats.distributions.{Gamma, RandBasis}
import org.apache.commons.math3.util.FastMath

object HelloWorld extends App {
  println("hello whold!")
  //hyper parameters
  val k = 200
  val vocabSize = 102660
  val corpusSize = 269565
  val workersize = 8
  val iter = 1
  val weight = 0.001
  val maxRecursive = 2.0
  val a1 = 1.0 - weight
  val a2 = weight * corpusSize
  val a3 = weight * 0.005
  val seed = 0L
  val randomGenerator = new Random(seed)
  val alpha = BDV.fill[Double](k, 1.0 / k)
  val gammaShape = 100

  val lambdaBc = getGammaMatrix(k, vocabSize) //  k * v, 1.5 s
  val index = 1L
  // TODO: read file from documents.txt
  val docs = generateDocs()
  val LambdaQ = lambdaBc // k * v
  var nonEmptyDocCount: Long = 0L
  var rowSumQ: BDV[Double] = sum(LambdaQ(breeze.linalg.*, ::)) // 1 * k <- k * v
  var a1factorial = 1.0
  var a1factsum = 0.0

  val finalResult = SingleThread(index, docs)

  def generateDocs(): Iterator[List[(Int, Double)]] = {
    val documents = 
    // val fileName = "datasets/nytimes_libsvm_block0.txt"
    val fileName = "datasets/documents.txt"
    for (line <- scala.io.Source.fromFile(fileName).getLines) {
      val parts = line.split(" ")

    }
    documents
  }

  // running time in this function means the running time for each document
  def SingleThread(index: Long, docs: Iterator[List[(Int, Double)]]): BDM[Double] = {
    docs.foreach { case termCounts: List[(Int, Double)] =>
      nonEmptyDocCount += 1
      // YY Sparse Words
      val cts = new Array[Double](0)
      val idsA = new Array[Int](0)
      termCounts.map { case (a, b) =>
        idsA :+ a
        cts :+ b
      }
      val ids = idsA.toList
      //        val (ids: List[Int], cts: Array[Double]) = termCounts match {
      //          case v: BDV => ((0 until v.size).toList, v.values)
      //          case v: SparseVector => (v.indices.toList, v.values)
      //        }

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

  def getGammaMatrix(row: Int, col: Int): BDM[Double] = {
    val randBasis = new RandBasis(new org.apache.commons.math3.random.MersenneTwister(
      randomGenerator.nextLong()))
    val gammaRandomGenerator = new Gamma(gammaShape, 1.0 / gammaShape)(randBasis)
    val temp = gammaRandomGenerator.sample(row * col).toArray
    new BDM[Double](col, row, temp).t
  }

  def partLowPVI(expElogbetad: BDM[Double],
                 alpha: breeze.linalg.Vector[Double],
                 gammaShape: Double,
                 k: Int,
                 maxRecursive: Double,
                 seed: Long,
                 iter: Int,
                 ids: List[Int],
                 cts: Array[Double]): (BDV[Double], BDM[Double]) = {
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


