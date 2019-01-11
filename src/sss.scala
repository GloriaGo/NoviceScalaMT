import java.util.Random

import breeze.linalg.{Axis, max, sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import breeze.stats.distributions.{Gamma, RandBasis}
import org.apache.commons.math3.util.FastMath

object sss extends App {
  val gammaShape = 100
  val seed = 0L
  val randomGenerator = new Random(seed)
  val k = 100
  val vocabSize = 102660
   val lambdaInit = getGammaMatrix(k, vocabSize) //  k * v, 1.5 s
  val LambdaQ = new BDM[Double](k, vocabSize, lambdaInit.toArray) // k * v
  println(LambdaQ.apply(0, 0))

  SingleProcess()

  def SingleProcess() {
    val startT = System.currentTimeMillis()
    sum(LambdaQ, Axis._0) // 1 * k <- k * v
    println((System.currentTimeMillis() -startT).toString)
  }

  def getGammaMatrix(row: Int, col: Int): BDM[Double] = {
    val randBasis = new RandBasis(new org.apache.commons.math3.random.MersenneTwister(
      randomGenerator.nextLong()))
    val gammaRandomGenerator = new Gamma(gammaShape, 1.0 / gammaShape)(randBasis)
    val temp = gammaRandomGenerator.sample(row * col).toArray
    new BDM[Double](col, row, temp).t
  }

}
