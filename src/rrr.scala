import breeze.linalg.{Axis, sum, DenseMatrix => BDM, DenseVector => BDV}

object rrr extends App {
  val k = 10
  val vocabSize = 102660
  val LambdaQ = new BDM[Double](k, vocabSize) // k * v

  SingleProcess()

  def SingleProcess() {
    val startT = System.currentTimeMillis()
    sum(LambdaQ, Axis._1) // 1 * k <- k * v
    println((System.currentTimeMillis() -startT).toString)
    val startT2 = System.currentTimeMillis()
    sum(LambdaQ) // 1 * k <- k * v
    println((System.currentTimeMillis() -startT2).toString)
    val startT3 = System.currentTimeMillis()
    var dv = new BDV[Double](k)
    for (i <- 0 until k){
      dv.update(i, sum(LambdaQ(i, ::)))
    }
    println((System.currentTimeMillis() -startT3).toString)
  }

}
