package org.apache.spark.mllib.feature

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom

import scala.collection.mutable


/**
  * Entry in vocabulary
  * add `weight` atrribute to weigh the importance of this `word`
  *
  * @author lulin
  *         $date 2017/11/14 9:07
  */
private case class MyVocabWord(var word: String, //分词
                               var wcn: Float, //weightedCount
                               var point: Array[Int], //存储路径,即经过的结点
                               var code: Array[Int], //记录Huffman编码
                               var codeLen: Int //存储到达该叶子结点,要经过的结点
                              )


/**
  * 一个经过权重调整后的word2vec实现，主要改动是：
  * 对每个paragraph赋予不同的权重，
  * 故统计词频时，每个词不再是初始为1的词频，而是1*word_weight的频率
  */
class WeightedWord2Vec extends Serializable with Logging {

  // members
  private var vectorSize = 100
  private var learningRate = 0.025
  private var numPartitions = 1
  private var numIterations = 1
  private var seed = Utils.random.nextLong()
  private var minCount = 5 //过滤单词阈值
  private var maxSentenceLength = 1000

  /**
    * Sets the maximum length (in words) of each sentence in the input data.
    * Any sentence longer than this threshold will be divided into chunks of
    * up to `maxSentenceLength` size (default: 1000)
    */
  @Since("2.0.0")
  def setMaxSentenceLength(maxSentenceLength: Int): this.type = {
    require(maxSentenceLength > 0,
      s"Maximum length of sentences must be positive but got ${maxSentenceLength}")
    this.maxSentenceLength = maxSentenceLength
    this
  }

  /**
    * Sets vector size (default: 100).
    */
  @Since("1.1.0")
  def setVectorSize(vectorSize: Int): this.type = {
    require(vectorSize > 0,
      s"vector size must be positive but got ${vectorSize}")
    this.vectorSize = vectorSize
    this
  }

  /**
    * Sets initial learning rate (default: 0.025).
    */
  @Since("1.1.0")
  def setLearningRate(learningRate: Double): this.type = {
    require(learningRate > 0,
      s"Initial learning rate must be positive but got ${learningRate}")
    this.learningRate = learningRate
    this
  }

  /**
    * Sets number of partitions (default: 1). Use a small number for accuracy.
    */
  @Since("1.1.0")
  def setNumPartitions(numPartitions: Int): this.type = {
    require(numPartitions > 0,
      s"Number of partitions must be positive but got ${numPartitions}")
    this.numPartitions = numPartitions
    this
  }

  /**
    * Sets number of iterations (default: 1), which should be smaller than or equal to number of
    * partitions.
    */
  @Since("1.1.0")
  def setNumIterations(numIterations: Int): this.type = {
    require(numIterations >= 0,
      s"Number of iterations must be nonnegative but got ${numIterations}")
    this.numIterations = numIterations
    this
  }

  /**
    * Sets random seed (default: a random long integer).
    */
  @Since("1.1.0")
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  /**
    * Sets the window of words (default: 5)
    */
  @Since("1.6.0")
  def setWindowSize(window: Int): this.type = {
    require(window > 0,
      s"Window of words must be positive but got ${window}")
    this.window = window
    this
  }

  /**
    * Sets minCount, the minimum number of times a token must appear to be included in the word2vec
    * model's vocabulary (default: 5).
    */
  @Since("1.3.0")
  def setMinCount(minCount: Int): this.type = {
    require(minCount >= 0,
      s"Minimum number of times must be nonnegative but got ${minCount}")
    this.minCount = minCount
    this
  }

  private val EXP_TABLE_SIZE = 1000
  private val MAX_EXP = 6
  private val MAX_CODE_LENGTH = 40

  /** context words from [-window, window] */
  private var window = 5 //context 范围限定

  private var trainWordsCount = 0.0
  private var vocabSize = 0 //词表内单词总数
  @transient private var vocab: Array[MyVocabWord] = null //词表
  @transient private var vocabHash = mutable.HashMap.empty[String, Int] //词表反查索引


  private def learnVocab[S <: Iterable[String]](dataset: RDD[(S, Float)]): Unit = {
    vocab = dataset.flatMap { case (vocab, wParagraph) =>
      vocab.map(w => (w, 1 + wParagraph))
    }
      .reduceByKey(_ + _)
      .filter(_._2 >= minCount)
      .map(x => MyVocabWord(
        x._1,
        x._2,
        new Array[Int](MAX_CODE_LENGTH),
        new Array[Int](MAX_CODE_LENGTH),
        0))
      .collect()
      .sortWith((a, b) => a.wcn > b.wcn)

    vocabSize = vocab.length
    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check " +
      "the setting of minCount, which could be large enough to remove all your words in sentences.")

    var a = 0
    while (a < vocabSize) {
      vocabHash += vocab(a).word -> a
      trainWordsCount += vocab(a).wcn
      a += 1
    }
    logInfo(s"vocabSize = $vocabSize, trainWordsCount = $trainWordsCount")
  }

  private def createExpTable(): Array[Float] = { //指数运算查表
    val expTable = new Array[Float](EXP_TABLE_SIZE)
    var i = 0
    while (i < EXP_TABLE_SIZE) {
      val tmp = math.exp((2.0 * i / EXP_TABLE_SIZE - 1.0) * MAX_EXP)
      expTable(i) = (tmp / (tmp + 1.0)).toFloat
      i += 1
    }
    expTable
  }

  private def createBinaryTree(): Unit = {
    val count = new Array[Double](vocabSize * 2 + 1) //二叉树中所有的结点
    val binary = new Array[Int](vocabSize * 2 + 1) //设置每个结点的Huffman编码：左1，右0
    val parentNode = new Array[Int](vocabSize * 2 + 1) //存储每个结点的父结点
    val code = new Array[Int](MAX_CODE_LENGTH) //存储每个叶子结点的huffman编码
    val point = new Array[Int](MAX_CODE_LENGTH) //存储每个叶子结点的路径
    var a = 0
    while (a < vocabSize) {
      count(a) = vocab(a).wcn //初始化叶子结点，结点的权值即为词频
      a += 1 //叶子结点编号为0~vocabSize-1与vocabHash中对分词的编号是一致的
    }
    while (a < 2 * vocabSize) {
      count(a) = 1e9.toInt //初始化非叶子结点，结点权值为无穷大；非叶子结点编号为vocabSize ~ 2* vocabSize-1
      a += 1
    }
    var pos1 = vocabSize - 1
    var pos2 = vocabSize

    var min1i = 0
    var min2i = 0

    a = 0
    while (a < vocabSize - 1) { //构建Huffman树
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min1i = pos1
          pos1 -= 1
        } else {
          min1i = pos2
          pos2 += 1
        }
      } else {
        min1i = pos2
        pos2 += 1
      }
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min2i = pos1
          pos1 -= 1
        } else {
          min2i = pos2
          pos2 += 1
        }
      } else {
        min2i = pos2
        pos2 += 1
      }
      count(vocabSize + a) = count(min1i) + count(min2i)
      parentNode(min1i) = vocabSize + a
      parentNode(min2i) = vocabSize + a
      binary(min2i) = 1
      a += 1
    }
    // Now assign binary code to each vocabulary word
    var i = 0
    a = 0
    while (a < vocabSize) {
      var b = a
      i = 0
      while (b != vocabSize * 2 - 2) {
        code(i) = binary(b)
        point(i) = b
        i += 1
        b = parentNode(b)
      }
      vocab(a).codeLen = i
      vocab(a).point(0) = vocabSize - 2
      b = 0
      while (b < i) {
        vocab(a).code(i - b - 1) = code(b)
        vocab(a).point(i - b) = point(b) - vocabSize
        b += 1
      }
      a += 1
    }
  }


  /**
    * Computes the vector representation of each word in vocabulary.
    *
    * @param dataset an RDD of sentences,
    *                each sentence is expressed as an iterable collection of words
    * @return a Word2VecModel
    */
  @Since("1.1.0")
  def fit[S <: Iterable[String]](dataset: RDD[(S, Float)]): Word2VecModel = {

    learnVocab(dataset)

    createBinaryTree()

    val sc = dataset.context

    val expTable = sc.broadcast(createExpTable())
    val bcVocab = sc.broadcast(vocab)
    val bcVocabHash = sc.broadcast(vocabHash)
    // each partition is a collection of sentences,
    // will be translated into arrays of Index integer
    val sentences: RDD[Array[Int]] = dataset.mapPartitions { sentenceIter =>
      // Each sentence will map to 0 or more Array[Int]
      sentenceIter.flatMap { case (sentence, _) =>
        // Sentence of words, some of which map to a word index
        val wordIndexes = sentence.flatMap(bcVocabHash.value.get)
        // break wordIndexes into trunks of maxSentenceLength when has more
        wordIndexes.grouped(maxSentenceLength).map(_.toArray)
      }
    }

    val newSentences = sentences.repartition(numPartitions).cache()
    val initRandom = new XORShiftRandom(seed)

    if (vocabSize.toLong * vectorSize >= Int.MaxValue) {
      throw new RuntimeException("Please increase minCount or decrease vectorSize in Word2Vec" +
        " to avoid an OOM. You are highly recommended to make your vocabSize*vectorSize, " +
        "which is " + vocabSize + "*" + vectorSize + " for now, less than `Int.MaxValue`.")
    }

    val syn0Global =
      Array.fill[Float](vocabSize * vectorSize)((initRandom.nextFloat() - 0.5f) / vectorSize)
    val syn1Global = new Array[Float](vocabSize * vectorSize)
    var alpha = learningRate

    for (k <- 1 to numIterations) {
      val bcSyn0Global = sc.broadcast(syn0Global)
      val bcSyn1Global = sc.broadcast(syn1Global)
      val partial = newSentences.mapPartitionsWithIndex { case (idx, iter) =>
        val random = new XORShiftRandom(seed ^ ((idx + 1) << 16) ^ ((-k - 1) << 8))
        val syn0Modify = new Array[Int](vocabSize)
        val syn1Modify = new Array[Int](vocabSize)
        val model = iter.foldLeft((bcSyn0Global.value, bcSyn1Global.value, 0L, 0L)) {
          case ((syn0, syn1, lastWordCount, wordCount), sentence) =>
            var lwc = lastWordCount
            var wc = wordCount
            if (wordCount - lastWordCount > 10000) {
              lwc = wordCount
              // TODO: discount by iteration?
              alpha =
                learningRate * (1 - numPartitions * wordCount.toDouble / (trainWordsCount + 1))
              if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
              logInfo("wordCount = " + wordCount + ", alpha = " + alpha)
            }
            wc += sentence.length
            var pos = 0
            while (pos < sentence.length) {
              val word = sentence(pos)
              val b = random.nextInt(window)
              // Train Skip-gram
              var a = b
              while (a < window * 2 + 1 - b) {
                if (a != window) {
                  val c = pos - window + a
                  if (c >= 0 && c < sentence.length) {
                    val lastWord = sentence(c)
                    val l1 = lastWord * vectorSize
                    val neu1e = new Array[Float](vectorSize)
                    // Hierarchical softmax
                    var d = 0
                    while (d < bcVocab.value(word).codeLen) {
                      val inner = bcVocab.value(word).point(d)
                      val l2 = inner * vectorSize
                      // Propagate hidden -> output
                      var f = blas.sdot(vectorSize, syn0, l1, 1, syn1, l2, 1)
                      if (f > -MAX_EXP && f < MAX_EXP) {
                        val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                        f = expTable.value(ind)
                        val g = ((1 - bcVocab.value(word).code(d) - f) * alpha).toFloat
                        blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1)
                        blas.saxpy(vectorSize, g, syn0, l1, 1, syn1, l2, 1)
                        syn1Modify(inner) += 1
                      }
                      d += 1
                    }
                    blas.saxpy(vectorSize, 1.0f, neu1e, 0, 1, syn0, l1, 1)
                    syn0Modify(lastWord) += 1
                  }
                }
                a += 1
              }
              pos += 1
            }
            (syn0, syn1, lwc, wc)
        }
        val syn0Local = model._1
        val syn1Local = model._2
        // Only output modified vectors.
        Iterator.tabulate(vocabSize) { index =>
          if (syn0Modify(index) > 0) {
            Some((index, syn0Local.slice(index * vectorSize, (index + 1) * vectorSize)))
          } else {
            None
          }
        }.flatten ++ Iterator.tabulate(vocabSize) { index =>
          if (syn1Modify(index) > 0) {
            Some((index + vocabSize, syn1Local.slice(index * vectorSize, (index + 1) * vectorSize)))
          } else {
            None
          }
        }.flatten
      }
      val synAgg = partial.reduceByKey { case (v1, v2) =>
        blas.saxpy(vectorSize, 1.0f, v2, 1, v1, 1)
        v1
      }.collect()
      var i = 0
      while (i < synAgg.length) {
        val index = synAgg(i)._1
        if (index < vocabSize) {
          Array.copy(synAgg(i)._2, 0, syn0Global, index * vectorSize, vectorSize)
        } else {
          Array.copy(synAgg(i)._2, 0, syn1Global, (index - vocabSize) * vectorSize, vectorSize)
        }
        i += 1
      }
      bcSyn0Global.destroy(false)
      bcSyn1Global.destroy(false)
    }
    newSentences.unpersist()
    expTable.destroy(false)
    bcVocab.destroy(false)
    bcVocabHash.destroy(false)

    val wordArray = vocab.map(_.word)
    new Word2VecModel(wordArray.zipWithIndex.toMap, syn0Global)
  }

}