// src/main/scala/Main.scala
import com.github.luben.zstd.ZstdInputStream
import java.io.{BufferedReader, InputStreamReader, FileInputStream}
import scala.util.Using
import os.Path
import com.github.bhlangonijr.chesslib.pgn.{PgnIterator}
import com.github.bhlangonijr.chesslib.util.{LargeFile}
import scala.jdk.CollectionConverters._
import com.github.bhlangonijr.chesslib.game.Game
import com.github.bhlangonijr.chesslib.Board
import scala.collection.mutable.ListBuffer
import scala.util.matching.Regex
import scala.util.boundary, boundary.break
import com.github.bhlangonijr.chesslib.Piece
import akka.actor.typed.scaladsl.AskPattern._
import akka.util.Timeout
import scala.concurrent.duration._
import akka.stream._
import akka.stream.scaladsl._
import scala.concurrent.{ExecutionContext, Future}
import scala.concurrent.duration._
import com.google.common.hash.{BloomFilter, Funnels}
import java.util.concurrent.atomic.AtomicLong
import akka.actor.typed.ActorRef
import akka.actor.typed.ActorSystem
import akka.actor.typed.Behavior
import akka.actor.typed.scaladsl.Behaviors
import java.nio.charset.StandardCharsets
import scala.util.Random
import scala.concurrent.Future
import scala.util.{Success, Failure, Try}

import com.github.mjakubowski84.parquet4s.{ParquetReader, ParquetStreams, ParquetWriter, Path => ParquetPath}
import org.apache.parquet.hadoop.ParquetFileWriter.Mode
import org.apache.parquet.hadoop.metadata.CompressionCodecName
import org.apache.parquet.hadoop.{ParquetWriter => HadoopParquetWriter}
import org.apache.hadoop.conf.Configuration
import com.github.mjakubowski84.parquet4s.Col


// Messages the actor understands
sealed trait BloomCommand
case class CheckAndAdd(fen: String, replyTo: ActorRef[Boolean]) extends BloomCommand


case class PositionData(fen: String, eval: Int)
// case class TrainingRecord(features: String, eval: Int, setType: Int)
case class TrainingRecord(fen: String, features: List[Byte], eval: Int, setType: String)

// The actor that owns the bloom filter
def bloomFilterActor(bloomFilter: BloomFilter[CharSequence], uniqueCount: AtomicLong): Behavior[BloomCommand] = {
  Behaviors.receive { (context, message) =>
    message match {
      case CheckAndAdd(fen, replyTo) =>
        val isNew = !bloomFilter.mightContain(fen)
        if (isNew) {
          bloomFilter.put(fen)
          uniqueCount.incrementAndGet()
        }
        replyTo ! isNew
        Behaviors.same
    }
  }
}

@main def processZst(positionLimit: Int, enginePath: String, filename: String, output: String): Unit = {
  implicit val system: ActorSystem[Nothing] = ActorSystem(Behaviors.empty, "chess-processor" )
  implicit val ec: ExecutionContext = system.executionContext
  implicit val scheduler = system.scheduler 
  implicit val timeout: Timeout = 5.seconds

  val path = os.Path(filename, os.pwd)
  val outputPath = os.Path(output, os.pwd)
  os.makeDir.all(outputPath)
  // Check that the output path is empty
  if (os.exists(outputPath) && os.list(outputPath).nonEmpty) {
    throw new IllegalArgumentException(s"Output path is not empty: $outputPath")
  }
  
  // Initialize bloom filter and counters
  val bloomFilter = BloomFilter.create(Funnels.stringFunnel(StandardCharsets.UTF_8), positionLimit, 0.1)
  val uniqueCount = new AtomicLong(0)
  val bloomActor = system.systemActorOf(bloomFilterActor(bloomFilter, uniqueCount), "bloom-filter")  
  val startTime = System.currentTimeMillis()

  // Create the engine pool
  // Check if the engine path is valid
  if (!os.exists(os.Path(enginePath, os.pwd))) {
    throw new IllegalArgumentException(s"Engine path does not exist: $enginePath")
  }

  val engine = AdmeteChessEngine(enginePath)

  val weights = Array(0.8, 0.1, 0.1) // Train, Test, Validation
  
  Using.resource(new ZstdInputStream(path.getInputStream)) { zstdStream =>
    val largeFile = new LargeFile(zstdStream)
    val pgnIterator = new PgnIterator(largeFile)
    
    val future = Source.fromIterator(() => pgnIterator.iterator.asScala)      
      // Parallel game processing 
      .mapAsync(parallelism = 8) { game =>
        Future {
          processGame(game)
        }
      }
      // Flatten to individual positions
      .mapConcat(identity)
      // Parallel bloom filter processing    
      .mapAsync(parallelism = 4) { positionData =>
        implicit val timeout: Timeout = 3.seconds
        
        bloomActor.ask[Boolean](CheckAndAdd(positionData.fen, _)).map { isNew =>
          if (isNew) Some(positionData) else None
        }
      }
      
      // Filter out duplicates
      .collect { case Some(pos) => pos }
      
      // Stop when we hit 100M unique positions
      .takeWhile(_ => uniqueCount.get() < positionLimit)
      // Convert to TrainingRecord by calling the engine
      .mapAsync(parallelism = 4) { positionData => {
        Future {
          val features = engine.encodeFeatures(positionData.fen, quiesce = true)
          features match {
            case Success(features) =>
              // Determine set type based on weights
              val setType = sampleCategorical(weights) match {
                case 0 => "train"
                case 1 => "test"
                case 2 => "validation"
              }
              val evalRelative = features.whiteToMove match {
                case true => positionData.eval
                case false => -positionData.eval
              }
              Some(TrainingRecord(positionData.fen, features.features.toList, evalRelative, setType))
            case Failure(ex) =>
              println(s"Failed to encode features for position ${positionData.fen}: ${ex.getMessage}")
              None
          }
        }
      }}
      // Filter out failed engine calls
      .collect { case Some(trainingRecord) => trainingRecord }
      .wireTap { record =>
        // Log progress every 1000 records
        val currentCount = uniqueCount.get()
        val currentPct = (currentCount * 100.0 / positionLimit).toInt
        if (currentCount % 10000 == 0) {
          val elapsedTime = (System.currentTimeMillis() - startTime) / 1000.0
          println(s"Processed $currentCount unique positions ($currentPct%), elapsed time: ${elapsedTime}s")
        }
      }
      .via(
        ParquetStreams
        .viaParquet
        .of[TrainingRecord]
        .partitionBy(Col("setType"))
        .write(ParquetPath(outputPath.toNIO))
      )
      .runWith(Sink.ignore)
        
    // Wait for completion and report final stats
    val result = scala.concurrent.Await.result(future, Duration.Inf)
    val totalTime = (System.currentTimeMillis() - startTime) / 1000.0
    
    println(s"Final stats:")
    println(s"Unique positions: ${uniqueCount.get()}")
    println(s"Total time: ${totalTime}s")
    println(s"Rate: ${uniqueCount.get() / totalTime} positions/sec")
    
    system.terminate()
  }
}


def sampleCategorical(weights: Array[Double]): Int = {
  val r = Random.nextDouble()
  val cumulative = weights.scanLeft(0.0)(_ + _).tail
  cumulative.indexWhere(_ >= r)
}


def processGame(game: Game, skipMoves: Int = 10, minPieces: Int = 7, minElo: Int =2000): List[PositionData] = {
  boundary {
    val board = new Board()
    val positions = ListBuffer[PositionData]()
    val evalPattern: Regex = raw"\[%eval ([+-]?\d+\.\d+)\]".r

    // Check if both players have ELO ratings above the threshold
    if (game.getWhitePlayer.getElo < minElo || game.getBlackPlayer.getElo < minElo) {
      return List.empty // Skip this game if ELO is too low
    }
    
    game.loadMoveText()
    val maybeComments = Option(game.getComments()).map(_.asScala.toMap)

    val comments = Option(game.getComments()).map(_.asScala.toMap) match {
      case None => return List.empty // No comments, nothing to process
      case Some(a) => a
    }
    
    // Early exit check
    val firstCheckMove = skipMoves + 1
    comments.get(firstCheckMove) match {
      case Some(comment) if evalPattern.findFirstIn(comment).isDefined =>
      case _ => break(List.empty)
    }
    
    for ((move, index) <- game.getHalfMoves().asScala.zipWithIndex) {
      board.doMove(move)
      val pieceCount = board.boardToArray.count(p => p != Piece.NONE)
      
      // Break equivalent
      if (pieceCount < minPieces) 
        break(positions.toList)
      
      // Skip first n moves (continue equivalent)
      if (index >= skipMoves) {
        val fen = board.getFen()
        val moveNumber = index + 1
        val comment = comments.get(moveNumber)
        
        val eval: Option[Int] = comment.flatMap { commentText =>
          evalPattern.findFirstMatchIn(commentText).map { m =>
            (m.group(1).toDouble * 100).toInt
          }
        }
        
        eval match {
          case None =>
          case Some(value) => positions += PositionData(fen, value)
        }
      }
    }
    
    positions.toList
  }
}