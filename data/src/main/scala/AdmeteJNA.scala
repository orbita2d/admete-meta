
// File: AdmeteChessEngine.scala

import com.sun.jna.{Library, Native, Pointer}
import java.nio.ByteBuffer
import scala.util.{Try, Success, Failure}

// JNA Interface - mirrors the C API
trait AdmeteChessEngineJNA extends Library {
  // Initialise the engine, will either return 0 on success, or throw
  def init(): Int
  // Maps to: const int encode_features(const char* fen, char* buffer, size_t buffer_size, int quiesce)
  def encode_features(fen: String, buffer: ByteBuffer, buffer_size: Int, white_to_move: ByteBuffer, quiesce: Int): Int
}

case class AdmeteFeatures(
  features: Array[Byte], // 64 bytes
  whiteToMove: Boolean, // 1 byte
)

// Scala wrapper class for easier usage
class AdmeteChessEngine(libraryPath: Option[String] = None) {
  
  // Load the native library
  private val nativeLib: AdmeteChessEngineJNA = libraryPath match {
    case Some(path) => Native.load(path, classOf[AdmeteChessEngineJNA])
    case None => Native.load("admete", classOf[AdmeteChessEngineJNA]) // assumes libadmete.so/admete.dll in library path
  }

  private val initResult: Int = {
    val result = nativeLib.init()
    if (result != 0) {
      throw new RuntimeException(s"Failed to initialize Admete Chess Engine. Error code: $result")
    }
    result
  }
  
  def encodeFeatures(fen: String, quiesce: Boolean = false): Try[AdmeteFeatures] = {
    val bufferSize = 64;
    val buffer = ByteBuffer.allocateDirect(bufferSize)
    val whiteToMoveBuf = ByteBuffer.allocateDirect(1) // Pointer to a single byte for moveAfter
    val quiesceInt = if (quiesce) 1 else 0
    val result = nativeLib.encode_features(fen, buffer, bufferSize, whiteToMoveBuf, quiesceInt)
    result match {
      case 0 => // Success
        val bytes = new Array[Byte](bufferSize)
        buffer.get(bytes)
        val whiteToMove = whiteToMoveBuf.get(0) == 1 // Convert single byte to boolean
        Success(AdmeteFeatures(bytes, whiteToMove))
      case 1 => Failure(new RuntimeException("Buffer too small"))
      case 2 => Failure(new RuntimeException("Null pointer"))
      case _ => Failure(new RuntimeException(s"Unknown error code: $result"))
    }
  }
}

// Companion object for easy instantiation
object AdmeteChessEngine {
  def apply(): AdmeteChessEngine = new AdmeteChessEngine()
  def apply(libraryPath: String): AdmeteChessEngine = new AdmeteChessEngine(Some(libraryPath))
}
