package nvidia.inferenceserver.client.utils

import com.google.protobuf.ByteString
import koma.ndarray.NDArray
import nvidia.inferenceserver.ModelConfigOuterClass
import nvidia.inferenceserver.ModelConfigOuterClass.DataType.*
import java.nio.ByteBuffer
import kotlin.reflect.KClass


@Suppress("UNCHECKED_CAST")
fun convertProtoMatTypeToKotlin(protoType: ModelConfigOuterClass.DataType?): KClass<out Number> {
    val dtype = when (protoType) {
        TYPE_UINT8 -> Byte::class // treat as byte
        TYPE_INVALID -> TODO()
        TYPE_BOOL -> TODO()
        TYPE_UINT16 -> TODO()
        TYPE_UINT32 -> TODO()
        TYPE_UINT64 -> TODO()
        TYPE_INT8 -> Byte::class
        TYPE_INT16 -> Short::class
        TYPE_INT32 -> Int::class
        TYPE_INT64 -> Long::class
        TYPE_FP16 -> TODO()
        TYPE_FP32 -> Float::class
        TYPE_FP64 -> Double::class
        //TYPE_STRING -> TODO()
        UNRECOGNIZED -> TODO()
        null -> TODO()
        else -> TODO()
    }
    return dtype as KClass<out Number>
}

@Suppress("UNCHECKED_CAST")
fun <T: Number> Number.asType(type: KClass<T>): T {
    return when (type) {
        Byte::class -> toByte() as T
        Float::class -> toFloat() as T
        Double::class -> toDouble() as T
        Long::class -> toLong() as T
        Int::class -> toInt() as T
        Short::class -> toShort() as T
        else -> TODO()
    }

}

@Suppress("UNCHECKED_CAST")
fun <T: Number> Number.createConstNdArray(type: KClass<T>, vararg dims: Int): NDArray<T> {
    return when (type) {
        Byte::class -> NDArray(*dims){toByte()} as NDArray<T>
        Float::class -> NDArray(*dims){toFloat()} as NDArray<T>
        Double::class -> NDArray(*dims){toDouble()} as NDArray<T>
        Long::class -> NDArray(*dims){toLong()} as NDArray<T>
        Int::class -> NDArray(*dims){toInt()} as NDArray<T>
        Short::class -> NDArray(*dims){toShort()} as NDArray<T>
        else -> TODO()
    }

}

fun Number.createConstNdArrayAndConvertToByteString(type: KClass<out Number>, vararg dims: Int): ByteString? {
    return when (type) {
        Byte::class -> (NDArray(*dims){toByte()}).toByteString()
        Float::class -> (NDArray(*dims){toFloat()}).toByteString()
        Double::class -> (NDArray(*dims){toDouble()}).toByteString()
        Long::class -> (NDArray(*dims){toLong()}).toByteString()
        Int::class -> (NDArray(*dims){toInt()}).toByteString()
        Short::class -> (NDArray(*dims){toShort()}).toByteString()
        else -> TODO()
    }

}

@Suppress("UNCHECKED_CAST")
inline fun <reified T: Number> NDArray<T>.toBytes(): ByteBuffer {
    val ndAsList = toList()
    val byteBuffer: ByteBuffer
    byteBuffer =  when (T::class) {
        Byte::class -> {
            val b = ByteBuffer.allocateDirect(ndAsList.size)
            b.put((ndAsList as List<Byte>).toByteArray())
            b
        }
        Float::class -> {
            val b = ByteBuffer.allocateDirect(ndAsList.size * 4)
            b.asFloatBuffer().put((ndAsList as List<Float>).toFloatArray())
            b
        }
        Double::class -> {
            val b = ByteBuffer.allocateDirect(ndAsList.size * 8)
            b.asDoubleBuffer().put((ndAsList as List<Double>).toDoubleArray())
            b
        }
        Int::class -> {
            val b = ByteBuffer.allocateDirect(ndAsList.size * 4)
            b.asIntBuffer().put((ndAsList as List<Int>).toIntArray())
            b
        }
        else -> TODO()
    }
    return byteBuffer
}


inline fun <reified T: Number> NDArray<T>.toByteString(): ByteString? {
    return ByteString.copyFrom(toBytes())
}

@Suppress("UNCHECKED_CAST")
fun <T: Number> ByteBuffer.toNdArray(type: KClass<T>, shape: List<Long>): NDArray<T>{
    val result = when (type) {
        Byte::class -> {
            val arr = ByteArray(remaining())
            get(arr)
            ndArrayOf(shape.map { it.toInt() }, arr.asList())
        }
        Float::class -> {
            val floatBuffer = asFloatBuffer()
            val arr = FloatArray(floatBuffer.remaining())
            floatBuffer.get(arr)
            ndArrayOf(shape.map { it.toInt() }, arr.asList())
        }
        Double::class -> {
            val doubleBuffer = asDoubleBuffer()
            val arr = DoubleArray(doubleBuffer.remaining())
            doubleBuffer.get(arr)
            ndArrayOf(shape.map { it.toInt() }, arr.asList())
        }
        Int::class -> {
            val intBuffer = asIntBuffer()
            val arr = IntArray(intBuffer.remaining())
            intBuffer.get(arr)
            ndArrayOf(shape.map { it.toInt() }, arr.asList())
        }
        else -> TODO()
    }

    return result as NDArray<T>

}

@Suppress("UNCHECKED_CAST")
fun <T: Number> ByteString.toNdArray(type: KClass<T>, shape: List<Long>): NDArray<T>{
    return this.asReadOnlyByteBuffer().toNdArray(type, shape)
}
