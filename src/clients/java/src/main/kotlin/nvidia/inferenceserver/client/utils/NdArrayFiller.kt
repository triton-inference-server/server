package nvidia.inferenceserver.client.utils

import koma.extensions.fillLinear
import koma.ndarray.NDArray

@Suppress("UNCHECKED_CAST")
inline fun <reified T : Number> createLinearNdArray(vararg dims: Int,
                                           crossinline filler: (Int) -> T) =
    when(T::class) {
        Double::class -> NDArray.doubleFactory.zeros(*dims).fillLinear { filler(it) as Double }
        Float::class  -> NDArray.floatFactory.zeros(*dims).fillLinear { filler(it) as Float }
        Long::class   -> NDArray.longFactory.zeros(*dims).fillLinear { filler(it) as Long }
        Int::class    -> NDArray.intFactory.zeros(*dims).fillLinear { filler(it) as Int }
        Short::class  -> NDArray.shortFactory.zeros(*dims).fillLinear { filler(it) as Short }
        Byte::class   -> NDArray.byteFactory.zeros(*dims).fillLinear { filler(it) as Byte }
        else          -> NDArray.createGenericNulls<T>(*dims).fillLinear { filler(it) }
    } as NDArray<T>

/**
 * Create a new NDArray from a series of elements.  By default this creates a 1D array.  To create a multidimensional
 * array, list the elements in flattened order and include the shape argument.
 */
inline fun <reified T: Number> ndArrayOf(shape: List<Int>, elements: List<T>): NDArray<T> {
    return createLinearNdArray(dims = *shape.toIntArray(), filler = { elements[it] })
}