package nvidia.inferenceserver.client

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.options.default
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.options.required
import com.github.ajalt.clikt.parameters.types.int

open class ClientArguments : CliktCommand() {
    val model: String by option(help = "Model name").required()
    val modelVersion: Int by option(help = "Model version").int().default(-1)
    val batchSize: Int by option(help = "Batch size").int().default(1)
    val serverUrl: String by option(help = "Server url").required()

    override fun run() {

    }
}
