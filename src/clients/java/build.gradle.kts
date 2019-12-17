import com.google.protobuf.gradle.*

application {
    applicationName = "nvidia-trtis-java-client"
    group = "nvidia.inferenceserver"
    mainClassName = "nvidia.inferenceserver.client.MainKt"
}

plugins {
    val kotlinVersion = "1.3.60"
    java
    id("org.jetbrains.kotlin.jvm") version kotlinVersion
    id("com.google.protobuf") version "0.8.10"
    application
}

sourceSets {
    main {
        java {
            srcDirs("src/main/kotlin/")
            srcDirs("src/main/java/generated/")
        }
        proto {
            srcDir("../../core/")
            include("**/*.proto")
        }

    }
}

java {
    sourceCompatibility = JavaVersion.VERSION_1_6
    targetCompatibility = JavaVersion.VERSION_1_6
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.3.2")
    // implementation("org.jetbrains.kotlinx:kotlinx-coroutines-jdk8:1.3.2")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-guava:1.3.2")
    implementation("com.google.protobuf:protobuf-java:3.6.1")
    implementation("io.grpc:grpc-stub:1.15.1")
    implementation("io.grpc:grpc-protobuf:1.15.1")
    implementation("io.grpc:grpc-okhttp:1.25.0")
    implementation("com.github.ajalt:clikt:2.3.0")
    implementation("com.kyonifer:koma-core-ejml:0.12")
    implementation("io.github.microutils:kotlin-logging:1.7.7")
    implementation("org.slf4j:slf4j-simple:1.7.29")
}

repositories {
    mavenCentral()
    maven("https://dl.bintray.com/kyonifer/maven")
}

protobuf {
    generatedFilesBaseDir = "$projectDir/src/main/java/generated"
    protoc {
        artifact = "com.google.protobuf:protoc:3.6.1"
    }
    plugins {
        id("grpc") {
            artifact = "io.grpc:protoc-gen-grpc-java:1.15.1"
        }
    }
    generateProtoTasks {
        ofSourceSet("main").forEach {
            it.plugins {
                id("grpc")
            }
        }
    }
}

tasks {
    withType<Jar> {
        manifest {
            attributes(mapOf("Main-Class" to application.mainClassName))
        }
        val version = "1.0-SNAPSHOT"

        archiveName = "${application.applicationName}-$version.jar"
    }

    val copyDeps by registering(Copy::class) {
        from(configurations.runtimeClasspath)
        into("build/libs")
    }

}

tasks.getByName("build").dependsOn("copyDeps")