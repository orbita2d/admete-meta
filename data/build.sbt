val scala3Version = "3.7.0"

lazy val root = project
  .in(file("."))
  .enablePlugins(JavaAppPackaging) 
  .settings(
    name := "admete-nn-data",
    version := "0.1.0-SNAPSHOT",

    scalaVersion := scala3Version,
    javacOptions ++= Seq("-source", "17", "-target", "17"),

    Compile / mainClass := Some("processZst"),

    resolvers += "jitpack" at "https://jitpack.io",
    

    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % "1.0.0" % Test,
      "com.github.luben" % "zstd-jni" % "1.5.5-11",
      "com.lihaoyi" %% "os-lib" % "0.9.1",
      "co.fs2" %% "fs2-core" % "3.9.2",
      "co.fs2" %% "fs2-io" % "3.9.2",
      "com.github.bhlangonijr" % "chesslib" % "1.3.4",
      "com.google.guava" % "guava" % "32.1.2-jre",
      "com.typesafe.akka" %% "akka-stream" % "2.8.5",
      "com.typesafe.akka" %% "akka-actor-typed" % "2.8.5",
      "ch.qos.logback" % "logback-classic" % "1.4.14",
      "com.typesafe.akka" %% "akka-slf4j" % "2.8.0",
      "com.github.mjakubowski84" %% "parquet4s-core" % "2.22.0",
      "com.github.mjakubowski84" %% "parquet4s-akka" % "2.22.0",
      "org.apache.hadoop" % "hadoop-client" % "3.3.4",
      "org.apache.hadoop" % "hadoop-common" % "3.3.4",
      "net.java.dev.jna" % "jna" % "5.13.0",
      "net.java.dev.jna" % "jna-platform" % "5.13.0",
    ),
  )