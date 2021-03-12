# Example Java and Scala client Using Generated GRPC API


## Prerequsites
Maven 3.3+ and JDK 1.8+

## Generating java GRPC client stub
Copy __*.proto__ files to Library src/main/proto
```
$ cd library
$ cp ../../../core/*.proto src/main/proto/
```

__Note__ the *.proto files are copied from [triton-inference-server
/server/src/core/](https://github.com/triton-inference-server/server/tree/master/src/core). After copying the __libray__ dir should look as below.


<img src="images/proto-files.png" width="220" />

```
$ mvn compile
```
Once compiled, one should notice the generated *.java files under __target__ folder

<img src="images/grpc-stubs.png" width="400" />

## Use the generated files in any project

To run the examples clients, copy the above generated stub into __examples__ folder

```
$ cd ..

$ cp -R library/target/generated-sources/protobuf/java/inference  examples/src/main/java/inference

$ cp -R library/target/generated-sources/protobuf/grpc-java/inference/*.java  examples/src/main/java/inference/

```
See the __examples__ project which has __scala__ and __java__ sample client. 

## Running java example client 

```
$ cd examples

$ mvn clean install 

$ mvn exec:java -Dexec.mainClass=clients.SimpleJavaClient -Dexec.args="<host> <port>"
```

__host__  where triton inference server is running

__port__ default grpc port is 8001

## Running scala example client 

```
$ mvn exec:java -Dexec.mainClass=clients.SimpleClient -Dexec.args="<host> <port>"
```

Both the examples run inference with respect to __simple__ model. The __scala__ example is more comprehensive and checks APIs like server ready and model ready

### Output of the scala client once run succesfully

```
name: "OUTPUT0"
datatype: "INT32"
shape: 1
shape: 16

name: "OUTPUT1"
datatype: "INT32"
shape: 1
shape: 16

1 + 1 = 2
1 - 1 = 0
2 + 2 = 4
2 - 2 = 0
3 + 3 = 6
3 - 3 = 0
4 + 4 = 8
4 - 4 = 0
5 + 5 = 10
5 - 5 = 0
6 + 6 = 12
6 - 6 = 0
7 + 7 = 14
7 - 7 = 0
8 + 8 = 16
8 - 8 = 0
9 + 9 = 18
9 - 9 = 0
10 + 10 = 20
10 - 10 = 0
11 + 11 = 22
11 - 11 = 0
12 + 12 = 24
12 - 12 = 0
13 + 13 = 26
13 - 13 = 0
14 + 14 = 28
14 - 14 = 0
15 + 15 = 30
15 - 15 = 0
16 + 16 = 32
16 - 16 = 0
```