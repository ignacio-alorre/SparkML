
![pcaex1](https://user-images.githubusercontent.com/20033386/46548188-5b4f8380-c8df-11e8-8ba9-5f5322e4cbb5.jpg)

```scala
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.PCA

//udf to get items from a collection
val getItem = udf((v: Vector, i: Int) => v(i))

//Using Vector Assembler to group to group several columns into a vector
val assemblerDepTime = new VectorAssembler()
                .setInputCols(Array("deplocaltime", "depquarterofday", "depdaynight"))
                .setOutputCol("depTimeVect")

val assemblerArrTime = new VectorAssembler()
                .setInputCols(Array("arrlocaltime", "arrquarterofday", "arrdaynight"))
                .setOutputCol("arrTimeVect") 


//Applying the PCA over the Vector Assemblers
val pcaDepTime = new PCA()
  .setInputCol("depTimeVect")
  .setOutputCol("depTimePCA")
  .setK(3)

val pcaArrTime = new PCA()
  .setInputCol("arrTimeVect")
  .setOutputCol("arrTimePCA")
  .setK(3)

//Grouping the VectorAssemblers and PCA into an array of steps
val stepsAssembler = Array(assemblerDepTime,assemblerArrTime,pcaDepTime,pcaArrTime)

//Creating a pipeline with the previous defined steps
val pipelineAssembler = new Pipeline()
                     .setStages(stepsAssembler)

//Fitting the pipeline on the dataset
val finalPipeline = pipelineAssembler.fit(details_actuals_delays_cleansedFeteng2)

//Using the pipeline to transform the dataset
val details_actuals_delays_cleansedFeteng3 = finalPipeline.transform(details_actuals_delays_cleansedFeteng2)
.withColumn("deplocaltimePCA",getItem($"depTimePCA",lit(0)))
.withColumn("depquarterofdayPCA",getItem($"depTimePCA",lit(1)))
.withColumn("depdaynightPCA",getItem($"depTimePCA",lit(2)))
.withColumn("arrlocaltimePCA",getItem($"arrTimePCA",lit(0)))
.withColumn("arrquarterofdayPCA",getItem($"arrTimePCA",lit(1)))
.withColumn("arrdaynightPCA",getItem($"arrTimePCA",lit(2)))

display(details_actuals_delays_cleansedFeteng3.select("deplocaltime", "depquarterofday", "depdaynight","deplocaltimePCA","depquarterofdayPCA","depdaynightPCA"))
```


