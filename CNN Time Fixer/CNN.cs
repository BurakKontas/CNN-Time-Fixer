using BenchmarkDotNet.Attributes;
using CNN_Time_Fixer.Aspects;
using Microsoft.ML;
using Microsoft.ML.Vision;
using Tensorflow;

namespace CNN_Time_Fixer;

public class CNN
{
    private readonly PredictionEngine<ImageModelInput, ImagePrediction> predictionEngine;
    private readonly MLContext context;

    public CNN(string modelPath)
    {
        context = new MLContext();
        context.FallbackToCpu = false;

        ITransformer loadedModel;

        using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            loadedModel = context.Model.Load(stream, out _);
        }

        predictionEngine = context.Model.CreatePredictionEngine<ImageModelInput, ImagePrediction>(loadedModel);

    }

    public CNN() : this(Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "dnn_model.zip"))
    {
    }

    public string Train()
    {
        var imagesFolder = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "images");

        var files = Directory.GetFiles(imagesFolder, "*", SearchOption.AllDirectories);

        var images = files.Select(file => new ImageData
        {
            ImagePath = file,
            Label = Directory.GetParent(file)!.Name
        });

        var imageData = context.Data.LoadFromEnumerable(images);
        var imageDataShuffled = context.Data.ShuffleRows(imageData);

        var testTrainData = context.Data.TrainTestSplit(imageDataShuffled, testFraction: 0.2);

        var validationData = context.Transforms.Conversion.MapValueToKey("LabelKey", "Label", keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
            .Append(context.Transforms.LoadRawImageBytes("Image", imagesFolder, "ImagePath"))
            .Fit(testTrainData.TestSet)
            .Transform(testTrainData.TestSet);

        var imagesPipeline = context.Transforms.Conversion.MapValueToKey("LabelKey", "Label", keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
            .Append(context.Transforms.LoadRawImageBytes("Image", imagesFolder, "ImagePath"));

        var imageDataModel = imagesPipeline.Fit(testTrainData.TrainSet);

        var imageDataView = imageDataModel.Transform(testTrainData.TrainSet);

        var options = new ImageClassificationTrainer.Options()
        {
            Arch = ImageClassificationTrainer.Architecture.ResnetV250,
            Epoch = 100,
            BatchSize = 20,
            LearningRate = 0.01f,
            LabelColumnName = "LabelKey",
            FeatureColumnName = "Image",
            ValidationSet = validationData,
        };

        var pipeline = context.MulticlassClassification.Trainers.ImageClassification(options)
            .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        var model = pipeline.Fit(imageDataView);

        var testImagesFolder = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "test");

        var testFiles = Directory.GetFiles(testImagesFolder, "*", SearchOption.AllDirectories);

        var testImages = testFiles.Select(file => new ImageModelInput
        {
            ImagePath = file,
            Label = Directory.GetParent(file)!.Name
        });

        Console.WriteLine(Environment.NewLine);

        var testImagesData = context.Data.LoadFromEnumerable(testImages);

        var testImageDataView = imagesPipeline.Fit(testImagesData).Transform(testImagesData);

        var predictions = model.Transform(testImageDataView);

        var testPredictions = context.Data.CreateEnumerable<ImagePrediction>(predictions, reuseRowObject: false);

        foreach (var prediction in testPredictions)
        {
            var labelIndex = prediction.PredictedLabel;

            Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)}, Predicted Label: {prediction.PredictedLabel}");
        }

        var dnnFile = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "dnn_model.zip");

        context.Model.Save(model, imageData.Schema, dnnFile);

        return dnnFile;
    }

    [Benchmark]
    //[TimeFixing(20)]
    public string Predict()
    {
        var testImagePath = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "test", "cat", "cat (1).jpeg");

        var imageInput = new ImageModelInput
        {
            ImagePath = testImagePath,
        };

        var prediction = predictionEngine.Predict(imageInput);

        Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)}, Predicted Label: {prediction.PredictedLabel}");

        return prediction.PredictedLabel;
    }
}