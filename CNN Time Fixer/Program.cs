using CNN_Time_Fixer;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args, new DebugInProcessConfig());

//var modelPath = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "dnn_model.zip");

//var cnn = new CNN(modelPath);

//cnn.Train();