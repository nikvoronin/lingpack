using System.CommandLine;
using System.Globalization;
using LingPack.Core.Classification;
using LingPack.Core.Compression;
using LingPack.Core.ModelAssets;
using LingPack.Core.Tokenization;

namespace LingPack.Cli;

internal static class CompressCommandBuilder
{
    public static Command Build()
    {
        var inputOption = new Option<FileInfo?>("--input")
        {
            Description = "Path to the prompt file to compress. Omit to read from stdin.",
        };

        var outputOption = new Option<FileInfo?>("--output")
        {
            Description = "Path to write the compressed prompt to. Omit to write to stdout.",
        };

        var rateOption = new Option<double?>("--rate")
        {
            Description = "Fraction of words to keep (0-1). Mutually exclusive with --target-tokens.",
            // Parse with InvariantCulture (not the OS/current culture) so "0.5" works regardless of
            // the user's locale — some cultures (e.g. ru-RU) use ',' as the decimal separator.
            CustomParser = result =>
            {
                if (result.Tokens.Count == 0)
                {
                    return null;
                }

                var text = result.Tokens[0].Value;
                if (double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
                {
                    return value;
                }

                result.AddError($"Cannot parse '{text}' as --rate: expected a decimal number like 0.5 (using '.' as the decimal separator).");
                return null;
            },
        };

        var targetTokensOption = new Option<int?>("--target-tokens")
        {
            Description = "Absolute number of subword tokens to keep. Mutually exclusive with --rate.",
        };

        var modelDirOption = new Option<string?>("--model-dir")
        {
            Description = $"Directory containing the exported ONNX model + tokenizer assets. " +
                           $"Defaults to {ModelDirectoryResolver.EnvironmentVariableName} or " +
                           $"{ModelDirectoryResolver.DefaultModelDirectory}.",
        };

        var forceTokenOption = new Option<string[]>("--force-token")
        {
            Description = "A word that must always be kept, regardless of its score. Repeatable.",
            DefaultValueFactory = _ => [],
        };

        var chunkMaxTokensOption = new Option<int>("--chunk-max-tokens")
        {
            Description = "Model max sequence length used when chunking long inputs.",
            DefaultValueFactory = _ => 512,
        };

        var verboseOption = new Option<bool>("--verbose")
        {
            Description = "Print per-word scores and compression stats to stderr.",
        };

        var command = new Command("compress", "Compress a prompt using LLMLingua-2.")
        {
            inputOption,
            outputOption,
            rateOption,
            targetTokensOption,
            modelDirOption,
            forceTokenOption,
            chunkMaxTokensOption,
            verboseOption,
        };

        command.SetAction(async (parseResult, cancellationToken) =>
        {
            var input = parseResult.GetValue(inputOption);
            var output = parseResult.GetValue(outputOption);
            var rate = parseResult.GetValue(rateOption);
            var targetTokens = parseResult.GetValue(targetTokensOption);
            var modelDir = parseResult.GetValue(modelDirOption);
            var forceTokens = parseResult.GetValue(forceTokenOption) ?? [];
            var chunkMaxTokens = parseResult.GetValue(chunkMaxTokensOption);
            var verbose = parseResult.GetValue(verboseOption);

            if (rate is null && targetTokens is null)
            {
                Console.Error.WriteLine("Error: one of --rate or --target-tokens is required.");
                return 1;
            }

            if (rate is not null && targetTokens is not null)
            {
                Console.Error.WriteLine("Error: --rate and --target-tokens are mutually exclusive.");
                return 1;
            }

            ModelAssetPaths assetPaths;
            try
            {
                assetPaths = ModelDirectoryResolver.ResolveAndValidate(modelDir);
            }
            catch (FileNotFoundException ex)
            {
                Console.Error.WriteLine($"Error: {ex.Message}");
                return 1;
            }

            ITokenizerAdapter tokenizer = new XlmRobertaTokenizerAdapter(
                assetPaths.SentencePieceModelPath, assetPaths.TokenizerJsonPath, chunkMaxTokens);

            using var classifier = new OnnxTokenClassifier(assetPaths.OnnxModelPath);
            var compressor = new PromptCompressor(tokenizer, classifier);

            var text = await ConsoleIo.ReadInputAsync(input, cancellationToken);

            var options = new CompressionOptions
            {
                Rate = rate,
                TargetTokens = targetTokens,
                ForceKeepTokens = forceTokens,
                ChunkSafetyMargin = 8,
            };

            CompressionResult result;
            try
            {
                result = compressor.Compress(text, options);
            }
            catch (ArgumentException ex)
            {
                Console.Error.WriteLine($"Error: {ex.Message}");
                return 1;
            }

            await ConsoleIo.WriteOutputAsync(output, result.CompressedText, cancellationToken);

            if (verbose)
            {
                Console.Error.WriteLine(
                    $"Words: {result.KeptWordCount}/{result.OriginalWordCount} " +
                    $"({result.AchievedRate:P1}) kept; " +
                    $"Tokens: {result.CompressedTokenCount}/{result.OriginalTokenCount}");

                foreach (var score in result.WordScores)
                {
                    Console.Error.WriteLine($"  [{(score.Kept ? "keep" : "drop")}] {score.PreserveProbability:F3}  {score.Text}");
                }
            }

            return 0;
        });

        return command;
    }
}
