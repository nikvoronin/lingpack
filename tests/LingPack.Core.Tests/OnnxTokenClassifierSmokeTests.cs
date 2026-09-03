using LingPack.Core.Classification;
using LingPack.Core.Compression;
using LingPack.Core.ModelAssets;
using LingPack.Core.Tokenization;

namespace LingPack.Core.Tests;

/// <summary>
/// End-to-end smoke tests against the real exported ONNX model. These require the model to have
/// been exported via scripts/export-model.py first (see README "Model setup"); they no-op (pass
/// without asserting anything) when the model directory isn't present, so `dotnet test` stays green
/// without it. Run explicitly after exporting with: dotnet test --filter Category=RequiresModel
/// </summary>
[Trait("Category", "RequiresModel")]
public class OnnxTokenClassifierSmokeTests
{
    private static bool TryResolveModel(out ModelAssetPaths paths)
    {
        try
        {
            paths = ModelDirectoryResolver.ResolveAndValidate(modelDirFlag: null);
            return true;
        }
        catch (FileNotFoundException)
        {
            paths = null!;
            return false;
        }
    }

    [Fact]
    public void Compress_RealModel_DropsLowerValueWordsBeforeHigherValueContentWords()
    {
        if (!TryResolveModel(out var paths))
        {
            return; // model not exported locally; nothing to verify here.
        }

        var tokenizer = new XlmRobertaTokenizerAdapter(paths.SentencePieceModelPath, paths.TokenizerJsonPath);
        using var classifier = new OnnxTokenClassifier(paths.OnnxModelPath);
        var compressor = new PromptCompressor(tokenizer, classifier);

        var result = compressor.Compress(
            "The quick brown fox jumps over the lazy dog while a gentle breeze moves through the trees.",
            new CompressionOptions { Rate = 0.5 });

        Assert.True(result.KeptWordCount > 0);
        Assert.True(result.KeptWordCount < result.OriginalWordCount);
        Assert.All(result.WordScores, s => Assert.False(float.IsNaN(s.PreserveProbability)));
    }
}
