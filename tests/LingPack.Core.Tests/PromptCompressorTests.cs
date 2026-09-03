using LingPack.Core.Compression;
using LingPack.Core.Tests.Fakes;

namespace LingPack.Core.Tests;

public class PromptCompressorTests
{
    [Fact]
    public void Compress_WordScore_IsMeanOfItsTokenPreserveProbabilities()
    {
        var tokenizer = new FakeTokenizerAdapter(tokenCountPerWord: w => w == "alpha" ? 2 : 1);
        // sequence: BOS, alpha_t0, alpha_t1, beta_t0, EOS
        var classifier = new FakeTokenClassifier([0.5f, 0.9f, 0.7f, 0.2f, 0.5f]);
        var compressor = new PromptCompressor(tokenizer, classifier);

        var result = compressor.Compress("alpha beta", new CompressionOptions { Rate = 1.0 });

        Assert.Equal(2, result.WordScores.Count);
        Assert.Equal(0.8f, result.WordScores[0].PreserveProbability, precision: 5);
        Assert.Equal(0.2f, result.WordScores[1].PreserveProbability, precision: 5);
    }

    [Fact]
    public void Compress_RateSelection_KeepsTopScoringWordsAndPreservesOriginalOrder()
    {
        var tokenizer = new FakeTokenizerAdapter();
        // sequence: BOS, a, b, c, d, EOS
        var classifier = new FakeTokenClassifier([0f, 0.9f, 0.1f, 0.8f, 0.2f, 0f]);
        var compressor = new PromptCompressor(tokenizer, classifier);

        var result = compressor.Compress("a b c d", new CompressionOptions { Rate = 0.5 });

        Assert.Equal(2, result.KeptWordCount);
        Assert.Equal("a c", result.CompressedText);
    }

    [Fact]
    public void Compress_TargetTokensSelection_HonorsCumulativeTokenBudget()
    {
        // a costs 3 tokens, b costs 2, c costs 1. Ranked by score: a > b > c.
        var tokenizer = new FakeTokenizerAdapter(tokenCountPerWord: w => w switch { "a" => 3, "b" => 2, "c" => 1, _ => 1 });
        // sequence: BOS, a0,a1,a2, b0,b1, c0, EOS  (length 8)
        var classifier = new FakeTokenClassifier([0f, 0.9f, 0.9f, 0.9f, 0.5f, 0.5f, 0.1f, 0f]);
        var compressor = new PromptCompressor(tokenizer, classifier);

        // Budget 4: 'a' fits (3<=4), 'b' doesn't (3+2=5>4), 'c' fits (3+1=4<=4).
        var result = compressor.Compress("a b c", new CompressionOptions { TargetTokens = 4 });

        Assert.Equal("a c", result.CompressedText);
        Assert.Equal(4, result.CompressedTokenCount);
    }

    [Fact]
    public void Compress_ForceKeepToken_SurvivesEvenWithLowScore()
    {
        var tokenizer = new FakeTokenizerAdapter();
        // sequence: BOS, a, b, c, EOS
        var classifier = new FakeTokenClassifier([0f, 0.9f, 0.5f, 0.05f, 0f]);
        var compressor = new PromptCompressor(tokenizer, classifier);

        // rate 0.67 of 3 words -> round(2.01) = 2 words normally (a, b by score); force "c" in too.
        var result = compressor.Compress("a b c", new CompressionOptions
        {
            Rate = 0.67,
            ForceKeepTokens = ["c"],
        });

        var kept = result.WordScores.Where(s => s.Kept).Select(s => s.Text).ToArray();
        Assert.Contains("a", kept);
        Assert.Contains("c", kept);
        Assert.DoesNotContain("b", kept);
    }

    [Fact]
    public void Compress_Reassembly_ConsecutiveKeptWordsRetainOriginalWhitespace_DroppedGapUsesSingleSpace()
    {
        var tokenizer = new FakeTokenizerAdapter();
        // "a   b  c d" -> words: a, b(ws="   "), c(ws="  "), d(ws=" ")
        // sequence: BOS, a, b, c, d, EOS
        var classifier = new FakeTokenClassifier([0f, 0.9f, 0.8f, 0.1f, 0.7f, 0f]);
        var compressor = new PromptCompressor(tokenizer, classifier);

        var result = compressor.Compress("a   b  c d", new CompressionOptions { Rate = 0.75 });

        Assert.Equal(3, result.KeptWordCount);
        Assert.Equal("a   b d", result.CompressedText);
    }

    [Fact]
    public void Compress_ScoresAndOrder_AreCorrectAcrossAChunkSeam()
    {
        // maxModelLength=6, safety margin=2 -> budget=2 words/chunk -> 2 chunks of 2 words each.
        var tokenizer = new FakeTokenizerAdapter(maxModelLength: 6);
        var classifier = new FakeTokenClassifier(
            [0f, 0.9f, 0.2f, 0f],  // chunk 1: words "a","b"
            [0f, 0.3f, 0.95f, 0f]); // chunk 2: words "c","d"
        var compressor = new PromptCompressor(tokenizer, classifier);

        var result = compressor.Compress("a b c d", new CompressionOptions { Rate = 0.5, ChunkSafetyMargin = 2 });

        // Ranked by score desc: d(0.95), a(0.9), c(0.3), b(0.2) -> keep top 2: d, a.
        Assert.Equal(2, result.KeptWordCount);
        Assert.Equal("a d", result.CompressedText); // original order preserved despite cross-chunk ranking
        Assert.Equal(2, classifier.Calls.Count);
    }

    [Fact]
    public void Compress_EmptyInput_ReturnsEmptyResultWithoutInvokingClassifier()
    {
        var tokenizer = new FakeTokenizerAdapter();
        var classifier = new FakeTokenClassifier();
        var compressor = new PromptCompressor(tokenizer, classifier);

        var result = compressor.Compress("   ", new CompressionOptions { Rate = 0.5 });

        Assert.Equal(0, result.OriginalWordCount);
        Assert.Empty(classifier.Calls);
    }

    [Theory]
    [InlineData(null, null)]
    [InlineData(0.5, 10)]
    public void Compress_InvalidRateTargetTokensCombination_Throws(double? rate, int? targetTokens)
    {
        var tokenizer = new FakeTokenizerAdapter();
        var classifier = new FakeTokenClassifier();
        var compressor = new PromptCompressor(tokenizer, classifier);

        Assert.Throws<ArgumentException>(() =>
            compressor.Compress("a b", new CompressionOptions { Rate = rate, TargetTokens = targetTokens }));
    }
}
